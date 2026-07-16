import torch
import torch.nn as nn

from .attention import LaplaceAttention, MultiHeadCrossAttentionWrapper
from .positional_encoding import SpatialPositionalEncoding, MultiScalePositionalEncoding

def _create_mlp(input_dim, output_dim, hidden_size, num_hidden_layers, dropout_rate, final_activation=None):
    """Helper function to create a multi-layer perceptron."""
    layers = []
    if num_hidden_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, output_dim))

    if final_activation:
        layers.append(final_activation)
    return nn.Sequential(*layers)

class MeanLocationEncoder(nn.Module):

    def __init__(self, spatial_dim, y_dim, hidden_size, w_dim, dropout_rate, num_layers_mlp=1,
                 laplace_p=1.0):
        super().__init__()
        self.mlp_pre_attn = _create_mlp(spatial_dim + y_dim, w_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.laplace_attention = LaplaceAttention(p_norm=laplace_p)
    
    def forward(self, s_context, y_context, s_target):
        # w = MLP(s, y)
        sy_context = torch.cat([s_context, y_context], dim=-1)
        w = self.mlp_pre_attn(sy_context)
        
        # Laplace attention: query=s*, key=s, value=w
        return self.laplace_attention(query=s_target, key=s_context, value=w)
    

class GlobalMeanAttributeEncoder(nn.Module):
   
    def __init__(self, input_dim, y_dim, hidden_size, r_dim, num_heads, dropout_rate, 
                 num_layers_mlp=1, context_input_dim=None, use_positional_encoding=False, pe_dim=64, pe_type='sinusoidal', 
                 pe_scales=None, pe_scale_dims=None, pe_coordinate_range='normalized', 
                 pe_original_bounds=None, pe_learnable=False, use_layer_norm=True, **kwargs):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        # Initialize positional encoding
        if use_positional_encoding:
            if pe_type == 'multi_scale':
                if pe_scale_dims is None:
                    num_scales = len(pe_scales) if pe_scales else 3
                    base_dim = pe_dim // num_scales
                    remainder = pe_dim % num_scales
                    pe_scale_dims = [base_dim] * num_scales
                    if remainder > 0:
                        pe_scale_dims[-1] += remainder
                
                self.pos_encoder = MultiScalePositionalEncoding(
                    spatial_dim=2,
                    encoding_dims=pe_scale_dims,
                    scales=pe_scales,
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds,
                    learnable=pe_learnable
                )
                actual_pe_dim = sum(pe_scale_dims)
            else:
                self.pos_encoder = SpatialPositionalEncoding(
                    spatial_dim=2,
                    encoding_dim=pe_dim,
                    learnable=pe_learnable,
                    use_gaussian=(pe_type == 'gaussian'),
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds
                )
                actual_pe_dim = pe_dim
            
            actual_input_dim = input_dim + actual_pe_dim
        else:
            actual_input_dim = input_dim

        _pe_extra = actual_pe_dim if use_positional_encoding else 0
        actual_context_input_dim = (context_input_dim + _pe_extra
                                    if context_input_dim is not None
                                    else actual_input_dim)        
        
        # Value encoding: r = MLP(x, y) - includes target values
        self.mlp_value = _create_mlp(actual_context_input_dim + y_dim, r_dim, hidden_size, 
                                     num_layers_mlp, dropout_rate)
        
        # Key encoding: MLP(x) - attributes ONLY, NO target values!
        self.mlp_key = _create_mlp(actual_context_input_dim, r_dim, hidden_size, 
                                   num_layers_mlp, dropout_rate)
        
        # Query encoding: MLP(x*) - target attributes
        self.mlp_query = _create_mlp(actual_input_dim, r_dim, hidden_size, 
                                     num_layers_mlp, dropout_rate)
        
        self.cross_attention = MultiHeadCrossAttentionWrapper(r_dim, num_heads, dropout_rate)
        
        if use_layer_norm:
            self.key_norm = nn.LayerNorm(r_dim)
            self.value_norm = nn.LayerNorm(r_dim)
            self.query_norm = nn.LayerNorm(r_dim)
            self.output_norm = nn.LayerNorm(r_dim)
    
    def forward(self, x_context_all, y_context, x_target_all, s_context=None, s_target=None, **kwargs):
        # Add positional encoding if enabled
        if self.use_positional_encoding and s_context is not None:
            pe_context = self.pos_encoder(s_context)
            pe_target = self.pos_encoder(s_target)
            x_context_enhanced = torch.cat([x_context_all, pe_context], dim=-1)
            x_target_enhanced = torch.cat([x_target_all, pe_target], dim=-1)
        else:
            x_context_enhanced = x_context_all
            x_target_enhanced = x_target_all
        
        
        # Value: r = MLP(x, y) - includes target values
        xy_context = torch.cat([x_context_enhanced, y_context], dim=-1)
        r_value = self.mlp_value(xy_context)
        
        # Key: MLP(x) - NO target values! Pure attribute-based similarity
        r_key = self.mlp_key(x_context_enhanced)
        
        # Query: MLP(x*) - target attributes
        r_query = self.mlp_query(x_target_enhanced)
        
        if self.use_layer_norm:
            r_value = self.value_norm(r_value)
            r_key = self.key_norm(r_key)
            r_query = self.query_norm(r_query)
        
        # Cross attention with SEPARATE key and value
        output = self.cross_attention(query=r_query, key=r_key, value=r_value)
        
        if self.use_layer_norm:
            output = self.output_norm(output)
        
        return output


class GlobalVarianceEncoder(nn.Module):
    """
    PAPER-FAITHFUL: Variance encoder.
    
    From paper Section 3.1:
    - Variance is INDEPENDENT of target values y
    - v = MLP_φ(s, x) - encodes spatial + attributes
    - v* = Multi-head attention(v, s*, x*)
    """
    def __init__(self, input_dim, hidden_size, v_dim, num_heads, dropout_rate, 
                 num_layers_mlp=1, context_input_dim=None, use_positional_encoding=False, pe_dim=64, pe_type='sinusoidal', 
                 pe_scales=None, pe_scale_dims=None, pe_coordinate_range='normalized', 
                 pe_original_bounds=None, pe_learnable=False, use_layer_norm=True, spatial_dim=2, **kwargs):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        # Initialize positional encoding
        if use_positional_encoding:
            if pe_type == 'multi_scale':
                if pe_scale_dims is None:
                    num_scales = len(pe_scales) if pe_scales else 3
                    base_dim = pe_dim // num_scales
                    remainder = pe_dim % num_scales
                    pe_scale_dims = [base_dim] * num_scales
                    if remainder > 0:
                        pe_scale_dims[-1] += remainder
                
                self.pos_encoder = MultiScalePositionalEncoding(
                    spatial_dim=2,
                    encoding_dims=pe_scale_dims,
                    scales=pe_scales,
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds,
                    learnable=pe_learnable
                )
                actual_pe_dim = sum(pe_scale_dims)
            else:
                self.pos_encoder = SpatialPositionalEncoding(
                    spatial_dim=2,
                    encoding_dim=pe_dim,
                    learnable=pe_learnable,
                    use_gaussian=(pe_type == 'gaussian'),
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds
                )
                actual_pe_dim = pe_dim
            
            actual_input_dim = input_dim + actual_pe_dim
        else:
            actual_input_dim = input_dim
        
        # Include spatial dimension in input (variance encoder uses s + x)
        actual_input_dim += spatial_dim

        _pe_extra = actual_pe_dim if use_positional_encoding else 0
        if context_input_dim is not None:
            actual_context_input_dim = context_input_dim + _pe_extra + spatial_dim
        else:
            actual_context_input_dim = actual_input_dim
        
        # v = MLP(s, x) for both key and value (no y involved anyway)
        self.mlp_kv = _create_mlp(actual_context_input_dim, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        
        # Query: MLP(s*, x*)
        self.mlp_query = _create_mlp(actual_input_dim, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        
        self.cross_attention = MultiHeadCrossAttentionWrapper(v_dim, num_heads, dropout_rate)
        
        if use_layer_norm:
            self.kv_norm = nn.LayerNorm(v_dim)
            self.query_norm = nn.LayerNorm(v_dim)
            self.output_norm = nn.LayerNorm(v_dim)
    
    def forward(self, x_context_all, x_target_all, s_context=None, s_target=None, **kwargs):
        # Add positional encoding if enabled
        if self.use_positional_encoding and s_context is not None:
            pe_context = self.pos_encoder(s_context)
            pe_target = self.pos_encoder(s_target)
            x_context_enhanced = torch.cat([x_context_all, pe_context], dim=-1)
            x_target_enhanced = torch.cat([x_target_all, pe_target], dim=-1)
        else:
            x_context_enhanced = x_context_all
            x_target_enhanced = x_target_all
        
        # Concatenate spatial coords (variance uses s + x)
        x_context_enhanced = torch.cat([s_context, x_context_enhanced], dim=-1)
        x_target_enhanced = torch.cat([s_target, x_target_enhanced], dim=-1)
        
        # v = MLP(s, x) - no y involved
        v_kv = self.mlp_kv(x_context_enhanced)
        v_query = self.mlp_query(x_target_enhanced)
        
        if self.use_layer_norm:
            v_kv = self.kv_norm(v_kv)
            v_query = self.query_norm(v_query)
        
        output = self.cross_attention(query=v_query, key=v_kv, value=v_kv)
        
        if self.use_layer_norm:
            output = self.output_norm(output)
        
        return output
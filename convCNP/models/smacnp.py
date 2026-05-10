import torch
import torch.nn as nn

from .smacnp_encoders import (MeanLocationEncoder,
                               GlobalMeanAttributeEncoder,
                               GlobalVarianceEncoder)
from .smacnp_decoders import MeanDecoder, VarianceDecoder

def separate_spatial_attributes(x, spatial_indices):
    """Split (B, N, D) into spatial coords and non-spatial attributes."""
    attr_indices = [i for i in range(x.shape[-1]) if i not in spatial_indices]
    return x[..., spatial_indices], x[..., attr_indices]


class SMACNP_ALL(nn.Module):
    def __init__(self, input_dim, spatial_dim, attr_dim, y_dim,
                 r_dim, w_dim, v_dim, hidden_size,
                 num_heads, dropout_rate, spatial_indices,
                 num_layers_mlp=1,
                 laplace_distance_p=1.0,
                 
                 # Layer Normalization Control
                 use_layer_norm=True,
                 
                 # Positional Encoding Controls (Complete)
                 use_positional_encoding=False,
                 pe_dim=64,
                 pe_type='sinusoidal',
                 pe_scales=None,
                 pe_scale_dims=None,
                 pe_coordinate_range='normalized',
                 pe_original_bounds=None,
                 pe_learnable=False,
                 
                 # Decoder Controls
                 gat_hidden_channels=16, gat_heads=4,
                 min_points_for_graph=10, initial_alpha_bias=-1.0,
                 **kwargs):
        
        super().__init__()
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        self.attr_dim = attr_dim
        self.y_dim = y_dim
        self.spatial_indices = spatial_indices
        self.laplace_distance_p = laplace_distance_p
        
        # Store configuration flags
        self.use_layer_norm = use_layer_norm
        self.use_positional_encoding = use_positional_encoding

    
        if input_dim != spatial_dim + attr_dim:
            raise ValueError(
                f"input_dim ({input_dim}) must equal spatial_dim + attr_dim ({spatial_dim + attr_dim})"
            )

        # Decoder always consumes attribute-only tensors
        encoder_attr_dim = attr_dim

        encoder_input_dim = attr_dim


        # --- 1. Mean-Location Encoder (Laplace Attention) ---
        # "w = MLP(s, y)" -> Always same structure
        self.mean_loc_encoder = MeanLocationEncoder(
            spatial_dim, y_dim, hidden_size, w_dim, dropout_rate, num_layers_mlp,
            laplace_p=self.laplace_distance_p
        )
        
        # --- 2. Mean-Attribute & Variance Encoders ---
        encoder_kwargs = {
            'dropout_rate': dropout_rate,
            'num_layers_mlp': num_layers_mlp,
            'use_positional_encoding': use_positional_encoding,
            'pe_dim': pe_dim,
            'pe_type': pe_type,
            'pe_scales': pe_scales,
            'pe_scale_dims': pe_scale_dims,
            'pe_coordinate_range': pe_coordinate_range,
            'pe_original_bounds': pe_original_bounds,
            'pe_learnable': pe_learnable,
        }
        
        self.mean_attr_encoder = GlobalMeanAttributeEncoder(
            encoder_input_dim, y_dim, hidden_size, r_dim, num_heads,
            use_layer_norm=use_layer_norm, **encoder_kwargs
        )
        self.variance_encoder = GlobalVarianceEncoder(
            encoder_input_dim, hidden_size, v_dim, num_heads,
            use_layer_norm=use_layer_norm, spatial_dim=spatial_dim, **encoder_kwargs
        )
        
        # --- 3. Decoders ---
        # Base local decoders
        local_temp_mean_decoder = MeanDecoder(r_dim, w_dim, spatial_dim, encoder_attr_dim, 
                                              hidden_size, 1, dropout_rate, num_layers_mlp)
        local_temp_variance_decoder = VarianceDecoder(v_dim, spatial_dim, encoder_attr_dim, 
                                                      hidden_size, 1, dropout_rate, num_layers_mlp)
        
        self.temp_mean_decoder = local_temp_mean_decoder
        self.temp_variance_decoder = local_temp_variance_decoder

    def forward(self, x_context_all, y_context, x_target_all):
        """
        Forward pass with robust handling of spatial vs attribute data.
        """
        # 1. Separate Spatial (s) and Attributes (x_attr)
        s_context, x_attr_context = separate_spatial_attributes(x_context_all, self.spatial_indices)
        s_target, x_attr_target = separate_spatial_attributes(x_target_all, self.spatial_indices)

        # 3. Prepare Inputs for Encoders
        
        x_context_input = x_attr_context
        x_target_input = x_attr_target

        # 5. Execute Encoders
        
        # A. Mean-Location Encoder (Laplace: s, y)
        w_star = self.mean_loc_encoder(s_context, y_context, s_target)
        
        # B. Mean-Attribute Encoder (Attention: x -> r)
        # Note: We pass all possible args; specific encoders pick what they need
        encoder_kwargs = {
            'x_context_all': x_context_input,
            'y_context': y_context,
            'x_target_all': x_target_input,
            's_context': s_context,
            's_target': s_target,
        }
        
        r_star = self.mean_attr_encoder(**encoder_kwargs)
        
        # C. Variance Encoder (Attention: s+x -> v)
        variance_kwargs = {
            'x_context_all': x_context_input, # Or attr only, depending on class
            'x_target_all': x_target_input,
            's_context': s_context,
            's_target': s_target,
        }
        
        v_star = self.variance_encoder(**variance_kwargs)

        # 6. Execute Decoders
        # Decoders take: r*, w*, s_target, and target_attributes
        temp_mu = self.temp_mean_decoder(r_star, w_star, s_target, x_attr_target)
        temp_variance = self.temp_variance_decoder(v_star, s_target, x_attr_target)
        
        temp_mu    = self.temp_mean_decoder(r_star, w_star, s_target, x_attr_target)   # (B, M, 1)
        temp_sigma = self.temp_variance_decoder(v_star, s_target, x_attr_target)        # (B, M, 1) > 0

        return torch.cat([temp_mu, temp_sigma], dim=-1)   # (B, M, 2) — matches gll and get_value_tmax
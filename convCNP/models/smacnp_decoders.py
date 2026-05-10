import torch
import torch.nn as nn

from .smacnp_encoders import _create_mlp

class MeanDecoder(nn.Module):
    """Decoder for mean predictions, combining r* and w*."""
    def __init__(self, r_dim, w_dim, spatial_dim, attr_dim, hidden_size, y_dim, dropout_rate, 
                 num_layers_mlp=1, final_activation=None):
        super().__init__()
        input_decoder_dim = r_dim + w_dim + spatial_dim + attr_dim
        self.mlp_decoder = _create_mlp(input_decoder_dim, y_dim, hidden_size, num_layers_mlp, 
                                       dropout_rate, final_activation)
    
    def forward(self, r_star, w_star, s_target, x_attr_target):
        decoder_input = torch.cat([r_star, w_star, s_target, x_attr_target], dim=-1)
        return self.mlp_decoder(decoder_input)


class VarianceDecoder(nn.Module):
    """Decoder for variance predictions."""
    def __init__(self, v_dim, spatial_dim, attr_dim, hidden_size, y_dim, dropout_rate, num_layers_mlp=1):
        super().__init__()
        input_decoder_dim = v_dim + spatial_dim + attr_dim
        self.mlp_decoder = _create_mlp(input_decoder_dim, y_dim, hidden_size, num_layers_mlp, 
                                       dropout_rate, nn.Softplus())
    
    def forward(self, v_star, s_target, x_attr_target):
        decoder_input = torch.cat([v_star, s_target, x_attr_target], dim=-1)
        return self.mlp_decoder(decoder_input) + 1e-6
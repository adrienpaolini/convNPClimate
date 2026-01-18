"""
Downscaling convCNPs including MLP for elevation data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .mlp import MLP
from .final_layers import GaussianFinalLayer, GammaFinalLayer, GammaGPFinalLayer
from .cnn import CNN, ResConvBlock
from .utils import force_positive


class TmaxBiasConvCNPElev(nn.Module):
    """
    Bias correction for temperature, including MLP for elevation and seasonal features.

    Parameters:
    ----------
    decoder: convolutional architecture
    in_channels: number of input channels
    ls: length scale for final layer
    use_seasonal: whether to include seasonal features (cos/sin of day-of-year)
    """

    def __init__(self, decoder, in_channels=1, ls=0.1, use_seasonal=True):

        super().__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = decoder
        self.use_seasonal = use_seasonal

        self.mlp = MLP(decoder.out_channels, 2,
            hidden_channels=64,
            hidden_layers=4)

        # Elevation MLP input: 2 (mu, sigma) + 3 (elev features) + 2 (seasonal) = 7
        # Or without seasonal: 2 + 3 = 5
        elev_mlp_in_channels = 7 if use_seasonal else 5
        self.elev_mlp = MLP(elev_mlp_in_channels, 2,
            hidden_channels=64,
            hidden_layers=4)

        self.activation_function = torch.relu
        self.final_layer = GaussianFinalLayer(0.1, 2)


    def forward(self, x, mask, dists, elev, seasonal=None):
        """
        Forward pass.

        Parameters:
        -----------
        x: input tensor (batch, channels, lat, lon)
        mask: context mask
        dists: distances tensor
        elev: elevation features (n_points, 3) - [true_elev, elev_diff, tpi]
        seasonal: seasonal features (batch, 2) - [cos_doy, sin_doy], or None
        """
        # Encode with set convolution
        x = self.encoder(x, mask)
        x = self.activation_function(x)
        # Decode with CNN
        x = self.decoder(x)
        x = self.activation_function(x)
        # MLP
        x = self.mlp(x)

        mu, sigma = self.final_layer(x, dists)

        out = torch.cat([mu.view(*mu.shape, 1),
            sigma.view(*sigma.shape, 1)], dim=2)

        # Do elevation (and seasonal)
        batch_size = out.shape[0]
        n_points = out.shape[1]

        elev_expanded = elev.repeat(batch_size, 1, 1)  # (batch, n_points, 3)

        if self.use_seasonal and seasonal is not None:
            # seasonal is (batch, 2), expand to (batch, n_points, 2)
            seasonal_expanded = seasonal.unsqueeze(1).expand(-1, n_points, -1)
            out = torch.cat([out, elev_expanded, seasonal_expanded], dim=2)
        else:
            out = torch.cat([out, elev_expanded], dim=2)

        out = self.elev_mlp(out)
        out[..., 1] = force_positive(out[..., 1])

        return out

class GammaBiasConvCNPElev(nn.Module):
    """
    Bias correction for precipitation, including MLP for elevation and seasonal features.

    Parameters:
    ----------
    decoder: convolutional architecture
    in_channels: number of input channels
    ls: length scale for final layer
    use_seasonal: whether to include seasonal features (cos/sin of day-of-year)
    """

    def __init__(self,
                 decoder,
                 in_channels=1,
                 ls=0.1,
                 use_seasonal=True):
        super().__init__()
        self.in_channels = in_channels
        self.activation = torch.relu
        self.sigmoid = nn.Sigmoid()
        self.use_seasonal = use_seasonal

        self.encoder = Encoder(in_channels=in_channels)
        self.mlp = MLP(in_channels=128,
            out_channels=3,
            hidden_channels=64,
            hidden_layers=4)
        self.decoder = decoder
        self.out_layer = GammaFinalLayer(
            init_ls=ls,
            n_params=3
        )

        # Elevation MLP input: 3 (rho, alpha, beta) + 3 (elev features) + 2 (seasonal) = 8
        # Or without seasonal: 3 + 3 = 6
        elev_mlp_in_channels = 8 if use_seasonal else 6
        self.elev_mlp = MLP(elev_mlp_in_channels,
            out_channels=3,
            hidden_channels=64,
            hidden_layers=4)

    def forward(self, h, mask, dists, elev, seasonal=None):
        """
        Forward pass.

        Parameters:
        -----------
        h: input tensor (batch, channels, lat, lon)
        mask: context mask
        dists: distances tensor
        elev: elevation features (n_points, 3) - [true_elev, elev_diff, tpi]
        seasonal: seasonal features (batch, 2) - [cos_doy, sin_doy], or None
        """
        # Encode with set convolution
        h = self.activation(self.encoder(h, mask))
        # Decode with CNN
        h = self.activation(self.decoder(h))
        # MLP
        h = self.mlp(h)
        # out layer
        rho, alpha, beta = self.out_layer(h, dists)
        out = torch.cat([rho.view(*rho.shape, 1),
            alpha.view(*alpha.shape, 1),
            beta.view(*beta.shape, 1)], dim=2)

        # Do elevation (and seasonal)
        batch_size = out.shape[0]
        n_points = out.shape[1]

        elev_expanded = elev.repeat(batch_size, 1, 1)  # (batch, n_points, 3)

        if self.use_seasonal and seasonal is not None:
            # seasonal is (batch, 2), expand to (batch, n_points, 2)
            seasonal_expanded = seasonal.unsqueeze(1).expand(-1, n_points, -1)
            out = torch.cat([out, elev_expanded, seasonal_expanded], dim=2)
        else:
            out = torch.cat([out, elev_expanded], dim=2)

        out = self.elev_mlp(out)
        out[..., 0] = self.sigmoid(out[..., 0])
        out[..., 1:] = force_positive(out[..., 1:])

        return out

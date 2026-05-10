import math
import numpy as np
import torch
import torch.nn as nn

class SpatialPositionalEncoding(nn.Module):
    """
    Generates positional encodings for spatial coordinates.
    Automatically handles normalized [0, 1] coordinates.
    """
    def __init__(self, spatial_dim=2, encoding_dim=64, 
                 coordinate_range='normalized',  
                 original_bounds=None,
                 max_wavelength=None,
                 learnable=False, use_gaussian=False):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.encoding_dim = encoding_dim
        self.use_gaussian = use_gaussian
        self.coordinate_range = coordinate_range
        
        # Validate configuration
        if coordinate_range == 'original' and original_bounds is None:
            raise ValueError("original_bounds must be provided when using coordinate_range='original'")
        
        # Store original bounds for denormalization if needed
        if original_bounds is not None:
            self.register_buffer('original_bounds', torch.tensor(original_bounds, dtype=torch.float32))
        else:
            self.original_bounds = None
        
        # Auto-compute max_wavelength based on coordinate range
        if max_wavelength is None:
            max_wavelength = 10.0 if coordinate_range == 'normalized' else 10000.0
        
        if use_gaussian:

            grid_resolution = 10
            num_implicit_centers = grid_resolution ** 2  # 100 centers
            
            # Create grid on [0, 1]^2
            x = torch.linspace(0, 1, grid_resolution)
            y = torch.linspace(0, 1, grid_resolution)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            grid_centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1) # (100, 2)
            
            # Handle coordinate range scaling
            if coordinate_range == 'normalized':
                centers = grid_centers
                # Width approx matching grid spacing (0.1)
                log_widths = torch.log(torch.ones(num_implicit_centers) * (1.0 / grid_resolution))
            else:
                # Naive scaling for original coords (risky but consistent with previous logic)
                centers = grid_centers * 100
                log_widths = torch.zeros(num_implicit_centers)
            
            # Register parameters
            if learnable:
                self.centers = nn.Parameter(centers)
                self.log_widths = nn.Parameter(log_widths)
            else:
                self.register_buffer('centers', centers)
                self.register_buffer('log_widths', log_widths)
                
            # Projection layer if output dim differs from implicit centers
            self.num_centers = num_implicit_centers
            if self.num_centers != encoding_dim:
                self.gaussian_proj = nn.Linear(self.num_centers, encoding_dim)
                # Initialize to keep signal magnitude
                nn.init.xavier_uniform_(self.gaussian_proj.weight)
                nn.init.zeros_(self.gaussian_proj.bias)
            else:
                self.gaussian_proj = None
                
        else:
            # Sinusoidal encoding - ensure exact dimension match
            # Each spatial dimension gets pairs of sin/cos
            num_pairs_per_dim = encoding_dim // (2 * spatial_dim)
            remaining = encoding_dim - (num_pairs_per_dim * 2 * spatial_dim)
            self.num_pairs_per_dim = num_pairs_per_dim
            self.remaining = remaining
            
            # Create linear projection layer in __init__ if needed
            if remaining > 0:
                self.linear_proj = nn.Linear(spatial_dim, remaining)
                nn.init.xavier_uniform_(self.linear_proj.weight)
            else:
                self.linear_proj = None
            
            # Treat freq_bands as true frequencies (not divisors) for standard 2π scaling
            min_freq = 0.5 if coordinate_range == 'normalized' else 1.0
            freq_bands = torch.exp(torch.linspace(
                np.log(min_freq), np.log(max_wavelength), num_pairs_per_dim
            ))
            if learnable:
                self.freq_bands = nn.Parameter(freq_bands)
            else:
                self.register_buffer('freq_bands', freq_bands)
    
    def denormalize_coords(self, coords):
        """Convert [0, 1] normalized coordinates back to original scale."""
        if self.original_bounds is not None:
            mins = self.original_bounds[:, 0].view(1, 1, -1)
            maxs = self.original_bounds[:, 1].view(1, 1, -1)
            return coords * (maxs - mins) + mins
        return coords
    
    def forward(self, spatial_coords):
        """
        Args:
            spatial_coords: (B, N, spatial_dim) tensor of coordinates
        Returns:
            (B, N, encoding_dim) positional encodings
        """
        B, N, D = spatial_coords.shape
        
        # Optionally denormalize coordinates
        if self.coordinate_range == 'original' and self.original_bounds is not None:
            coords = self.denormalize_coords(spatial_coords)
        else:
            coords = spatial_coords
        
        if self.use_gaussian:
            # Gaussian RBF encoding
            # (B, N, D) vs (1, implicit_centers, D) -> (B, N, implicit_centers)
            dists = torch.cdist(coords, self.centers.unsqueeze(0).expand(B, -1, -1), p=2)
            widths = torch.exp(self.log_widths).unsqueeze(0).unsqueeze(0) + 1e-8
            
            # Raw RBF features (B, N, 100)
            encoding = torch.exp(-dists ** 2 / (2 * widths ** 2))
            
            # Project to requested dimension if needed (e.g., 100 -> 32)
            if self.gaussian_proj is not None:
                encoding = self.gaussian_proj(encoding)
            
            return encoding
        else:
            # Sinusoidal encoding - guaranteed to produce exact encoding_dim
            encodings = []
            for dim in range(D):
                coord = coords[..., dim:dim+1]
                # Standard sinusoidal PE: use frequencies with 2π scaling
                scaled = 2 * math.pi * coord * (self.freq_bands.unsqueeze(0).unsqueeze(0) + 1e-8)
                sin_enc = torch.sin(scaled)
                cos_enc = torch.cos(scaled)
                encodings.append(sin_enc)
                encodings.append(cos_enc)
            
            pe = torch.cat(encodings, dim=-1)
            
            # Use pre-created linear projection
            if self.remaining > 0 and self.linear_proj is not None:
                extra_features = self.linear_proj(coords)
                pe = torch.cat([pe, extra_features], dim=-1)
            
            # Assert instead of truncate
            assert pe.shape[-1] == self.encoding_dim, \
                f"Dimension mismatch: got {pe.shape[-1]}, expected {self.encoding_dim}"
            
            return pe

class MultiScalePositionalEncoding(nn.Module):
    """
    Multi-scale positional encoding with proper dimension handling.
    """
    def __init__(self, spatial_dim=2, encoding_dims=None, 
                 scales=None, coordinate_range='normalized',
                 original_bounds=None, learnable=False):
        super().__init__()
        
        # Ensure encoding_dims are provided
        if encoding_dims is None:
            encoding_dims = [32, 32, 32]
        
        # Auto-compute scales based on coordinate range
        if scales is None:
            if coordinate_range == 'normalized':
                scales = [0.1, 0.5, 1.0]
            else:
                scales = [1.0, 10.0, 100.0]
        
        # Ensure matching dimensions
        if len(scales) != len(encoding_dims):
            raise ValueError(f"scales ({len(scales)}) and encoding_dims ({len(encoding_dims)}) must have same length")
        
        self.scales = scales
        self.encoding_dims = encoding_dims
        self.total_dim = sum(encoding_dims)
        
        self.encoders = nn.ModuleList([
            SpatialPositionalEncoding(
                spatial_dim, dim, 
                coordinate_range=coordinate_range,
                original_bounds=original_bounds,
                max_wavelength=scale * (10.0 if coordinate_range == 'normalized' else 10000.0),
                learnable=learnable
            )
            for dim, scale in zip(encoding_dims, scales)
        ])
        
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, spatial_coords):
        encodings = []
        weights = torch.softmax(self.scale_weights.to(spatial_coords.device), dim=0)
        
        for i, encoder in enumerate(self.encoders):
            encoding = encoder(spatial_coords) * weights[i]
            encodings.append(encoding)
        
        return torch.cat(encodings, dim=-1)
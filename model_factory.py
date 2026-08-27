"""
Shared model construction logic for convCNP climate models.

This module provides a single source of truth for building the model architecture,
used by both training and prediction notebooks.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from params import Params
from convCNP.models.elev_models import TmaxBiasConvCNPElev, GammaBiasConvCNPElev
from convCNP.models.cnn import CNN, ResConvBlock
from convCNP.training.loss_functions import gll, gamma_ll, crps_gaussian
from convCNP.training.utils import get_value_tmax
from convCNP.models.smacnp import SMACNP_ALL



LossFn = Callable
GetValueFn = Callable


DRY_PROBABILITY_THRESHOLD = 0.5


def build_model(params: Params) -> tuple[nn.Module, LossFn, GetValueFn]:
    """
    Build convCNP model based on variable type.

    Args:
        params: Training/model configuration parameters.

    Returns:
        model: The constructed nn.Module (not yet moved to device).
        loss_fn: The loss function for training.
        get_value_fn: Function to extract point predictions from model output.
    """
    if params.IN_CHANNELS <= 0:
        raise ValueError(
            f"IN_CHANNELS={params.IN_CHANNELS} is invalid. "
            "It must be set from the data before building the model (see cell 5)."
        )

    # Build CNN decoder (same architecture for both variables)
    decoder = CNN(
        n_channels=params.N_CHANNELS,
        ConvBlock=ResConvBlock,
        n_blocks=params.N_BLOCKS,
        Conv=nn.Conv2d,
        Normalization=nn.Identity,
        kernel_size=params.KERNEL_SIZE,
    )

    # Build model based on variable
    if params.VARIABLE == "tmax":
        model = TmaxBiasConvCNPElev(
            decoder=decoder,
            in_channels=params.IN_CHANNELS,
            ls=params.LENGTH_SCALE,
            use_seasonal_in_mlp=params.SEASONAL_FEATURES_IN_MLP,
        )
        loss_fn = crps_gaussian if params.LOSS_FN == 'crps' else gll
        get_value_fn = get_value_tmax

    elif params.VARIABLE == "precip":
        model = GammaBiasConvCNPElev(
            decoder=decoder,
            in_channels=params.IN_CHANNELS,
            ls=params.LENGTH_SCALE,
            use_seasonal_in_mlp=params.SEASONAL_FEATURES_IN_MLP,
        )
        loss_fn = gamma_ll

        def get_value_precip(p):
            """Extract mean prediction for precipitation (Gamma distribution)."""
            mean = p[:, :, 1] / p[:, :, 2]  # alpha / beta
            mean[p[:, :, 0] <= DRY_PROBABILITY_THRESHOLD] = 0  # Set to 0 when dry
            return mean

        get_value_fn = get_value_precip

    else:
        raise ValueError(f"Unknown variable: {params.VARIABLE}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built {type(model).__name__} with {n_params:,} trainable parameters")

    return model, loss_fn, get_value_fn

def build_smacnp(params: Params, input_dim: int, context_input_dim: int = None) -> tuple[nn.Module, LossFn, GetValueFn]:
    """
    Build the SMACNP temperature model.

    Parameters
    ----------
    params    : Params dataclass (uses R_DIM, W_DIM, V_DIM, N_HEADS, etc.)
    input_dim        : feature columns in x_target (always 6 — no source flag on targets)
    context_input_dim: feature columns in x_context (6 for ERA5-only or PW-only; 7 for combined)

    """
    if params.VARIABLE != 'tmax':
        raise ValueError("SMACNP currently supports only VARIABLE='tmax'.")

    attr_dim = input_dim - 2
    context_attr_dim = (context_input_dim - 2
                        if context_input_dim is not None
                        else None)  

    model = SMACNP_ALL(
        input_dim=input_dim,
        spatial_dim=2,
        attr_dim=attr_dim,
        context_attr_dim=context_attr_dim,          
        y_dim=1,
        r_dim=params.R_DIM,
        w_dim=params.W_DIM,
        v_dim=params.V_DIM,
        hidden_size=params.SMACNP_HIDDEN,       
        num_heads=params.N_HEADS,
        num_layers_mlp=params.NUM_LAYERS_MLP,               
        dropout_rate=params.DROPOUT_RATE,
        laplace_distance_p=params.LAPLACE_P,                       
        spatial_indices=[0, 1],                 
        use_positional_encoding=params.USE_PE,  
        pe_dim=params.PE_DIM,
        use_layer_norm=params.USE_LAYER_NORM,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built SMACNP with {n_params:,} trainable parameters")

    loss_fn = crps_gaussian if params.LOSS_FN == 'crps' else gll
    return model, loss_fn, get_value_tmax

def load_model_checkpoint(checkpoint_path, p, device, input_dim=None, context_input_dim=None):
    if p.MODEL_TYPE == 'smacnp':
        if input_dim is None:
            raise ValueError("input_dim required for SMACNP — pass x_context.shape[-1]")
        model, _, _ = build_smacnp(p, input_dim, context_input_dim)
    else:
        model, _, _ = build_model(p)
    
    model.to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # When a model is trained with nn.DataParallel, PyTorch saves each key in
    # the state dict with a 'module.' prefix (e.g. 'module.decoder.weight').
    # Since we load onto a plain (non-DataParallel) model, we need to strip
    # that prefix so the keys match the model's own parameter names.
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict([
            (k.removeprefix('module.'), v) for k, v in state_dict.items()
        ])

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint.get('epoch', -1)

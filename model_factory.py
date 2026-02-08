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
from convCNP.training.loss_functions import gll, gamma_ll
from convCNP.training.utils import get_value_tmax


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
        loss_fn = gll
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


def load_model_checkpoint(
    checkpoint_path: Path,
    p: Params,
    device: torch.device
) -> tuple[nn.Module, int]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file.
        p: Training parameters used to build the model architecture.
        device: Torch device to load the model onto.

    Returns:
        Tuple of (model, epoch) where epoch is the training epoch of the checkpoint.
    """
    # Build model architecture using shared factory
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

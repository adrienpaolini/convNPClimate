"""
Post-training evaluation metrics for the convNP climate downscaling model.
"""

import numpy as np


def skill_score(crps_model, crps_reference):
    """
    Compute the skill score: 1 - (CRPS_model / CRPS_reference).

    Interpretation:
        1.0  = perfect model (zero CRPS)
        0.0  = model is as good as the reference
        <0   = model is worse than the reference

    Args:
        crps_model: mean CRPS of the probabilistic model predictions
        crps_reference: mean CRPS of the reference (for a deterministic
                        reference, CRPS equals the MAE)

    Returns:
        Skill score (scalar).
    """
    return 1.0 - (crps_model / crps_reference)

import torch
import numpy as np
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

def shuffle_data(context, task, seasonal=None):
    """
    Shuffle data before each epoch.

    Parameters:
    -----------
    context: input context tensor (time, ...)
    task: target tensor (time, ...)
    seasonal: optional seasonal features tensor (time, 2)

    Returns:
    --------
    Shuffled tensors in the same order
    """
    inds = np.linspace(0, context.shape[0]-1, context.shape[0], dtype=int)
    np.random.shuffle(inds)
    if seasonal is not None:
        return context[inds], task[inds], seasonal[inds]
    return context[inds], task[inds], None


def get_fold_data(inds, context, targets, shuffle=True, batch_size=16, seasonal=None):
    """
    Split the context and target data into folds.

    Parameters:
    -----------
    inds: tuple of (start, end) indices for held-out fold
    context: input context tensor (time, channels, lat, lon)
    targets: target tensor (time, n_points)
    shuffle: whether to shuffle data
    batch_size: batch size for splitting
    seasonal: optional seasonal features tensor (time, 2)

    Returns:
    --------
    training_data: list of task dicts with 'y_context', 'y_target', and optionally 'seasonal'
    held_out: list of task dicts for validation
    """
    # Get held out
    held_out_context = context[inds[0]:inds[1], ...]
    held_out_targets = targets[inds[0]:inds[1], :]
    held_out_seasonal = seasonal[inds[0]:inds[1], :] if seasonal is not None else None

    # Get training
    train_context = torch.cat(
        [context[0:inds[0], ...], context[inds[1]:, ...]]
    )
    train_targets = torch.cat(
        [targets[0:inds[0], ...], targets[inds[1]:, ...]]
    )
    train_seasonal = None
    if seasonal is not None:
        train_seasonal = torch.cat(
            [seasonal[0:inds[0], ...], seasonal[inds[1]:, ...]]
        )

    # Shuffle data
    if shuffle:
        train_context, train_targets, train_seasonal = shuffle_data(
            train_context, train_targets, train_seasonal
        )
        held_out_context, held_out_targets, held_out_seasonal = shuffle_data(
            held_out_context, held_out_targets, held_out_seasonal
        )

    train_targets = torch.split(train_targets, batch_size)
    train_context = torch.split(train_context, batch_size)
    held_out_context = torch.split(held_out_context, batch_size)
    held_out_targets = torch.split(held_out_targets, batch_size)

    # Split seasonal if provided
    if train_seasonal is not None:
        train_seasonal = torch.split(train_seasonal, batch_size)
        held_out_seasonal = torch.split(held_out_seasonal, batch_size)

    # Build task dicts
    if train_seasonal is not None:
        training_data = [
            {"y_context": train_context[i], "y_target": train_targets[i], "seasonal": train_seasonal[i]}
            for i in range(len(train_targets))
        ]
        held_out = [
            {"y_context": held_out_context[i], "y_target": held_out_targets[i], "seasonal": held_out_seasonal[i]}
            for i in range(len(held_out_targets))
        ]
    else:
        training_data = [
            {"y_context": train_context[i], "y_target": train_targets[i]}
            for i in range(len(train_targets))
        ]
        held_out = [
            {"y_context": held_out_context[i], "y_target": held_out_targets[i]}
            for i in range(len(held_out_targets))
        ]

    return training_data, held_out

def get_fold_data_smacnp(inds, x_context, y_context, x_target, y_target,
                          shuffle=True, batch_size=16,
                          x_context_val=None, y_context_val=None,
                          x_target_val=None,  y_target_val=None):

    """
    Split point-format SMACNP data into training and held-out folds.

    Parameters
    ----------
    inds      : (start, end) held-out slice
    x_context : (T, N, input_dim)  context attributes (ERA5 grid cells)
    y_context : (T, N, 1)          context observations (ERA5 temperature)
    x_target  : (T, M, input_dim) or (M, input_dim)  target attributes
    y_target  : (T, M)             target observations (station temperature)
    shuffle   : whether to shuffle time axis
    batch_size: number of time steps per batch

    Returns
    -------
    training_data, held_out  — lists of task dicts with keys:
        'x_context', 'y_context', 'x_target', 'y_target'
    """
    # Broadcast static target attributes across time if needed
    if x_target.dim() == 2:
        x_target = x_target.unsqueeze(0).expand(x_context.shape[0], -1, -1)

    def _split(arr, start, end):
        train = torch.cat([arr[:start], arr[end:]], dim=0)
        held  = arr[start:end]
        return train, held

    xc_tr, xc_ho = _split(x_context, *inds)
    yc_tr, yc_ho = _split(y_context, *inds)
    xt_tr, xt_ho = _split(x_target,  *inds)
    yt_tr, yt_ho = _split(y_target,  *inds)

    if x_context_val is not None:
        _, xc_ho = _split(x_context_val, *inds)
        _, yc_ho = _split(y_context_val, *inds)
    if x_target_val is not None:
        if x_target_val.dim() == 2:
            x_target_val = x_target_val.unsqueeze(0).expand(x_context.shape[0], -1, -1)
        _, xt_ho = _split(x_target_val, *inds)
        _, yt_ho = _split(y_target_val, *inds)

    if shuffle:
        idx = torch.randperm(xc_tr.shape[0])
        xc_tr, yc_tr, xt_tr, yt_tr = xc_tr[idx], yc_tr[idx], xt_tr[idx], yt_tr[idx]
        idx = torch.randperm(xc_ho.shape[0])
        xc_ho, yc_ho, xt_ho, yt_ho = xc_ho[idx], yc_ho[idx], xt_ho[idx], yt_ho[idx]

    def _to_batches(xc, yc, xt, yt):
        xc_b = torch.split(xc, batch_size)
        yc_b = torch.split(yc, batch_size)
        xt_b = torch.split(xt, batch_size)
        yt_b = torch.split(yt, batch_size)
        return [{'x_context': xc_b[i], 'y_context': yc_b[i],
                 'x_target':  xt_b[i], 'y_target':  yt_b[i]}
                for i in range(len(yt_b))]

    return _to_batches(xc_tr, yc_tr, xt_tr, yt_tr), \
           _to_batches(xc_ho, yc_ho, xt_ho, yt_ho)


def make_r_mask(target_vals):
    """
    Make the r mask for the Bernoulli precipitation distribution
    """
    # Make r mask
    r = torch.ones(target_vals.shape[0]).cuda()
    r[target_vals==0] = 0
    
    # Set the target vals to one to stop the pesky error
    # (It doesn't contribute anyway)
    target_vals[target_vals == 0] = 0.01

    return r, target_vals

def log_exp(x):
    """
    Fix overflow
    """
    lt = torch.where(torch.exp(x)<1000)
    if lt[0].shape[0] > 0:
        x[lt] = torch.log(1+torch.exp(x[lt]))
    return x

def generate_context_mask(batch_size, n_channels, x, y, device=None):
    """
    Generate a context mask - in this simple case this will be one
    for all grid points
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.ones(batch_size, n_channels, x, y, device=device)

def get_value_pr_gammagp(p):
    """
    Return predicted mean to calculate stats each epoch
    """

    # output.shape = [time, stations]
    output = torch.zeros(p.shape[0], 3010).cuda()

    for st in range(3010):
        # For each need to calculate the normalisation 
        x = torch.linspace(0.2, 150, 1499)
        x = x.view(-1, 1).repeat(1, p.shape[0]).cuda()
        norms = torch.sum(tbi_func(x, p[:, st, :]), dim = 0)
        ev = torch.sum(x*(1/norms)*tbi_func(x, p[:, st, :]), dim = 0)       
        output[:, st][p[:,st,0]>0.5] = ev

    return output

def get_value_tmax(p):
    """
    Return predicted mean to calculate stats each epoch
    """
    return p[:, :, 0]

def get_sigma_tmax(p):
    """Return predicted sigma (standard deviation) for tmax model."""
    return p[:, :, 1]

def tbi_func(x, v):
    """
    Evaluate Bernoulli-gamma-GP mixture likelihood
    Parameters:
    ----------
    v: torch.Tensor(batch*86, channels)
        parameters from model
    x: torch.Tensor(batch*86)
        target vals to eval at
    """
    # Gamma distribution
    g = Gamma(concentration = v[:,2], rate = v[:,3])
    gamma = torch.exp(torch.clamp(g.log_prob(x), min=-1e5, max=1e5))

    # Weight term
    weight_term = (1/2)+(1/np.pi)*torch.atan((x-v[:,5])/v[:,6])

    # GP distribution
    gp = (1/v[:,4])*(1+(v[:,1]*x/v[:,4]))**((-1/v[:,1])-1)

    # total
    tbi = gamma*(1-weight_term)+gp*weight_term
    return torch.clamp(tbi, min = 1e-5)

"""
Training functions: models with MLP elevation
"""

import csv
import time
from datetime import datetime
import torch
import numpy as np
import os
import scipy
from scipy.stats import NearConstantInputWarning
import warnings
from .utils import log_exp, generate_context_mask, get_fold_data, get_fold_data_smacnp


def get_fold_holdout_indices(fold: int, n_folds: int, n_samples: int) -> tuple:
    """
    Get the start and end indices for a fold's holdout period.

    Parameters:
    -----------
    fold: current fold index (0-based)
    n_folds: total number of folds
    n_samples: total number of time samples

    Returns:
    --------
    (start, end) tuple of indices defining the holdout slice [start, end)
    """
    fold_size = n_samples // n_folds
    start = fold * fold_size
    end = start + fold_size
    if fold == n_folds - 1:
        end = n_samples
    return start, end


def select_holdout_day(fold: int, holdout_start: int, holdout_end: int, seed: int = 42) -> int:
    """
    Deterministically select one day from the holdout period.
    Uses the middle day of the holdout period for consistency.

    Parameters:
    -----------
    fold: current fold index (unused, kept for future seeded strategies)
    holdout_start: start index of the holdout period
    holdout_end: end index of the holdout period (exclusive)
    seed: random seed (unused, kept for future seeded strategies)

    Returns:
    --------
    Index of the selected day
    """
    holdout_length = holdout_end - holdout_start
    day_offset = holdout_length // 2
    return holdout_start + day_offset


def train_batch_elev(task, opt, model, ll, elev, dists, seasonal=None, device=None):
    """
    Train one batch
    Parameters:
    -----------
    task: dict
        ['y_context', 'y_target', 'seasonal' (optional)]
    opt: Optimizer
    model: convCNP model
    ll: loss function
    elev: elevation features tensor
    dists: distances tensor
    seasonal: seasonal features for this batch (batch, 2) or None
    device: torch.device (optional)
    """
    batch_size, channels, x, y = task['y_context'].shape

    # Generate mask
    mask = generate_context_mask(batch_size, channels, x, y, device=device)

    # Get seasonal features from task if available, otherwise use passed argument
    batch_seasonal = task.get('seasonal', seasonal)

    # Forward pass
    v = model(task['y_context'], mask, dists, elev, seasonal=batch_seasonal)

    # Backprop
    obj = -ll(task['y_target'], v)
    obj.backward()
    opt.step()
    opt.zero_grad()

    return obj, opt, model

def eval_epoch_elev(model, held_out, ll, elev, dists, y_target_t, get_value, device=None):
    """
    Calculate nll on held out dataset after each epoch.

    Parameters:
    -----------
    model: convCNP model
    held_out: list of task dicts with 'y_context', 'y_target', and optionally 'seasonal'
    ll: loss function
    elev: elevation features tensor (n_points, 3)
    dists: distances tensor
    y_target_t: target transformer (unused, kept for API compatibility)
    get_value: function to extract predictions from model output
    device: torch.device (optional)
    """
    model.eval()

    targets = [i['y_target'] for i in held_out]
    targets_complete = torch.cat(targets, axis=0)

    predictions = []
    with torch.no_grad():
        for task in held_out:
            batch_size, channels, x, y = task['y_context'].shape

            # Predict parameters for the batch
            mask = generate_context_mask(batch_size, channels, x, y, device=device)

            # Get seasonal features from task if available
            batch_seasonal = task.get('seasonal', None)

            predictions.append(model(task['y_context'], mask, dists, elev, seasonal=batch_seasonal))

    # Calculate NLL
    predictions = torch.cat(predictions)
    eval_ll = -ll(targets_complete, predictions)

    # Transform predicted parameters to amounts
    predictions = get_value(predictions)

    maes = np.zeros(predictions.shape[1])
    spearmans = np.zeros(predictions.shape[1])
    pearsons = np.zeros(predictions.shape[1])

    # Print output by station
    for st in range(predictions.shape[1]):
        true_mean = targets_complete[:, st].detach().cpu().numpy() #y_target_t.inverse_transform(targets_complete[:, st].view(-1, 1).cpu())
        pred_mean = predictions[:, st].detach().cpu().numpy() #y_target_t.inverse_transform(predictions[:, st].view(-1, 1).detach().cpu())
        pred_mean = pred_mean[~np.isnan(true_mean)]
        true_mean = true_mean[~np.isnan(true_mean)]

        # When the held-out set is shuffled, some target points may have all NaN values
        # (common in observational climate data where not all stations report daily).
        # With 0 valid samples, mean is undefined. With <2 valid samples, correlation is
        # undefined - skip to avoid warnings.
        if len(true_mean) < 2:
            maes[st] = np.nan
            pearsons[st] = np.nan
            spearmans[st] = np.nan
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NearConstantInputWarning)
                maes[st] = np.mean(np.abs(true_mean - pred_mean))
                pearsons[st] = scipy.stats.pearsonr(pred_mean, true_mean)[0]
                spearmans[st] = scipy.stats.spearmanr(pred_mean, true_mean).correlation
        except:
            maes[st] = np.nan
            pearsons[st] = np.nan
            spearmans[st] = np.nan
            continue
        #plt.plot(true_mean)
        #plt.plot(pred_mean)
        #plt.show()

    median_mae = np.median(maes[~np.isnan(maes)])
    median_pearson = np.median(pearsons[~np.isnan(pearsons)])
    median_spearman = np.median(spearmans[~np.isnan(spearmans)])
    
    return eval_ll, median_mae, median_pearson, median_spearman

def train_epoch_elev(model, opt, training_data, ll, elev, dists, device=None):
    """
    Outer training loop for each epoch.

    Parameters:
    -----------
    model: convCNP model
    opt: Optimizer
    training_data: list of task dicts with 'y_context', 'y_target', and optionally 'seasonal'
    ll: loss function
    elev: elevation features tensor (n_points, 3)
    dists: distances tensor
    device: torch.device (optional)
    """
    model.train()

    # Train and update the model
    batch_objs = []
    for task in training_data:
        # Generate a mask
        obj, opt, model = train_batch_elev(task, opt, model, ll, elev, dists, device=device)
        batch_objs.append(float(obj.item()))
    train_ll = np.mean(np.array(batch_objs)[-5:])

    return train_ll
            
def train_elev(model,
          opt,
          ll,
          elev,
          dists,
          y_context,
          y_target,
          output_dir,
          y_target_t,
          get_value,
          fold,
          n_folds,
          n_epochs=100,
          batch_size=16,
          patience=10,
          stats_file=None,
          seasonal=None,
          device=None):
    """
    Top level training loop for the model.

    Parameters:
    -----------
    model: convCNP model
    opt: Optimizer
    ll: loss function
    elev: elevation features tensor (n_points, 3)
    dists: distances tensor
    y_context: input context tensor (time, channels, lat, lon)
    y_target: target tensor (time, n_points)
    output_dir: directory to save model checkpoints
    y_target_t: target transformer (unused, kept for API compatibility)
    get_value: function to extract predictions from model output
    fold: current fold index
    n_folds: total number of folds
    n_epochs: maximum number of epochs
    batch_size: batch size
    patience: number of epochs to wait for improvement before early stopping (None to disable)
    stats_file: path to CSV file for logging training statistics
    seasonal: optional seasonal features tensor (time, 2) with [cos_doy, sin_doy]
    device: torch.device (optional)
    """
    if not stats_file:
        raise ValueError("Please provide a stats file to log training statistics to.")

    test_score = []

    best_obj = 5
    epochs_without_improvement = 0

    # Run the training loop.
    print(f"Training using batch_size={batch_size}, patience={patience}")

    fold_start_time = time.time()
    epoch_durations = []

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fold_start_time))
    print(f"{timestamp}   Fold {fold+1}/{n_folds}, elapsed 0s, est. remaining unknown")


    with open(stats_file, 'a') as f:
        writer = csv.writer(f)

        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            if epoch > 0:
                del training_data
                del held_out

            n_samples = y_context.shape[0]
            start, end = get_fold_holdout_indices(fold, n_folds, n_samples)

            training_data, held_out = get_fold_data(
                (start, end), y_context, y_target, batch_size=batch_size, seasonal=seasonal
            )

            # Compute training objective.
            train_obj = train_epoch_elev(model, opt, training_data, ll, elev, dists, device=device)
            test_obj, median_mae, median_pearson, median_spearman = eval_epoch_elev(
                model, held_out, ll, elev, dists, y_target_t, get_value, device=device)
            test_score.append(test_obj)

            # Timing statistics
            epoch_end_time = time.time()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_end_time))
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_durations.append(epoch_duration)
            elapsed_in_fold = epoch_end_time - fold_start_time
            avg_epoch_duration = np.mean(epoch_durations)
            remaining_epochs = n_epochs - epoch - 1
            estimated_remaining = avg_epoch_duration * remaining_epochs

            def format_duration(seconds):
                h, remainder = divmod(int(seconds), 3600)
                m, s = divmod(remainder, 60)
                if h > 0:
                    return f"{h}h {m}m {s}s"
                elif m > 0:
                    return f"{m}m {s}s"
                else:
                    return f"{s}s"

            print(f"{timestamp}   Fold {fold+1}/{n_folds}, elapsed {format_duration(elapsed_in_fold)}, est. remaining {format_duration(estimated_remaining)} | Epoch {epoch} took {format_duration(epoch_duration)} | test NLL {test_obj:.3f} | train NLL {train_obj:.3f} | med MAE {median_mae:.3f} | med Pears {median_pearson:.3f} | med Spear {median_spearman:.3f}")

            writer.writerow([fold, median_mae, median_pearson, median_spearman, epoch, train_obj, test_obj.item()])
            f.flush()

            if test_obj < best_obj:
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': test_score}, os.path.join(output_dir, f"model_fold_{fold}"))
                best_obj = test_obj
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if patience is not None and epochs_without_improvement >= patience:
                print(f'Early stopping fold {fold} at epoch {epoch}: no improvement for {patience} epochs')
                break

def train_batch_smacnp(task, opt, model, ll, device=None):
    """
    Train one SMACNP batch.

    task keys: 'x_context' (B,N,D), 'y_context' (B,N,1),
               'x_target'  (B,M,D), 'y_target'  (B,M)
    """
    v   = model(task['x_context'], task['y_context'], task['x_target'])
    obj = -ll(task['y_target'], v)
    obj.backward()
    opt.step()
    opt.zero_grad()
    return obj, opt, model


def train_epoch_smacnp(model, opt, training_data, ll, device=None):
    """Outer loop over batches for one epoch."""
    model.train()
    batch_objs = []
    for task in training_data:
        obj, opt, model = train_batch_smacnp(task, opt, model, ll, device=device)
        batch_objs.append(float(obj.item()))
    return np.mean(np.array(batch_objs)[-5:])


def eval_epoch_smacnp(model, held_out, ll, get_value, device=None):
    """Evaluate NLL and point metrics on the held-out fold."""
    model.eval()
    targets_list, preds_list = [], []

    with torch.no_grad():
        for task in held_out:
            preds_list.append(model(task['x_context'],
                                    task['y_context'],
                                    task['x_target']))
            targets_list.append(task['y_target'])

    predictions     = torch.cat(preds_list)
    targets_complete = torch.cat(targets_list)

    eval_ll = -ll(targets_complete, predictions)
    predictions = get_value(predictions)

    n_stations = predictions.shape[1]
    maes, pearsons, spearmans = (np.zeros(n_stations) for _ in range(3))

    for st in range(n_stations):
        true = targets_complete[:, st].detach().cpu().numpy()
        pred = predictions[:, st].detach().cpu().numpy()
        mask = ~np.isnan(true)
        true, pred = true[mask], pred[mask]
        if len(true) < 2:
            maes[st] = pearsons[st] = spearmans[st] = np.nan
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                maes[st]      = np.mean(np.abs(true - pred))
                pearsons[st]  = scipy.stats.pearsonr(pred, true)[0]
                spearmans[st] = scipy.stats.spearmanr(pred, true).correlation
        except Exception:
            maes[st] = pearsons[st] = spearmans[st] = np.nan

    return (eval_ll,
            np.nanmedian(maes),
            np.nanmedian(pearsons),
            np.nanmedian(spearmans))


def train_smacnp(model, opt, ll, x_context, y_context, x_target, y_target,
                 output_dir, get_value, fold, n_folds,
                 n_epochs=100, batch_size=16, patience=10,
                 stats_file=None, device=None):
    """
    Top-level SMACNP training loop. Mirrors train_elev() in structure.

    Parameters
    ----------
    x_context : (T, N, input_dim)
    y_context : (T, N, 1)
    x_target  : (T, M, input_dim) or (M, input_dim)
    y_target  : (T, M)
    """
    if not stats_file:
        raise ValueError("Please provide a stats_file.")

    best_obj = float('inf')
    epochs_without_improvement = 0
    fold_start_time = time.time()
    epoch_durations = []

    with open(stats_file, 'a') as f:
        writer = csv.writer(f)

        for epoch in range(n_epochs):
            epoch_start = time.time()

            n_samples = x_context.shape[0]
            start, end = get_fold_holdout_indices(fold, n_folds, n_samples)

            training_data, held_out = get_fold_data_smacnp(
                (start, end), x_context, y_context, x_target, y_target,
                batch_size=batch_size
            )

            train_obj = train_epoch_smacnp(model, opt, training_data, ll, device=device)
            test_obj, med_mae, med_pears, med_spear = eval_epoch_smacnp(
                model, held_out, ll, get_value, device=device)

            epoch_dur = time.time() - epoch_start
            epoch_durations.append(epoch_dur)
            elapsed   = time.time() - fold_start_time
            remaining = np.mean(epoch_durations) * (n_epochs - epoch - 1)

            print(f"Fold {fold+1}/{n_folds} | Epoch {epoch} | "
                  f"train NLL {train_obj:.3f} | test NLL {test_obj:.3f} | "
                  f"med MAE {med_mae:.3f} | elapsed {elapsed:.0f}s | "
                  f"est. remaining {remaining:.0f}s")

            writer.writerow([fold, med_mae, med_pears, med_spear,
                             epoch, train_obj, test_obj.item()])
            f.flush()

            if test_obj < best_obj:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'loss': test_obj},
                           os.path.join(output_dir, f"model_fold_{fold}"))
                best_obj = test_obj
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping fold {fold} at epoch {epoch}.")
                break

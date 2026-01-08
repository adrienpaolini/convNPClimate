"""
Training functions: models with MLP elevation
"""

import csv
import torch
import numpy as np 
import os
import scipy
from .utils import log_exp, generate_context_mask, get_fold_data

def train_batch_elev(task, opt, model, ll, elev, dists, device=None):
    """
    Train one batch
    Parameters:
    -----------
    task: dict
        ['y_context', 'y_target']
    opt: Optimizer
    model: convCNP model
    ll: loss function
    device: torch.device (optional)
    """
    batch_size, channels, x, y = task['y_context'].shape

    # Generate mask
    mask = generate_context_mask(batch_size, channels, x, y, device=device)

    # Forward pass
    v = model(task['y_context'], mask, dists, elev)

    # Backprop
    obj = -ll(task['y_target'], v)
    obj.backward()
    opt.step()
    opt.zero_grad()

    return obj, opt, model

def eval_epoch_elev(model, held_out, ll, elev, dists, y_target_t, get_value, device=None):
    """
    Calculate nll on held out dataset after each epoch
    """
    model.eval()

    targets = [i['y_target'] for i in held_out]
    targets_complete = torch.cat(targets, axis = 0)

    predictions = []
    with torch.no_grad():
        for task in held_out:
            batch_size, channels, x, y = task['y_context'].shape

            # Predict parameters for the batch
            mask = generate_context_mask(batch_size, channels, x, y, device=device)
            predictions.append(model(task['y_context'], mask, dists, elev))

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
    print("Mean absolute error: {}".format(median_mae))
    print("Pearson correlation: {}".format(median_pearson))
    print("Spearman correlation: {}".format(median_spearman))
    
    return eval_ll, median_mae, median_pearson, median_spearman

def train_epoch_elev(model, opt, training_data, ll, elev, dists, device=None):
    """
    Outer training loop for each epoch
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
          n_epochs = 100,
          batch_size = 16,
          patience = 10,
          stats_file = None,
          device = None):
    """
    Top level training loop for the model

    Parameters:
    -----------
    patience : int
        Number of epochs to wait for improvement before early stopping.
        Set to None to disable early stopping.
    """
    if not stats_file:
        raise ValueError("Please provide a stats file to log training statistics to.")

    test_score = []

    best_obj = 5
    epochs_without_improvement = 0

    # Run the training loop.
    print(f"Training using batch_size={batch_size}, patience={patience}")

    with open(stats_file, 'a') as f:
        writer = csv.writer(f)

        for epoch in range(n_epochs):
            print("Starting epoch: {}".format(epoch))
            if epoch>0:
                del training_data
                del held_out

            n_samples = y_context.shape[0]
            fold_size = n_samples // n_folds
            start = fold * fold_size
            end = start + fold_size
            if fold == n_folds - 1:
                end = n_samples

            training_data, held_out = get_fold_data((start, end), y_context, y_target, batch_size)

            # Compute training objective.
            train_obj = train_epoch_elev(model, opt, training_data, ll, elev, dists, device=device)
            test_obj, median_mae, median_pearson, median_spearman = eval_epoch_elev(
                model, held_out, ll, elev, dists, y_target_t, get_value, device=device)
            test_score.append(test_obj)
            print('Epoch %s: train NLL %.3f, test NLL %.3f' % (epoch, train_obj, test_obj))

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
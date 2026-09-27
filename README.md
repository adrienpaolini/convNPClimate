# Attention-Based Neural Processes for Climate Downscaling over Switzerland

Spatial Multi-Attention Conditional Neural Processes (SMACNP) for probabilistic prediction of local-scale daily mean 2-meter air temperature over Switzerland, evaluated on statistical downscaling, off-grid station prediction, and a combined setting using both data sources as context. Builds on a ConvCNP baseline adapted from [Vaughan et al. (2022)](https://gmd.copernicus.org/preprints/gmd-2020-420/) and [Passos (2026)] (https://arxiv.org/abs/2607.04190), and a SMACNP architecture adapted from [Bao et al.](https://www.sciencedirect.com/science/article/pii/S0893608024001254) and [Dumitrescu et al.](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-223/).

This project was conducted during the 2026 Spring semester as a capstone project for a Diploma in Advanced Studies at ETH Zürich, under the supervision of Dr. Christian Donner from the Swiss Data Science Center. 

## Three Experiments

The project evaluates SMACNP in three settings, using different combinations of context and target data:

| Experiment | Context | Target | Task |
|---|---|---|---|
| Downscaling (DS) | ERA5-Land (~9 km grid) | MeteoSwiss (~1 km grid) | Predict on a regular grid, finer than the input |
| Off-grid (OffG) | PeakWeather stations | PeakWeather stations (held-out) | Predict at unseen, irregularly distributed stations |
| Combined (Comb) | ERA5-Land + PeakWeather | PeakWeather stations (held-out) | Same as off-grid, with additional gridded context |

Unlike ConvCNP, SMACNP does not require inputs or outputs to lie on a regular grid, which is what makes the off-grid and combined settings possible: these are closer to many practical use cases, where a temperature estimate is needed at a specific, unevenly spaced set of locations (e.g. a weather station network) rather than at the center of a grid cell.

## Downscaling in Practice

The model takes a coarse ERA5-Land temperature grid as input and produces high-resolution predictions that resolve Alpine valleys, ridges, and plateaus invisible at the input resolution

## Off-Grid Prediction and Attention

In the off-grid setting, SMACNP predicts temperature at weather stations that were not used as context, without any assumption of a regular spatial grid.

## Combining Gridded and Station Data

The combined experiment adds ERA5-Land as further context to the station data, using a source flag so the model can distinguish the two. 

## How SMACNP Works

SMACNP extends the Conditional Neural Process framework with multiple, separated attention mechanisms instead of a single grid-based convolutional encoder:

1. **Encode** — context observations are split into a spatial input (location) and an attribute input (explanatory variables such as altitude and topographic position); each is processed by its own attention mechanism
2. **Attend** — a Laplace (distance-based) attention mechanism captures spatial dependencies, while a separate cross-attention mechanism captures dependencies in the explanatory attributes; a third encoder produces a representation used to estimate the variance
3. **Decode** — the outputs of the encoders are combined by a decoder with two branches, producing a Gaussian predictive distribution (mean and variance) at each target location

This removes the requirement, inherent to ConvCNP, that inputs and outputs be defined on a regular grid.


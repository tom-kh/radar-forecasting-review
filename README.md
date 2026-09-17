# Anonymous Review Artifact: Radar Footprint Forecasting

This repository accompanies an anonymous robotics conference submission.

The implementation studies radar-only future road-user footprint forecasting
using causal front-radar histories, temporal encoding, partial line-of-sight
motion transport, JEPA-style predictive pretraining, and future-measurement
reconstruction.

## Scope

Model exteroceptive input:
- front automotive radar only

Offline registration:
- sensor calibration
- ego odometry

Not used as deployed model inputs:
- camera
- LiDAR
- future radar

The downstream target is a 2D annotation-derived vehicle/pedestrian footprint.
It is not physical free-space occupancy and AP values are not nuScenes
detection mAP.

## Repository structure

doppler_jepa/
    Core dataset, model, training, evaluation, metrics, geometry, and losses.

scripts/
    Experiment launchers and method variants.

tests/
    Unit/regression tests.

analysis/
    Paper analysis utilities where applicable.

## Installation

Create a Python environment and install the package:

    python -m pip install -e .

If a requirements file is included:

    python -m pip install -r requirements.txt

## Dataset

nuScenes is not redistributed in this repository.

Obtain nuScenes separately according to its official license and set your
local dataset/cache paths when preparing the data.

## Useful commands

Inspect training options:

    python -m doppler_jepa.train --help

Inspect evaluation options:

    python -m doppler_jepa.evaluate --help

Inspect preprocessing:

    python -m doppler_jepa.prepare --help

Run tests:

    python -m pytest -q

## Main experimental setting

Final experiments use:

- FP32 training
- 10 pretraining epochs
- 15 supervised fine-tuning epochs
- batch size 16
- raster size 128
- network width 96
- seeds 0, 1, 2
- 10% labeled training scenes for the reported low-label experiment

The experiment launchers under scripts/ encode the model variants used in the
paper.

## Important evaluation terminology

AP refers to AP_hist4096, an approximate grid-cell average precision.
It must not be interpreted as official nuScenes detection mAP.

Moving-footprint recall is recall over positive cells associated with moving
road users and is not a tracking or motion-segmentation metric.

## Reproducibility

The repository intentionally excludes:

- nuScenes data
- user-specific cache paths
- local logs
- unredacted checkpoints
- author information
- institutional information

These exclusions are required for double-anonymous review.

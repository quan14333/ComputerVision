# Small-Object-Aware WSSS

## Baseline

ExCEL — CVPR 2025

ExCEL is pinned as the Git submodule `third_party/ExCEL`. Clone this repository with `--recurse-submodules`, or run `git submodule update --init --recursive` after cloning.

## Research Direction

Small-Object-Aware Pseudo-Label Generation for Image-Level Weakly Supervised Semantic Segmentation.

## Project Structure

- `third_party/ExCEL/`: pinned ExCEL baseline submodule.
- `configs/`: versioned experiment configurations.
- `data_lists/`: reproducible split definitions without dataset contents.
- `src/`: new data, metric, pseudo-label, and analysis code.
- `scripts/`: validated workflow entry points.
- `results/`: lightweight experiment metadata templates.
- `tables/` and `figures/`: report-ready outputs.
- `docs/`: data documentation, decisions, literature, and experiment logs.

## Environment

Create an isolated Python environment and install dependencies from `third_party/ExCEL/requirements.txt`. Record exact package and runtime versions for every experiment.

## Data

Dataset locations are supplied through configuration or command-line arguments. Datasets are not committed to this repository.

## Experiments

Copy `results/_template` for each run and name the directory using `E###_<dataset>_<method>_<seed>_<YYYYMMDD>`.

Record the exact config, command, Git commit, environment, metrics, time, and memory for every run.

## Kaggle Training

Datasets are mounted from `/kaggle/input`. Code is cloned from GitHub. Large checkpoints and logs are stored in Kaggle Output or a Kaggle Dataset, not in Git.

Dataset paths must be provided through configuration or CLI options and must not be hard-coded in source code.

## Reproducibility

Use fixed split lists and seeds, pin the parent and submodule commits, preserve the effective configuration and command, and log artifacts in `docs/experiment_log.csv`.

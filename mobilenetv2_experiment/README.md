# MobileNetV2 Split-Inference Experiment

This folder provides a self-contained MobileNetV2 split-inference pipeline for validating EnSplit+ on a second DNN architecture. MobileNetV2 is partitioned between its top-level `features` modules, ensuring that each inverted-residual block and its internal skip connection remain on one compute node.

## Files

- `action_generator.py` defines the canonical split/compression actions and the current RANDOM selector.
- `generate_all_actions.py` exports the complete indexed action space to `results/action_space.csv`.
- `run_experiment.py` loads pretrained MobileNetV2, profiles FLOPs and activation sizes, performs segmented inference, and calculates computation latency, communication latency, and UE energy.
- `optimizer.py` evaluates all feasible actions and returns the constrained OPT action for the current system state.
- `compare_random_opt.py` runs the reproducible OPT, RANDOM, and LOCAL comparison and produces the paper-table results.
- `drl_environment.py` exposes the verified MobileNetV2 action, state-trace, inference, latency, energy, and accuracy interface for a DRL implementation.
- `results/mobilenetv2_comparison_runs.csv` contains individual inference results.
- `results/mobilenetv2_comparison_summary.csv` contains the aggregated results used in the paper table.

## Action Definition

MobileNetV2 contains 19 top-level feature modules and 20 boundaries, numbered from `0` to `19`. An action contains three ordered cut positions and one compression factor:

```python
Action(action_id=1235, cuts=(2, 10, 12), rho=0.25)
```

Generate or inspect every action with:

```bash
python mobilenetv2_experiment/generate_all_actions.py
```

## Setup

The experiment uses Python 3.11, PyTorch, torchvision, pandas, and Pillow. From the repository root:


The scripts use torchvision's pretrained ImageNet-1K MobileNetV2 weights. `--device auto` selects CUDA when available and otherwise uses CPU. The selected execution device affects only the time required to run PyTorch; simulated latency and energy are calculated from the configured UE/network resources.

## Run RANDOM, OPT, and LOCAL

Run the comparison used for the paper table with:

```bash
python mobilenetv2_experiment/compare_random_opt.py \
  --runs 10 \
  --steps 10 \
  --seed 1001 \
  --device auto
```

This performs 100 inference evaluations per method using ten labelled images, RANDOM seeds 1001--1010, and time-varying radio and compute traces. OPT searches all 5,317 actions, RANDOM samples from the same action list, and LOCAL executes the complete model on the UE. Accuracy is calculated by comparing the predicted ImageNet class with the ground-truth class. Total UE energy is the sum of UE computation and communication energy.

The experiment uses the same main assumptions as the VGG evaluation: four compute nodes, compression factors `1.0`, `0.75`, `0.50`, and `0.25`, UE power of 5 W, maximum energy credit of 90%, and the repository's radio and system traces. OPT uses latency and accuracy weights of `0.1` and `0.3`, respectively, with a maximum latency of 2 s.

For a simple RANDOM-only pipeline test, run:

```bash
python mobilenetv2_experiment/run_experiment.py \
  --weights default \
  --steps 10 \
  --seed 1001 \
  --device auto
```

## Adding EnSplit+ DDQN

Create the MobileNetV2 DDQN implementation as `ddqn_agent.py` and its training/evaluation entry point as `run_ddqn.py`. Use `MobileNetEnvironment` from `drl_environment.py` for the action mapping, traces, FLOP profiling, inference, latency, energy, and accuracy calculations. Network output index `i` must select `environment.actions[i]`, giving the DDQN output layer 5,317 Q-values.

The VGG DDQN can be used as an implementation reference, but its 137-action output layer and replay-buffer experiences cannot be reused directly.

After implementing the DRL files, train with the new entry point, for example:

```bash
python mobilenetv2_experiment/run_ddqn.py --mode train --episodes 5000 --device auto
```

Then evaluate the converged policy using the same protocol as OPT and RANDOM:

```bash
python mobilenetv2_experiment/run_ddqn.py \
  --mode evaluate \
  --runs 10 \
  --steps 10 \
  --device auto
```

The evaluation script should write individual decisions and a summary row named `EnSplit+ (DDQN)` using the same columns as `mobilenetv2_comparison_summary.csv`.

"""Compare exhaustive OPT, RANDOM, and LOCAL for MobileNetV2."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from action_generator import Action, COMPRESSION_RATES, enumerate_actions, sample_random_action
from optimizer import MobileNetOptimizer
from run_experiment import (
    ROOT, analytical_action_metrics, build_model, execute_action, load_states,
    profile_boundary_bytes, profile_flops, resolve_device, write_rows,
)


GROUND_TRUTH = {
    "input1.JPEG": 0, "input2.JPEG": 217, "input3.JPEG": 481,
    "input4.JPEG": 477, "input5.JPEG": 497, "input6.JPEG": 566,
    "input7.JPEG": 867, "input8.JPEG": 412, "input9.JPEG": 574,
    "input10.JPEG": 701,
}


def quality_cache(model, x, state, stage_flops, classifier_flops, ue_power, comm_scale):
    """Execute only 73 distinct (UE cut, rho) neural paths, not 5,317 actions."""
    quality = {}
    for ue_end in range(1, 20):
        rates = (1.0,) if ue_end == 19 else COMPRESSION_RATES
        for rho in rates:
            canonical = Action(-1, (ue_end, 19, 19), rho)
            result = execute_action(
                model, x, canonical, state, stage_flops, classifier_flops,
                ue_power, comm_scale,
            )
            quality[(ue_end, rho)] = (
                result["top1_confidence"], result["predicted_class"]
            )
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--radio-csv", type=Path, default=ROOT / "input/episode_parameters/radio_parameters_moving_1.csv")
    parser.add_argument("--system-csv", type=Path, default=ROOT / "input/episode_parameters/2000/system_parameters_1.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--ue-power-w", type=float, default=5.0)
    parser.add_argument("--comm-energy-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.runs <= 0:
        parser.error("--steps and --runs must be positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, preprocess = build_model("default")
    model.to(device)
    images = sorted((ROOT / "input").glob("input*.JPEG"))
    sample = preprocess(Image.open(images[0]).convert("RGB")).unsqueeze(0).to(device)
    stage_flops, classifier_flops = profile_flops(model, sample)
    boundary_bytes = profile_boundary_bytes(model, sample)
    states = load_states(args.radio_csv, args.system_csv)
    actions = enumerate_actions()
    local_action = next(a for a in actions if a.cuts == (19, 19, 19) and a.rho == 1.0)

    # Neural output depends on image, UE cut, and rho, not on system state or
    # allocation of the remaining modules. Cache these expensive evaluations
    # once per image and reuse them across independent system/random runs.
    qualities = {}
    for image_path in images:
        x = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        qualities[image_path.name] = quality_cache(
            model, x, states[0], stage_flops, classifier_flops,
            args.ue_power_w, args.comm_energy_scale,
        )

    rows = []
    for run in range(args.runs):
        rng = random.Random(args.seed + run)
        optimizer = MobileNetOptimizer()
        for step in range(args.steps):
            image_path = images[step % len(images)]
            state_idx = (run * args.steps + step) % len(states)
            state = states[state_idx]
            quality = qualities[image_path.name]
            baseline_confidence = quality[(19, 1.0)][0]
            opt_action, opt_metrics, opt_confidence, opt_predicted = optimizer.select(
                actions, state, quality, baseline_confidence, stage_flops,
                classifier_flops, boundary_bytes, args.ue_power_w,
                args.comm_energy_scale,
            )
            random_action = sample_random_action(actions, rng)
            random_metrics = analytical_action_metrics(
                random_action, state, stage_flops, classifier_flops,
                boundary_bytes, args.ue_power_w, args.comm_energy_scale,
            )
            random_confidence, random_predicted = quality[(random_action.cuts[0], random_action.rho)]
            local_metrics = analytical_action_metrics(
                local_action, state, stage_flops, classifier_flops,
                boundary_bytes, args.ue_power_w, args.comm_energy_scale,
            )
            local_confidence, local_predicted = quality[(19, 1.0)]
            target = GROUND_TRUTH.get(image_path.name)
            for method, action, metrics, confidence, predicted in (
                ("OPT", opt_action, opt_metrics, opt_confidence, opt_predicted),
                ("RANDOM", random_action, random_metrics, random_confidence, random_predicted),
                ("LOCAL", local_action, local_metrics, local_confidence, local_predicted),
            ):
                rows.append({
                    "run": run + 1, "seed": args.seed + run,
                    "step": step + 1, "state_index": state_idx,
                    "image": image_path.name, "method": method,
                    "action_id": action.action_id, "cuts": str(action.cuts), "rho": action.rho,
                    **metrics, "top1_confidence": confidence,
                    "predicted_class": predicted, "target_class": target,
                    "correct": int(target is not None and predicted == target),
                })
    write_rows(args.output_dir / "mobilenetv2_comparison_runs.csv", rows)
    frame = pd.DataFrame(rows)
    summary = []
    for method, group in frame.groupby("method", sort=False):
        summary.append({
            "model": "MobileNetV2", "method": method, "samples": len(group),
            "mean_latency_s": group["simulated_latency_s"].mean(),
            "std_latency_s": group["simulated_latency_s"].std(ddof=1),
            "mean_ue_compute_energy_j": group["ue_compute_energy_j"].mean(),
            "mean_ue_comm_energy_j": group["ue_comm_energy_j"].mean(),
            "mean_total_ue_energy_j": group["total_ue_energy_j"].mean(),
            "std_total_ue_energy_j": group["total_ue_energy_j"].std(ddof=1),
            "top1_accuracy_pct": 100.0 * group["correct"].mean(),
            "mean_top1_confidence_pct": 100.0 * group["top1_confidence"].mean(),
        })
    write_rows(args.output_dir / "mobilenetv2_comparison_summary.csv", summary)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

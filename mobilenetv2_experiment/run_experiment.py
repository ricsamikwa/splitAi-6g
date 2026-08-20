"""Random-action split inference for MobileNetV2 across UE/gNB/edge/core.

This module is intentionally independent from the VGG-specific experiment.
Reported latency is an analytical system-model value; PyTorch wall-clock time is
also captured separately as a diagnostic and is not used as simulated latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor

from action_generator import Action, SPLIT_BOUNDARIES, enumerate_actions, sample_random_action


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
NODE_NAMES = ("ue", "gnb", "edge", "core")
FINAL_OUTPUT_BYTES = 1000 * 4  # 1,000 float32 ImageNet logits


@dataclass(frozen=True)
class SystemState:
    cpu_ghz: tuple[float, float, float, float]
    flops_per_cycle: tuple[float, float, float, float]
    link_bandwidth_mbps: tuple[float, float, float]


def build_model(weights_mode: str) -> tuple[nn.Module, Compose]:
    if weights_mode == "default":
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        preprocess = weights.transforms()
    else:
        model = mobilenet_v2(weights=None)
        preprocess = Compose([
            Resize(256), CenterCrop(224), ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    model.eval()
    return model, preprocess


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda and fail clearly for an unavailable forced GPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def profile_flops(model: nn.Module, sample: torch.Tensor) -> tuple[list[int], int]:
    """Profile Conv2d/Linear multiply-add FLOPs for each splittable module."""
    stage_flops = [0 for _ in model.features]
    classifier_flops = 0
    hooks = []

    def conv_hook(stage: int):
        def hook(module: nn.Conv2d, inputs, output):
            out = output
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            kernel_ops *= module.in_channels // module.groups
            stage_flops[stage] += int(out.numel() * kernel_ops * 2)
        return hook

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal classifier_flops
        classifier_flops += int(module.in_features * module.out_features * 2)

    for stage_idx, stage in enumerate(model.features):
        for module in stage.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook(stage_idx)))
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(sample)
    for hook in hooks:
        hook.remove()
    return stage_flops, classifier_flops


def profile_boundary_bytes(model: nn.Module, sample: torch.Tensor) -> list[int]:
    """Return uncompressed activation bytes at boundaries 0..19."""
    sizes = [sample.numel() * sample.element_size()]
    output = sample
    with torch.inference_mode():
        for stage in model.features:
            output = stage(output)
            sizes.append(output.numel() * output.element_size())
    return sizes


def compress_channels(x: torch.Tensor, rho: float) -> tuple[torch.Tensor, int]:
    original_channels = x.shape[1]
    if rho >= 1.0 or x.ndim != 4:
        return x, original_channels
    kept = max(1, int(original_channels * rho))
    return x[:, :kept].contiguous(), original_channels


def restore_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] == channels:
        return x
    padded = x.new_zeros((x.shape[0], channels, x.shape[2], x.shape[3]))
    padded[:, : x.shape[1]] = x
    return padded


def compute_seconds(flops: int, ghz: float, flops_per_cycle: float) -> float:
    return flops / (ghz * 1e9 * flops_per_cycle)


def analytical_action_metrics(
    action: Action,
    state: SystemState,
    stage_flops: Sequence[int],
    classifier_flops: int,
    boundary_bytes: Sequence[int],
    ue_power_w: float,
    comm_energy_scale: float,
) -> dict[str, float | int]:
    """Evaluate latency/energy without executing the neural network."""
    total_latency = 0.0
    ue_compute_energy = 0.0
    ue_comm_energy = 0.0
    total_comm_bytes = 0
    segments = action.segments
    active = [i for i, (start, end) in enumerate(segments) if start < end]
    last_active = active[-1]
    for node_idx in active:
        start, end = segments[node_idx]
        node_flops = sum(stage_flops[start:end])
        if node_idx == last_active:
            node_flops += classifier_flops
        node_time = compute_seconds(
            node_flops, state.cpu_ghz[node_idx], state.flops_per_cycle[node_idx]
        )
        total_latency += node_time
        if node_idx == 0:
            ue_compute_energy += node_time * ue_power_w
        remaining = [j for j in active if j > node_idx]
        if remaining:
            next_node = remaining[0]
            data_bytes = boundary_bytes[end]
            if node_idx == 0 and action.rho < 1.0:
                data_bytes = max(1, int(data_bytes * action.rho))
            total_comm_bytes += data_bytes
            bandwidth = min(state.link_bandwidth_mbps[node_idx:next_node])
            total_latency += data_bytes / (bandwidth * 1e6)
            if node_idx == 0:
                ue_comm_energy += data_bytes * comm_energy_scale * 1e-7
    # Match the VGG NO_SPLIT convention: even fully local inference sends the
    # final 1,000-class result to the network/application endpoint.
    if last_active == 0:
        total_comm_bytes += FINAL_OUTPUT_BYTES
        total_latency += FINAL_OUTPUT_BYTES / (state.link_bandwidth_mbps[0] * 1e6)
        ue_comm_energy += FINAL_OUTPUT_BYTES * comm_energy_scale * 1e-7
    total_flops = sum(stage_flops) + classifier_flops
    ue_flops = sum(stage_flops[:segments[0][1]])
    if last_active == 0:
        ue_flops += classifier_flops
    return {
        "simulated_latency_s": total_latency,
        "ue_compute_energy_j": ue_compute_energy,
        "ue_comm_energy_j": ue_comm_energy,
        "total_ue_energy_j": ue_compute_energy + ue_comm_energy,
        "communication_bytes": total_comm_bytes,
        "offloaded_flops_pct": 100.0 * (total_flops - ue_flops) / total_flops,
    }


def execute_action(
    model: nn.Module,
    x: torch.Tensor,
    action: Action,
    state: SystemState,
    stage_flops: Sequence[int],
    classifier_flops: int,
    ue_power_w: float,
    comm_energy_scale: float,
) -> dict[str, float | int | str]:
    output = x
    total_latency = 0.0
    ue_compute_energy = 0.0
    ue_comm_energy = 0.0
    total_comm_bytes = 0
    compressed = False
    restore_to = 0
    wall_start = time.perf_counter()

    segments = action.segments
    active = [i for i, (start, end) in enumerate(segments) if start < end]
    last_active = active[-1]

    with torch.inference_mode():
        for node_idx, (start, end) in enumerate(segments):
            if start == end:
                continue
            if compressed:
                output = restore_channels(output, restore_to)
                compressed = False
            for stage_idx in range(start, end):
                output = model.features[stage_idx](output)

            node_flops = sum(stage_flops[start:end])
            if node_idx == last_active:
                output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
                output = torch.flatten(output, 1)
                output = model.classifier(output)
                node_flops += classifier_flops

            node_time = compute_seconds(
                node_flops, state.cpu_ghz[node_idx], state.flops_per_cycle[node_idx]
            )
            total_latency += node_time
            if node_idx == 0:
                ue_compute_energy += node_time * ue_power_w

            remaining = [j for j in active if j > node_idx]
            if remaining:
                next_node = remaining[0]
                if node_idx == 0:
                    output, restore_to = compress_channels(output, action.rho)
                    compressed = action.rho < 1.0
                data_bytes = output.numel() * output.element_size()
                total_comm_bytes += data_bytes
                # Link i represents the path from compute node i toward i+1. If
                # idle nodes are skipped, use the bottleneck of the traversed links.
                bandwidth = min(state.link_bandwidth_mbps[node_idx:next_node])
                total_latency += data_bytes / (bandwidth * 1e6)
                if node_idx == 0:
                    ue_comm_energy += data_bytes * comm_energy_scale * 1e-7

        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    if last_active == 0:
        total_comm_bytes += FINAL_OUTPUT_BYTES
        total_latency += FINAL_OUTPUT_BYTES / (state.link_bandwidth_mbps[0] * 1e6)
        ue_comm_energy += FINAL_OUTPUT_BYTES * comm_energy_scale * 1e-7

    total_flops = sum(stage_flops) + classifier_flops
    ue_flops = sum(stage_flops[segments[0][0]:segments[0][1]])
    if last_active == 0:
        ue_flops += classifier_flops
    return {
        "simulated_latency_s": total_latency,
        "wall_clock_s": time.perf_counter() - wall_start,
        "ue_compute_energy_j": ue_compute_energy,
        "ue_comm_energy_j": ue_comm_energy,
        "total_ue_energy_j": ue_compute_energy + ue_comm_energy,
        "communication_bytes": total_comm_bytes,
        "top1_confidence": float(confidence.item()),
        "predicted_class": int(predicted.item()),
        "offloaded_flops_pct": 100.0 * (total_flops - ue_flops) / total_flops,
    }


def load_states(radio_csv: Path, system_csv: Path) -> list[SystemState]:
    radio = pd.read_csv(radio_csv)
    system = pd.read_csv(system_csv)
    count = min(len(radio), len(system))
    states = []
    for i in range(count):
        # PDCP throughput is kbps in the repository trace; divide by 8000 for MB/s.
        ue_link = max(float(radio.iloc[i]["PDCP Throughput"]) / 8000.0, 1e-9)
        states.append(SystemState(
            cpu_ghz=(float(system.iloc[i]["ue_freq"]), *(float(system.iloc[i][f"freqs{n}"]) for n in range(1, 4))),
            flops_per_cycle=(float(system.iloc[i]["ue_flops_per_cycle"]), *(float(system.iloc[i][f"flops_per_cycle{n}"]) for n in range(1, 4))),
            link_bandwidth_mbps=(ue_link, float(system.iloc[i]["bandwidth2"]), float(system.iloc[i]["bandwidth3"])),
        ))
    return states


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=("none", "default"), default="none")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Tensor execution device. Auto uses CUDA when available.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--radio-csv", type=Path, default=ROOT / "input/episode_parameters/radio_parameters_moving_1.csv")
    parser.add_argument("--system-csv", type=Path, default=ROOT / "input/episode_parameters/2000/system_parameters_1.csv")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--ue-power-w", type=float, default=5.0)
    parser.add_argument("--comm-energy-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, preprocess = build_model(args.weights)
    model.to(device)
    images = sorted(args.input_dir.glob("input*.JPEG"))
    if not images:
        raise FileNotFoundError(f"No input*.JPEG images in {args.input_dir}")
    sample = preprocess(Image.open(images[0]).convert("RGB")).unsqueeze(0).to(device)
    stage_flops, classifier_flops = profile_flops(model, sample)
    actions = enumerate_actions()
    states = load_states(args.radio_csv, args.system_csv)

    action_rows = [{
        "action_id": a.action_id,
        "cuts": json.dumps(a.cuts),
        "segments": json.dumps(a.segments),
        "rho": a.rho,
    } for a in actions]
    write_rows(args.output_dir / "action_space.csv", action_rows)

    rows = []
    for step in range(args.steps):
        action = sample_random_action(actions, random)
        image_path = images[step % len(images)]
        x = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        metrics = execute_action(
            model, x, action, states[step % len(states)], stage_flops,
            classifier_flops, args.ue_power_w, args.comm_energy_scale,
        )
        rows.append({
            "step": step + 1,
            "image": image_path.name,
            "action_id": action.action_id,
            "cuts": json.dumps(action.cuts),
            "rho": action.rho,
            **metrics,
        })
    write_rows(args.output_dir / "mobilenetv2_random_runs.csv", rows)

    numeric = pd.DataFrame(rows)
    total_flops = sum(stage_flops) + classifier_flops
    summary = [{
        "model": "MobileNetV2",
        "weights": args.weights,
        "execution_device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "gflops": total_flops / 1e9,
        "candidate_split_boundaries": len(SPLIT_BOUNDARIES),
        "number_of_actions": len(actions),
        "samples": len(rows),
        "mean_latency_s": numeric["simulated_latency_s"].mean(),
        "std_latency_s": numeric["simulated_latency_s"].std(ddof=1),
        "mean_ue_compute_energy_j": numeric["ue_compute_energy_j"].mean(),
        "mean_ue_comm_energy_j": numeric["ue_comm_energy_j"].mean(),
        "mean_total_ue_energy_j": numeric["total_ue_energy_j"].mean(),
        "mean_top1_confidence": numeric["top1_confidence"].mean(),
        "mean_offloaded_flops_pct": numeric["offloaded_flops_pct"].mean(),
        "seed": args.seed,
    }]
    write_rows(args.output_dir / "mobilenetv2_random_summary.csv", summary)
    print(json.dumps(summary[0], indent=2))


if __name__ == "__main__":
    main()

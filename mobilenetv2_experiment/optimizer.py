"""Exhaustive constrained optimizer for MobileNetV2 split actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from action_generator import Action
from run_experiment import SystemState, analytical_action_metrics


@dataclass
class MobileNetOptimizer:
    weight_latency: float = 0.1
    weight_accuracy: float = 0.3
    max_latency_s: float = 2.0
    max_confidence_decrease: float = 0.80
    max_energy_credit: float = 0.90
    total_flops_offloaded: float = 0.0
    total_flops_on_ue: float = 0.0

    def select(
        self,
        actions: Sequence[Action],
        state: SystemState,
        quality: Mapping[tuple[int, float], tuple[float, int]],
        baseline_confidence: float,
        stage_flops: Sequence[int],
        classifier_flops: int,
        boundary_bytes: Sequence[int],
        ue_power_w: float,
        comm_energy_scale: float,
    ) -> tuple[Action, dict, float, int]:
        total_flops = sum(stage_flops) + classifier_flops
        best = None
        for action in actions:
            ue_end = action.cuts[0]
            confidence, predicted = quality[(ue_end, action.rho)]
            metrics = analytical_action_metrics(
                action, state, stage_flops, classifier_flops, boundary_bytes,
                ue_power_w, comm_energy_scale,
            )
            offloaded = total_flops * metrics["offloaded_flops_pct"] / 100.0
            ue_flops = total_flops - offloaded
            credit = (offloaded + self.total_flops_offloaded) / (
                total_flops + self.total_flops_on_ue
            )
            feasible = (
                metrics["simulated_latency_s"] <= self.max_latency_s
                and baseline_confidence - confidence <= self.max_confidence_decrease
                and credit <= self.max_energy_credit
            )
            if not feasible:
                continue
            objective = (
                self.weight_latency * metrics["simulated_latency_s"]
                + (1.0 - self.weight_latency) * metrics["total_ue_energy_j"]
                - self.weight_accuracy * confidence
            )
            candidate = (objective, action.action_id, action, metrics, confidence, predicted, offloaded, ue_flops)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("No MobileNetV2 action satisfies the OPT constraints")
        _, _, action, metrics, confidence, predicted, offloaded, ue_flops = best
        self.total_flops_offloaded += offloaded
        self.total_flops_on_ue += ue_flops
        return action, metrics, confidence, predicted

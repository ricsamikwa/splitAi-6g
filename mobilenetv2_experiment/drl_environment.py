"""MobileNetV2 environment adapter for DRL implementations.

The model, traces, actions, FLOP profiling, split execution, latency, and
energy calculations are wired here. No DRL algorithm is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image

from action_generator import Action, enumerate_actions
from compare_random_opt import GROUND_TRUTH
from run_experiment import (
    ROOT,
    SystemState,
    analytical_action_metrics,
    build_model,
    execute_action,
    load_states,
    profile_boundary_bytes,
    profile_flops,
    resolve_device,
)


@dataclass
class StepResult:
    action: Action
    state_index: int
    image: str
    metrics: dict[str, float | int]
    confidence: float
    predicted_class: int
    target_class: int | None
    correct: bool


class MobileNetEnvironment:
    """Adapter exposing the existing MobileNetV2 pipeline to a DRL agent."""

    def __init__(
        self,
        device: str = "auto",
        radio_csv: Path = ROOT / "input/episode_parameters/radio_parameters_moving_1.csv",
        system_csv: Path = ROOT / "input/episode_parameters/2000/system_parameters_1.csv",
        ue_power_w: float = 5.0,
        communication_energy_scale: float = 1.0,
    ) -> None:
        self.device = resolve_device(device)
        self.ue_power_w = ue_power_w
        self.communication_energy_scale = communication_energy_scale

        self.actions = enumerate_actions()
        self.number_of_actions = len(self.actions)
        self.states = load_states(radio_csv, system_csv)
        self.images = sorted((ROOT / "input").glob("input*.JPEG"))

        self.model, self.preprocess = build_model("default")
        self.model.to(self.device)
        self.model.eval()

        sample = self._load_image(self.images[0])
        self.stage_flops, self.classifier_flops = profile_flops(self.model, sample)
        self.boundary_bytes = profile_boundary_bytes(self.model, sample)

        # Prediction depends only on image, UE cut, and rho. This cache avoids
        # repeating real inference for actions that differ only in allocation
        # of the remaining modules across gNB/edge/core.
        self._quality_cache: dict[tuple[str, int, float], tuple[float, int]] = {}

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def action_from_index(self, action_index: int) -> Action:
        """Map a DDQN output neuron directly to the canonical action."""
        return self.actions[action_index]

    def raw_state(self, state_index: int) -> SystemState:
        """Return the compute/network state loaded by the existing pipeline."""
        return self.states[state_index % len(self.states)]

    def analytical_metrics(self, action: Action, state_index: int) -> dict[str, float | int]:
        """Reuse the existing analytical latency and energy calculation."""
        return analytical_action_metrics(
            action=action,
            state=self.raw_state(state_index),
            stage_flops=self.stage_flops,
            classifier_flops=self.classifier_flops,
            boundary_bytes=self.boundary_bytes,
            ue_power_w=self.ue_power_w,
            comm_energy_scale=self.communication_energy_scale,
        )

    def action_quality(self, action: Action, image_index: int) -> tuple[float, int]:
        """Return cached `(confidence, predicted_class)` for an action."""
        image_path = self.images[image_index % len(self.images)]
        ue_boundary = action.cuts[0]
        key = (image_path.name, ue_boundary, action.rho)
        if key not in self._quality_cache:
            # Later network allocations do not change model output, so use one
            # canonical allocation for this UE boundary and compression value.
            canonical = Action(-1, (ue_boundary, 19, 19), action.rho)
            result = execute_action(
                model=self.model,
                x=self._load_image(image_path),
                action=canonical,
                state=self.states[0],
                stage_flops=self.stage_flops,
                classifier_flops=self.classifier_flops,
                ue_power_w=self.ue_power_w,
                comm_energy_scale=self.communication_energy_scale,
            )
            self._quality_cache[key] = (
                float(result["top1_confidence"]),
                int(result["predicted_class"]),
            )
        return self._quality_cache[key]

    def step(self, action_index: int, state_index: int, image_index: int) -> StepResult:
        """Evaluate one selected DDQN action with the existing pipeline."""
        action = self.action_from_index(action_index)
        metrics = self.analytical_metrics(action, state_index)
        confidence, predicted = self.action_quality(action, image_index)
        image_path = self.images[image_index % len(self.images)]
        target = GROUND_TRUTH.get(image_path.name)
        return StepResult(
            action=action,
            state_index=state_index,
            image=image_path.name,
            metrics=metrics,
            confidence=confidence,
            predicted_class=predicted,
            target_class=target,
            correct=target is not None and predicted == target,
        )

if __name__ == "__main__":
    env = MobileNetEnvironment(device="auto")
    print(f"Device: {env.device}")
    print(f"States: {len(env.states)}")
    print(f"Actions: {env.number_of_actions}")
    print("Environment ready for a DRL implementation.")

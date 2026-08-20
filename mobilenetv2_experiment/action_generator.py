"""Replaceable action generator for the MobileNetV2 experiment.

The random policy currently samples from this module's action list. A future RL
policy can import the same Action objects and action indices without changing
the inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


NUM_FEATURE_MODULES = 19
# A split is safe between any two top-level torchvision MobileNetV2 `features`
# modules. Residual connections remain internal to an InvertedResidual module.
SPLIT_BOUNDARIES = tuple(range(NUM_FEATURE_MODULES + 1))
COMPRESSION_RATES = (1.0, 0.75, 0.5, 0.25)


@dataclass(frozen=True)
class Action:
    action_id: int
    cuts: tuple[int, int, int]
    rho: float

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        points = (0, *self.cuts, NUM_FEATURE_MODULES)
        return tuple(zip(points[:-1], points[1:]))


def enumerate_actions(
    compression_rates: tuple[float, ...] = COMPRESSION_RATES,
    allow_idle_network_nodes: bool = True,
) -> list[Action]:
    """Return all four-node MobileNetV2 split/compression actions.

    The UE always receives at least the first feature module. With idle network
    nodes enabled, cut indices may repeat. UE-only inference is represented by
    cuts `(19, 19, 19)` and has only rho=1 because nothing is transmitted.
    """
    actions: list[Action] = []
    internal = SPLIT_BOUNDARIES[1:]
    for c1 in internal:
        for c2 in (x for x in internal if x >= c1):
            for c3 in (x for x in internal if x >= c2):
                if not allow_idle_network_nodes and (c1 == c2 or c2 == c3 or c3 == NUM_FEATURE_MODULES):
                    continue
                ue_only = c1 == c2 == c3 == NUM_FEATURE_MODULES
                for rho in compression_rates:
                    if ue_only and rho != 1.0:
                        continue
                    actions.append(Action(len(actions), (c1, c2, c3), rho))
    return actions


def sample_random_action(actions: list[Action], rng) -> Action:
    """Random-policy seam; replace the caller with an RL policy later."""
    if not actions:
        raise ValueError("Cannot sample from an empty action space")
    return rng.choice(actions)

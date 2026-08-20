"""Generate the complete MobileNetV2 split/compression action space as CSV.

This script does not load PyTorch or execute inference. It can therefore be run
independently when defining the output dimension and index mapping of a future
RL policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from action_generator import SPLIT_BOUNDARIES, enumerate_actions


NODE_NAMES = ("ue", "gnb", "edge", "core")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "action_space.csv",
    )
    parser.add_argument(
        "--require-all-nodes",
        action="store_true",
        help="Exclude actions containing idle gNB, edge, or core nodes.",
    )
    args = parser.parse_args()

    actions = enumerate_actions(allow_idle_network_nodes=not args.require_all_nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "action_id", "rho", "cuts", "ue_layers", "gnb_layers",
        "edge_layers", "core_layers", "segments",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for action in actions:
            segments = action.segments
            writer.writerow({
                "action_id": action.action_id,
                "rho": action.rho,
                "cuts": json.dumps(action.cuts),
                "ue_layers": f"{segments[0][0]}:{segments[0][1]}",
                "gnb_layers": f"{segments[1][0]}:{segments[1][1]}",
                "edge_layers": f"{segments[2][0]}:{segments[2][1]}",
                "core_layers": f"{segments[3][0]}:{segments[3][1]}",
                "segments": json.dumps(segments),
            })

    print(f"MobileNetV2 boundaries: {len(SPLIT_BOUNDARIES)}")
    print(f"Split/compression actions: {len(actions)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

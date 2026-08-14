import os
import sys
import csv
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Path to project root: splitAi/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.vgg16_model import VGG16
from nodes.ue_node import UENode
from nodes.network_node import NetworkNode
from utils.comm_utils import calculate_comm_time


# ============================================================
# Config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_SPLITS = [0, 3, 6, 10, 14, 18]

COMPRESSION_RATES = [
    1.0,
    0.875,
    0.75,
    0.625,
    0.5,
    0.375,
    0.25,
]

INPUT_DIR = "input"
NUM_IMAGES = 10

# Fixed UE -> network bandwidth
BANDWIDTH_MBPS = 20.0

OUTPUT_CSV = "compression_accuracy_results.csv"
PER_IMAGE_CSV = "compression_accuracy_per_image.csv"


# ============================================================
# IMPORTANT:
# Ground-truth ImageNet class index for each input image.
#
# Replace these placeholder values with the actual class
# indices corresponding to input1.JPEG, ..., input10.JPEG.
#
# PyTorch ImageNet classes use indices 0 ... 999.
# ============================================================

GROUND_TRUTH_LABELS = {
    1: 1,   # <-- replace
    2: 2,   # <-- replace
    3: 3,   # <-- replace
    4: 4,   # <-- replace
    5: 5,   # <-- replace
    6: 6,   # <-- replace
    7: 7,   # <-- replace
    8: 8,   # <-- replace
    9: 9,   # <-- replace
    10: 10,  # <-- replace
}


# ============================================================
# Image preprocessing
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Load model
# ============================================================

def load_model():
    model = VGG16()

    model_dict = model.state_dict()

    weights = torch.load(
        "models/vgg16-modify.pth",
        map_location=DEVICE
    )

    model_dict.update(weights)
    model.load_state_dict(model_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Load images + ground-truth labels
# ============================================================

def load_test_images():
    images = []

    for i in range(1, NUM_IMAGES + 1):

        filename = os.path.join(
            INPUT_DIR,
            f"input{i}.JPEG"
        )

        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping.")
            continue

        if i not in GROUND_TRUTH_LABELS:
            print(
                f"Warning: no ground-truth label provided "
                f"for input{i}.JPEG, skipping."
            )
            continue

        img = Image.open(filename).convert("RGB")

        tensor = (
            preprocess(img)
            .unsqueeze(0)
            .to(DEVICE)
        )

        target = GROUND_TRUTH_LABELS[i]

        images.append(
            (i, tensor, target)
        )

    return images


# ============================================================
# Split inference
# ============================================================

@torch.no_grad()
def run_split_with_nodes(
    model,
    x,
    split_idx,
    rho
):

    total_layers = len(
        list(model.conv_layers.children())
    )

    # Dummy hardware parameters
    ue = UENode(
        cpu_freq=2.0,
        flops_per_cycle=4.0,
        power=5.0
    )

    net = NetworkNode(
        node_id=1,
        cpu_freq=3.0,
        flops_per_cycle=8.0
    )

    # --------------------------------------------------------
    # Full uncompressed feature size
    # --------------------------------------------------------

    ue_full = UENode(
        cpu_freq=2.0,
        flops_per_cycle=4.0,
        power=5.0
    )

    feat_full, _, _ = ue_full.compute(
        model=model,
        x=x,
        start_layer=0,
        end_layer=split_idx,
        flops=0.0,
        include_fc=False,
        rho=1.0,
    )

    Bf, Cf, Hf, Wf = feat_full.shape

    bytes_full = (
        Bf * Cf * Hf * Wf * 4
    )

    # --------------------------------------------------------
    # Actual compressed inference
    # --------------------------------------------------------

    feat_comp, _, _ = ue.compute(
        model=model,
        x=x,
        start_layer=0,
        end_layer=split_idx,
        flops=0.0,
        include_fc=False,
        rho=rho,
    )

    Bc, Cc, Hc, Wc = feat_comp.shape

    bytes_tx = (
        Bc * Cc * Hc * Wc * 4
    )

    comm_time = calculate_comm_time(
        bytes_tx,
        BANDWIDTH_MBPS
    )

    # --------------------------------------------------------
    # Network-side inference
    # --------------------------------------------------------

    net_out, _ = net.compute(
        model=model,
        x=feat_comp,
        start_layer=split_idx,
        end_layer=total_layers,
        flops=0.0,
        include_fc=True,
        rho=rho,
    )

    return (
        net_out,
        bytes_tx,
        bytes_full,
        comm_time
    )


# ============================================================
# Main
# ============================================================

def main():

    model = load_model()
    images = load_test_images()

    if not images:
        print(
            "No labeled images loaded. "
            "Check INPUT_DIR and GROUND_TRUTH_LABELS."
        )
        return

    print()
    print("=" * 80)
    print("Split + Compression Accuracy Evaluation")
    print("=" * 80)

    print(f"Device:          {DEVICE}")
    print(f"Number images:   {len(images)}")
    print(f"Splits:          {ALLOWED_SPLITS}")
    print(f"Compression:     {COMPRESSION_RATES}")

    print()

    # --------------------------------------------------------
    # Store baseline prediction (rho = 1.0) for each
    # split + image.
    #
    # Used to calculate class flip rate.
    # --------------------------------------------------------

    baseline_predictions = {}

    # Aggregate results
    summary_results = []

    # Detailed image-level results
    per_image_results = []

    # ========================================================
    # Evaluation
    # ========================================================

    for split_idx in ALLOWED_SPLITS:

        print()
        print("=" * 110)
        print(
            f"Split index {split_idx} "
            f"(UE -> network after conv layer {split_idx})"
        )
        print("=" * 110)

        # ----------------------------------------------------
        # First obtain rho=1.0 predictions
        # ----------------------------------------------------

        for image_id, x, target in images:

            logits, _, _, _ = run_split_with_nodes(
                model,
                x,
                split_idx,
                rho=1.0,
            )

            probs = F.softmax(
                logits,
                dim=1
            )

            top1_prob, top1_idx = probs.max(
                dim=1
            )

            baseline_predictions[
                (split_idx, image_id)
            ] = top1_idx.item()

        # ----------------------------------------------------
        # Evaluate all compression ratios
        # ----------------------------------------------------

        for rho in COMPRESSION_RATES:

            total = 0
            correct = 0
            flips = 0

            sum_top1_conf = 0.0

            total_bytes_tx = 0
            total_bytes_full = 0
            total_comm_time = 0.0

            for image_id, x, target in images:

                (
                    logits,
                    bytes_tx,
                    bytes_full,
                    comm_time
                ) = run_split_with_nodes(
                    model,
                    x,
                    split_idx,
                    rho
                )

                probs = F.softmax(
                    logits,
                    dim=1
                )

                top1_prob, top1_idx = probs.max(
                    dim=1
                )

                predicted_class = top1_idx.item()
                confidence = top1_prob.item()

                # --------------------------------------------
                # Actual Top-1 accuracy
                # --------------------------------------------

                is_correct = (
                    predicted_class == target
                )

                if is_correct:
                    correct += 1

                # --------------------------------------------
                # Has compression changed Top-1 class?
                # --------------------------------------------

                baseline_class = (
                    baseline_predictions[
                        (split_idx, image_id)
                    ]
                )

                class_flipped = (
                    predicted_class != baseline_class
                )

                if class_flipped:
                    flips += 1

                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                sum_top1_conf += confidence

                total_bytes_tx += bytes_tx
                total_bytes_full += bytes_full
                total_comm_time += comm_time

                total += 1

                # --------------------------------------------
                # Store individual image result
                # --------------------------------------------

                per_image_results.append({
                    "split_idx": split_idx,
                    "rho": rho,
                    "image_id": image_id,
                    "ground_truth": target,
                    "predicted_class": predicted_class,
                    "baseline_class": baseline_class,
                    "top1_confidence": confidence,
                    "correct": int(is_correct),
                    "class_flipped": int(class_flipped),
                })

            # ------------------------------------------------
            # Aggregate metrics
            # ------------------------------------------------

            top1_accuracy = (
                100.0 * correct / total
            )

            mean_top1_conf = (
                100.0 * sum_top1_conf / total
            )

            class_flip_rate = (
                100.0 * flips / total
            )

            avg_bytes_tx = (
                total_bytes_tx / total
            )

            avg_bytes_full = (
                total_bytes_full / total
            )

            avg_comm_time = (
                total_comm_time / total
            )

            data_reduction = (
                100.0
                * (
                    1.0
                    - avg_bytes_tx
                    / avg_bytes_full
                )
                if avg_bytes_full > 0
                else 0.0
            )

            summary_results.append({
                "split_idx": split_idx,
                "rho": rho,
                "top1_accuracy_pct": top1_accuracy,
                "mean_top1_confidence_pct": mean_top1_conf,
                "class_flip_rate_pct": class_flip_rate,
                "correct_images": correct,
                "total_images": total,
                "avg_bytes_full": avg_bytes_full,
                "avg_bytes_tx": avg_bytes_tx,
                "data_reduction_pct": data_reduction,
                "avg_comm_time_ms": avg_comm_time * 1000,
            })

            # ------------------------------------------------
            # Display immediately
            # ------------------------------------------------

            print(
                f"rho={rho:5.3f} | "
                f"Top-1 Acc={top1_accuracy:6.2f}% | "
                f"Top-1 Conf={mean_top1_conf:6.2f}% | "
                f"Class Flip={class_flip_rate:6.2f}% | "
                f"Correct={correct:2d}/{total:2d} | "
                f"Reduction={data_reduction:6.2f}%"
            )

    # ========================================================
    # Pretty final summary
    # ========================================================

    print()
    print("=" * 130)
    print("FINAL SUMMARY")
    print("=" * 130)

    header = (
        f"{'Split':>6} | "
        f"{'rho':>6} | "
        f"{'Top-1 Acc (%)':>15} | "
        f"{'Top-1 Conf (%)':>16} | "
        f"{'Flip Rate (%)':>13} | "
        f"{'Correct':>9} | "
        f"{'Reduction (%)':>13}"
    )

    print(header)
    print("-" * len(header))

    for r in summary_results:

        print(
            f"{r['split_idx']:6d} | "
            f"{r['rho']:6.3f} | "
            f"{r['top1_accuracy_pct']:15.2f} | "
            f"{r['mean_top1_confidence_pct']:16.2f} | "
            f"{r['class_flip_rate_pct']:13.2f} | "
            f"{r['correct_images']:2d}/{r['total_images']:<6d} | "
            f"{r['data_reduction_pct']:13.2f}"
        )

    # ========================================================
    # Save aggregate CSV
    # ========================================================

    if summary_results:

        with open(
            OUTPUT_CSV,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=summary_results[0].keys()
            )

            writer.writeheader()
            writer.writerows(summary_results)

    # ========================================================
    # Save per-image CSV
    # ========================================================

    if per_image_results:

        with open(
            PER_IMAGE_CSV,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=per_image_results[0].keys()
            )

            writer.writeheader()
            writer.writerows(per_image_results)

    print()
    print("=" * 80)
    print("Results saved:")
    print(f"  Aggregate: {OUTPUT_CSV}")
    print(f"  Per-image: {PER_IMAGE_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
import os
import sys
import csv
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path


# ============================================================
# Project root: splitAi-6g/
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from models.vgg16_model import VGG16
from nodes.ue_node import UENode
from nodes.network_node import NetworkNode
from utils.comm_utils import calculate_comm_time


# ============================================================
# Config
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


ALLOWED_SPLITS = [
    0,
    3,
    6,
    10,
    14,
    18
]


COMPRESSION_RATES = [
    1.0,
    0.875,
    0.75,
    0.625,
    0.5,
    0.375,
    0.25,
]


# ============================================================
# Dataset
#
# 10 samples per class x 10 classes = 100 images.
#
# Increase to 20 for 200 images, etc.
# ============================================================

VAL_DIR = Path(
    "data/imagenette2-160/val"
)

SAMPLES_PER_CLASS = 10

# Fixed seed makes the randomly selected subset reproducible.
SEED = 42


# ============================================================
# Imagenette synset -> ImageNet-1K output index
#
# These are the indices expected by an ImageNet-1K classifier.
# ============================================================

IMAGENETTE_TO_IMAGENET = {

    "n01440764": 0,      # tench

    "n02102040": 217,    # English springer

    "n02979186": 482,    # cassette player

    "n03000684": 491,    # chain saw

    "n03028079": 497,    # church

    "n03394916": 566,    # French horn

    "n03417042": 569,    # garbage truck

    "n03425413": 571,    # gas pump

    "n03445777": 574,    # golf ball

    "n03888257": 701,    # parachute
}


# ============================================================
# Fixed UE -> network bandwidth
# ============================================================

BANDWIDTH_MBPS = 20.0


# ============================================================
# Output files
# ============================================================

OUTPUT_CSV = (
    "compression_accuracy_results.csv"
)

PER_IMAGE_CSV = (
    "compression_accuracy_per_image.csv"
)


# ============================================================
# Image preprocessing
# ============================================================

preprocess = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# ============================================================
# Load model
#
# This is unchanged from your previous script.
# ============================================================

def load_model():

    model = VGG16()

    model_dict = model.state_dict()

    weights = torch.load(
        "models/vgg16-modify.pth",
        map_location=DEVICE
    )

    model_dict.update(weights)

    model.load_state_dict(
        model_dict
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# Load reproducible random Imagenette validation subset
# ============================================================

def load_test_images():

    random.seed(SEED)

    images = []

    image_id = 0

    print()
    print("=" * 80)
    print("LOADING IMAGENETTE VALIDATION SUBSET")
    print("=" * 80)

    if not VAL_DIR.exists():

        raise FileNotFoundError(
            f"Dataset not found: {VAL_DIR}\n"
            "Run first:\n"
            "python sandbox/download_imagenette.py"
        )

    # --------------------------------------------------------
    # Sample independently from each of the 10 classes
    # --------------------------------------------------------

    for synset, target in IMAGENETTE_TO_IMAGENET.items():

        class_dir = (
            VAL_DIR / synset
        )

        if not class_dir.exists():

            print(
                f"WARNING: class directory "
                f"{class_dir} not found."
            )

            continue

        files = []

        for extension in [
            "*.JPEG",
            "*.jpeg",
            "*.JPG",
            "*.jpg",
            "*.png"
        ]:

            files.extend(
                class_dir.glob(extension)
            )

        files = sorted(
            set(files)
        )

        if not files:

            print(
                f"WARNING: no images found "
                f"for {synset}"
            )

            continue

        number_to_sample = min(
            SAMPLES_PER_CLASS,
            len(files)
        )

        selected = random.sample(
            files,
            number_to_sample
        )

        print(
            f"{synset}: "
            f"{number_to_sample} images "
            f"-> ImageNet class {target}"
        )

        for filename in selected:

            img = Image.open(
                filename
            ).convert("RGB")

            tensor = (
                preprocess(img)
                .unsqueeze(0)
                .to(DEVICE)
            )

            images.append(
                (
                    image_id,
                    tensor,
                    target,
                    str(filename)
                )
            )

            image_id += 1

    print("-" * 80)

    print(
        f"Total images loaded: "
        f"{len(images)}"
    )

    print("=" * 80)

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
        list(
            model.conv_layers.children()
        )
    )

    # --------------------------------------------------------
    # UE
    # --------------------------------------------------------

    ue = UENode(
        cpu_freq=2.0,
        flops_per_cycle=4.0,
        power=5.0
    )

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    net = NetworkNode(
        node_id=1,
        cpu_freq=3.0,
        flops_per_cycle=8.0
    )

    # --------------------------------------------------------
    # Reference uncompressed feature size
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

    Bf, Cf, Hf, Wf = (
        feat_full.shape
    )

    bytes_full = (
        Bf
        * Cf
        * Hf
        * Wf
        * 4
    )

    # --------------------------------------------------------
    # Actual compressed UE output
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

    Bc, Cc, Hc, Wc = (
        feat_comp.shape
    )

    bytes_tx = (
        Bc
        * Cc
        * Hc
        * Wc
        * 4
    )

    # --------------------------------------------------------
    # Communication time
    # --------------------------------------------------------

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
# Main evaluation
# ============================================================

def main():

    model = load_model()

    images = load_test_images()

    if not images:

        print(
            "No labeled images loaded."
        )

        return

    print()

    print("=" * 90)

    print(
        "Split + Compression Accuracy Evaluation"
    )

    print("=" * 90)

    print(
        f"Device:              {DEVICE}"
    )

    print(
        f"Number images:       {len(images)}"
    )

    print(
        f"Samples per class:   "
        f"{SAMPLES_PER_CLASS}"
    )

    print(
        f"Random seed:         {SEED}"
    )

    print(
        f"Splits:              "
        f"{ALLOWED_SPLITS}"
    )

    print(
        f"Compression factors: "
        f"{COMPRESSION_RATES}"
    )

    print()


    # ========================================================
    # Store baseline predictions
    #
    # rho = 1.0 is used to determine whether compression
    # changes the predicted class.
    # ========================================================

    baseline_predictions = {}


    # Aggregate results
    summary_results = []


    # Individual-image results
    per_image_results = []


    # ========================================================
    # Evaluation
    # ========================================================

    for split_idx in ALLOWED_SPLITS:

        print()

        print("=" * 115)

        print(
            f"Split index {split_idx} "
            f"(UE -> network after "
            f"conv layer {split_idx})"
        )

        print("=" * 115)


        # ====================================================
        # Baseline predictions: rho = 1
        # ====================================================

        for (
            image_id,
            x,
            target,
            filename
        ) in images:

            (
                logits,
                _,
                _,
                _
            ) = run_split_with_nodes(

                model,

                x,

                split_idx,

                rho=1.0,
            )

            probs = F.softmax(
                logits,
                dim=1
            )

            (
                top1_prob,
                top1_idx
            ) = probs.max(
                dim=1
            )

            baseline_predictions[
                (
                    split_idx,
                    image_id
                )
            ] = top1_idx.item()


        # ====================================================
        # Evaluate each compression factor
        # ====================================================

        for rho in COMPRESSION_RATES:

            total = 0

            correct = 0

            flips = 0

            sum_top1_conf = 0.0

            total_bytes_tx = 0

            total_bytes_full = 0

            total_comm_time = 0.0


            for (
                image_id,
                x,
                target,
                filename
            ) in images:

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


                # --------------------------------------------
                # Prediction probabilities
                # --------------------------------------------

                probs = F.softmax(
                    logits,
                    dim=1
                )


                (
                    top1_prob,
                    top1_idx
                ) = probs.max(
                    dim=1
                )


                predicted_class = (
                    top1_idx.item()
                )

                confidence = (
                    top1_prob.item()
                )


                # --------------------------------------------
                # Actual classification accuracy
                # --------------------------------------------

                is_correct = (
                    predicted_class
                    == target
                )

                if is_correct:

                    correct += 1


                # --------------------------------------------
                # Class flip relative to rho = 1
                # --------------------------------------------

                baseline_class = (
                    baseline_predictions[
                        (
                            split_idx,
                            image_id
                        )
                    ]
                )


                class_flipped = (
                    predicted_class
                    != baseline_class
                )


                if class_flipped:

                    flips += 1


                # --------------------------------------------
                # Statistics
                # --------------------------------------------

                sum_top1_conf += (
                    confidence
                )

                total_bytes_tx += (
                    bytes_tx
                )

                total_bytes_full += (
                    bytes_full
                )

                total_comm_time += (
                    comm_time
                )

                total += 1


                # --------------------------------------------
                # Store per-image result
                # --------------------------------------------

                per_image_results.append({

                    "split_idx":
                        split_idx,

                    "rho":
                        rho,

                    "image_id":
                        image_id,

                    "filename":
                        filename,

                    "ground_truth":
                        target,

                    "predicted_class":
                        predicted_class,

                    "baseline_class":
                        baseline_class,

                    "top1_confidence":
                        confidence,

                    "correct":
                        int(is_correct),

                    "class_flipped":
                        int(class_flipped),
                })


            # =================================================
            # Aggregate metrics
            # =================================================

            top1_accuracy = (
                100.0
                * correct
                / total
            )


            mean_top1_conf = (
                100.0
                * sum_top1_conf
                / total
            )


            class_flip_rate = (
                100.0
                * flips
                / total
            )


            avg_bytes_tx = (
                total_bytes_tx
                / total
            )


            avg_bytes_full = (
                total_bytes_full
                / total
            )


            avg_comm_time = (
                total_comm_time
                / total
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


            # =================================================
            # Save aggregate result
            # =================================================

            summary_results.append({

                "split_idx":
                    split_idx,

                "rho":
                    rho,

                "top1_accuracy_pct":
                    top1_accuracy,

                "mean_top1_confidence_pct":
                    mean_top1_conf,

                "class_flip_rate_pct":
                    class_flip_rate,

                "correct_images":
                    correct,

                "total_images":
                    total,

                "avg_bytes_full":
                    avg_bytes_full,

                "avg_bytes_tx":
                    avg_bytes_tx,

                "data_reduction_pct":
                    data_reduction,

                "avg_comm_time_ms":
                    avg_comm_time * 1000,
            })


            # =================================================
            # Display immediately
            # =================================================

            print(

                f"rho={rho:5.3f} | "

                f"Top-1 Acc="
                f"{top1_accuracy:6.2f}% | "

                f"Top-1 Conf="
                f"{mean_top1_conf:6.2f}% | "

                f"Class Flip="
                f"{class_flip_rate:6.2f}% | "

                f"Correct="
                f"{correct:3d}/{total:3d} | "

                f"Reduction="
                f"{data_reduction:6.2f}%"
            )


    # ========================================================
    # Final summary
    # ========================================================

    print()

    print("=" * 135)

    print("FINAL SUMMARY")

    print("=" * 135)


    header = (

        f"{'Split':>6} | "

        f"{'rho':>6} | "

        f"{'Top-1 Acc (%)':>15} | "

        f"{'Top-1 Conf (%)':>16} | "

        f"{'Flip Rate (%)':>13} | "

        f"{'Correct':>11} | "

        f"{'Reduction (%)':>13}"
    )


    print(header)

    print(
        "-" * len(header)
    )


    for r in summary_results:

        print(

            f"{r['split_idx']:6d} | "

            f"{r['rho']:6.3f} | "

            f"{r['top1_accuracy_pct']:15.2f} | "

            f"{r['mean_top1_confidence_pct']:16.2f} | "

            f"{r['class_flip_rate_pct']:13.2f} | "

            f"{r['correct_images']:3d}/"
            f"{r['total_images']:<7d} | "

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

                fieldnames=
                    summary_results[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                summary_results
            )


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

                fieldnames=
                    per_image_results[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                per_image_results
            )


    # ========================================================
    # Done
    # ========================================================

    print()

    print("=" * 80)

    print("Results saved:")

    print(
        f"  Aggregate: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"  Per-image: "
        f"{PER_IMAGE_CSV}"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from models.vgg16_model import VGG16
from nodes.network_node import NetworkNode


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIR = "input"
NUM_IMAGES = 10


# ------------------------------------------------------------
# Replace these with the actual ground-truth ImageNet indices
# corresponding to input1.JPEG ... input10.JPEG.
#
# ImageNet model output indices normally range from 0 to 999.
# ------------------------------------------------------------

GROUND_TRUTH_LABELS = {
    1: 0,
    2: 217,
    3: 481,
    4: 477,
    5: 497,
    6: 566,
    7: 867,
    8: 412,
    9: 574,
    10: 701,
}


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Run full inference using NetworkNode
# ------------------------------------------------------------

@torch.no_grad()
def run_full_inference(model, x):
    """
    Executes the complete VGG-16 model using the same NetworkNode
    inference path used in the EnSplit evaluation.

    split_idx = 0
    rho = 1.0

    Therefore:
        - no layers are executed at the UE
        - no compression is applied
        - all convolutional + FC layers are executed at NetworkNode
    """

    total_layers = len(
        list(model.conv_layers.children())
    )

    net = NetworkNode(
        node_id=1,
        cpu_freq=3.0,
        flops_per_cycle=8.0
    )

    logits, _ = net.compute(
        model=model,
        x=x,
        start_layer=0,
        end_layer=total_layers,
        flops=0.0,
        include_fc=True,
        rho=1.0,
    )

    return logits


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

@torch.no_grad()
def main():

    model = load_model()

    print()
    print("=" * 100)
    print("CHECK IMAGE LABELS - FULL VGG-16 INFERENCE")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print()

    correct = 0
    total = 0

    for i in range(1, NUM_IMAGES + 1):

        filename = os.path.join(
            INPUT_DIR,
            f"input{i}.JPEG"
        )

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if not os.path.exists(filename):
            print(
                f"input{i}.JPEG | NOT FOUND"
            )
            continue

        # ----------------------------------------------------
        # Check label
        # ----------------------------------------------------

        if i not in GROUND_TRUTH_LABELS:
            print(
                f"input{i}.JPEG | NO GROUND-TRUTH LABEL"
            )
            continue

        target = GROUND_TRUTH_LABELS[i]

        # ----------------------------------------------------
        # Load + preprocess image
        # ----------------------------------------------------

        img = Image.open(filename).convert("RGB")

        x = (
            preprocess(img)
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Full inference
        # ----------------------------------------------------

        logits = run_full_inference(
            model,
            x
        )

        # Print once to verify classifier output
        if total == 0:
            print(
                f"Model output shape: {tuple(logits.shape)}"
            )

            if logits.ndim != 2:
                print(
                    "WARNING: expected output shape [batch, classes]."
                )

            print("-" * 100)

        # ----------------------------------------------------
        # Convert logits -> probabilities
        # ----------------------------------------------------

        probs = F.softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # Top-1 prediction
        # ----------------------------------------------------

        top1_prob, top1_idx = probs.max(
            dim=1
        )

        pred = top1_idx.item()

        confidence = (
            top1_prob.item() * 100.0
        )

        # ----------------------------------------------------
        # Compare with ground truth
        # ----------------------------------------------------

        is_correct = (
            pred == target
        )

        if is_correct:
            correct += 1

        total += 1

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"input{i}.JPEG | "
            f"GT={target:4d} | "
            f"Pred={pred:4d} | "
            f"Conf={confidence:7.2f}% | "
            f"{'CORRECT' if is_correct else 'WRONG'}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("-" * 100)

    if total > 0:

        accuracy = (
            100.0 * correct / total
        )

        print(
            f"Top-1 Accuracy: "
            f"{accuracy:.2f}% "
            f"({correct}/{total})"
        )

    else:
        print(
            "No images were evaluated."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.vgg16_model import VGG16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "input"
NUM_IMAGES = 10

# Replace with your current labels
GROUND_TRUTH_LABELS = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
}

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


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


@torch.no_grad()
def main():
    model = load_model()

    print()
    print("=" * 90)
    print("CHECK IMAGE LABELS")
    print("=" * 90)

    correct = 0
    total = 0

    for i in range(1, NUM_IMAGES + 1):

        filename = os.path.join(INPUT_DIR, f"input{i}.JPEG")

        if not os.path.exists(filename):
            print(f"{filename} not found")
            continue

        img = Image.open(filename).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)

        # Full model inference -- NO split, NO compression
        logits = model(x)

        probs = F.softmax(logits, dim=1)

        top1_prob, top1_idx = probs.max(dim=1)

        pred = top1_idx.item()
        conf = top1_prob.item() * 100.0
        target = GROUND_TRUTH_LABELS[i]

        match = pred == target

        if match:
            correct += 1

        total += 1

        print(
            f"input{i}.JPEG | "
            f"GT={target:4d} | "
            f"Pred={pred:4d} | "
            f"Conf={conf:6.2f}% | "
            f"{'CORRECT' if match else 'WRONG'}"
        )

    print("-" * 90)

    if total > 0:
        print(
            f"Top-1 Accuracy: "
            f"{100.0 * correct / total:.2f}% "
            f"({correct}/{total})"
        )


if __name__ == "__main__":
    main()
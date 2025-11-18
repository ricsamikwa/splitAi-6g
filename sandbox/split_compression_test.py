import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Path to project root: splitAi/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.vgg16_model import VGG16  # your existing model

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same splits you already use
ALLOWED_SPLITS = [0, 3, 6, 10, 14, 18]  # indices in conv_layers
COMPRESSION_RATES = [1.0, 0.75, 0.5, 0.25]  # ρ values

# Directory with your test images: input1.JPEG ... input10.JPEG
INPUT_DIR = "input"
NUM_IMAGES = 10  # adjust if you have more/less

# -----------------------
# Preprocessing (same as main script)
# -----------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------
# Helper: load model
# -----------------------
def load_model():
    model = VGG16()
    model_dict = model.state_dict()
    model_dict.update(torch.load("models/vgg16-modify.pth", map_location=DEVICE))
    model.load_state_dict(model_dict)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Helper: baseline inference (no split, no compression)
# -----------------------
@torch.no_grad()
def baseline_inference(model, x):
    """
    Full inference without splitting or compression.
    Assumes model.forward(x, 0, total_layers) + fc1,fc2,fc3 as in your existing code.
    """
    total_layers = len(list(model.conv_layers.children()))
    out = model.forward(x, 0, total_layers)  # all conv layers
    out = torch.flatten(out, 1)
    out = model.fc1(out)
    out = model.fc2(out)
    out = model.fc3(out)
    return out  # logits

# -----------------------
# Simple compression: channel reduction + 8-bit quant/dequant + zero pad
# -----------------------
def compress_feature(feat, rho):
    B, C, H, W = feat.shape
    bytes_full = B * C * H * W * 4

    if abs(rho - 1.0) < 1e-8:
        return feat.clone(), bytes_full, bytes_full

    C_red = max(1, int(rho * C))
    feat_red = feat[:, :C_red, :, :]

    # Decide whether to quantize based on rho
    # if rho >= 0.75:
    if True:
        # No quantization, just channel reduction
        x_q = feat_red
        num_elements = B * C_red * H * W
        bytes_tx = num_elements * 4  # model as float32 if you want
    else:
        # Quantize to 8-bit
        x = feat_red
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min) < 1e-8:
            x_q = x.clone()
        else:
            x_norm = (x - x_min) / (x_max - x_min)
            x_int = torch.round(x_norm * 255.0)
            x_q = x_int / 255.0 * (x_max - x_min) + x_min

        num_elements = B * C_red * H * W
        bytes_tx = num_elements  # 1 byte per element

    feat_hat = torch.zeros(B, C, H, W, device=feat.device, dtype=feat.dtype)
    feat_hat[:, :C_red, :, :] = x_q

    return feat_hat, bytes_tx, bytes_full


# -----------------------
# Split + compression inference
# -----------------------
@torch.no_grad()
def split_inference_with_compression(model, x, split_idx, rho):
    """
    x: input [B,3,224,224]
    split_idx: index in conv_layers where we split UE vs network
    rho: compression rate
    """
    total_layers = len(list(model.conv_layers.children()))

    # UE part: conv layers up to split_idx
    ue_out = model.forward(x, 0, split_idx)  # [B, C, H, W]

    # Compress and "transmit"
    ue_out_hat, bytes_tx, bytes_full = compress_feature(ue_out, rho)

    # Network part: remaining conv layers
    net_out = model.forward(ue_out_hat, split_idx, total_layers)

    # FC layers at network side
    net_out = torch.flatten(net_out, 1)
    net_out = model.fc1(net_out)
    net_out = model.fc2(net_out)
    net_out = model.fc3(net_out)

    return net_out, bytes_tx, bytes_full  # logits, bytes transmitted, baseline bytes

# -----------------------
# Load all test images
# -----------------------
def load_test_images():
    images = []
    for i in range(1, NUM_IMAGES + 1):
        filename = os.path.join(INPUT_DIR, f"input{i}.JPEG")
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping.")
            continue
        img = Image.open(filename).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
        images.append((i, tensor))
    return images

# -----------------------
# Main evaluation
# -----------------------
def main():
    model = load_model()
    images = load_test_images()
    if not images:
        print("No images loaded. Check INPUT_DIR and NUM_IMAGES.")
        return

    print("Evaluating split + compression combinations...")
    print(f"Number of test images: {len(images)}")
    print()

    # Precompute baseline logits & top1 classes
    baseline_top1 = {}
    for idx, x in images:
        logits_base = baseline_inference(model, x)
        pred_base = torch.argmax(logits_base, dim=1).item()
        baseline_top1[idx] = pred_base


    results = {}  # dict keyed by split_idx: list of (rho, acc, avg_full_bytes, avg_tx_bytes, red%)

    for split_idx in ALLOWED_SPLITS:
        results[split_idx] = []
        for rho in COMPRESSION_RATES:
            correct = 0
            total = 0
            total_bytes_tx = 0
            total_bytes_full = 0

            for idx, x in images:
                logits_split, bytes_tx, bytes_full = split_inference_with_compression(model, x, split_idx, rho)
                pred_split = torch.argmax(logits_split, dim=1).item()

                # "Accuracy": top-1 agreement with baseline prediction
                if pred_split == baseline_top1[idx]:
                    correct += 1
                total += 1

                total_bytes_tx += bytes_tx
                total_bytes_full += bytes_full

            acc = correct / total if total > 0 else 0.0
            avg_bytes_tx = total_bytes_tx / max(total, 1)
            avg_bytes_full = total_bytes_full / max(total, 1)
            reduction = 0.0
            if avg_bytes_full > 0:
                reduction = 100.0 * (1.0 - (avg_bytes_tx / avg_bytes_full))

            results[split_idx].append((rho, acc, avg_bytes_full, avg_bytes_tx, reduction))

    # Print results grouped by split index
    for split_idx in ALLOWED_SPLITS:

        print(f"\n=== Split index {split_idx} (UE→Net after conv layer {split_idx}) ===")
        print("rho   | Top-1 acc | Rel (%) | Avg bytes full (float32) | Avg bytes UE→Net | Reduction")
        print("-" * 110)

        # The first entry is rho = 1.0 → its acc is the reference
        base_acc = results[split_idx][0][1]  # acc at rho=1.0

        for rho, acc, avg_full, avg_tx, red in results[split_idx]:

            # Relative percentage vs the uncompressed case
            rel = (acc / base_acc * 100.0) if base_acc > 0 else 0.0

            print(f"{rho:4.2f} | {acc:8.4f} | {rel:7.2f}% | "
                f"{avg_full:23.0f} | {avg_tx:17.0f} | {red:8.2f}%")





if __name__ == "__main__":
    main()

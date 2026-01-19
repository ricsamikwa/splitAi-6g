import os
import sys
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
from utils.comm_utils import calculate_comm_time   # <-- added

# -----------------------
# Config
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_SPLITS = [0, 3, 6, 10, 14, 18]
COMPRESSION_RATES = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
# COMPRESSION_RATES = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375]

INPUT_DIR = "input"
NUM_IMAGES = 10

# Fixed UE→Net bandwidth in MB/s for this offline test
BANDWIDTH_MBPS = 20.0   # e.g., 20 MB/s ≈ 160 Mbps

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


@torch.no_grad()
def run_split_with_nodes(model, x, split_idx, rho):
    """
    Uses your updated UENode and NetworkNode.

    Layout:
      - UE:  conv[0 : split_idx)   (no FC)
      - Net: conv[split_idx : end) + FC
    """

    total_layers = len(list(model.conv_layers.children()))

    # dummy “hardware” params just to instantiate nodes
    ue = UENode(cpu_freq=2.0, flops_per_cycle=4.0, power=5.0)
    net = NetworkNode(node_id=1, cpu_freq=3.0, flops_per_cycle=8.0)

    # --- reference full feature at split (for bytes_full baseline) ---
    ue_full = UENode(cpu_freq=2.0, flops_per_cycle=4.0, power=5.0)
    feat_full, _, _ = ue_full.compute(
        model=model,
        x=x,
        start_layer=0,
        end_layer=split_idx,
        flops=0.0,
        include_fc=False,
        rho=1.0,              # force no compression
    )
    Bf, Cf, Hf, Wf = feat_full.shape
    bytes_full = Bf * Cf * Hf * Wf * 4  # float32

    # --- actual path with rho (compression is done *inside* UENode) ---
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
    bytes_tx = Bc * Cc * Hc * Wc * 4  # still float32 here

    # communication time for UE→Net transfer
    comm_time = calculate_comm_time(bytes_tx, BANDWIDTH_MBPS)

    # --- network side with your NetworkNode ---
    net_out, _ = net.compute(
        model=model,
        x=feat_comp,
        start_layer=split_idx,
        end_layer=total_layers,
        flops=0.0,
        include_fc=True,
        rho=rho,
    )

    return net_out, bytes_tx, bytes_full, comm_time

# -----------------------
# Main evaluation
# -----------------------
def main():
    model = load_model()
    images = load_test_images()
    if not images:
        print("No images loaded. Check INPUT_DIR and NUM_IMAGES.")
        return

    print("Evaluating split + compression combinations using UENode & NetworkNode...")
    print(f"Number of test images: {len(images)}")
    print()

    # results[split_idx] = list of (rho, mean_top1, avg_full_bytes, avg_tx_bytes, reduction, avg_comm_time)
    results = {}

    for split_idx in ALLOWED_SPLITS:
        results[split_idx] = []
        for rho in COMPRESSION_RATES:
            total = 0
            total_bytes_tx = 0
            total_bytes_full = 0
            total_comm_time = 0.0
            sum_top1_conf = 0.0

            for idx, x in images:
                logits_split, bytes_tx, bytes_full, comm_time = run_split_with_nodes(
                    model, x, split_idx, rho
                )

                final_output = F.softmax(logits_split, dim=1)
                top1_prob, top1_idx = torch.topk(final_output, 1)
                top1_conf = top1_prob.item()

                sum_top1_conf += top1_conf
                total += 1
                total_bytes_tx += bytes_tx
                total_bytes_full += bytes_full
                total_comm_time += comm_time

            mean_top1 = sum_top1_conf / max(total, 1)
            avg_bytes_tx = total_bytes_tx / max(total, 1)
            avg_bytes_full = total_bytes_full / max(total, 1)
            avg_comm_time = total_comm_time / max(total, 1)

            reduction = 0.0
            if avg_bytes_full > 0:
                reduction = 100.0 * (1.0 - (avg_bytes_tx / avg_bytes_full))

            results[split_idx].append(
                (rho, mean_top1, avg_bytes_full, avg_bytes_tx, reduction, avg_comm_time)
            )

    # -----------------------
    # Summary table
    # -----------------------
    for split_idx in ALLOWED_SPLITS:
        print(f"\n=== Split index {split_idx} (UE→Net after conv layer {split_idx}) ===")

        # Find reference mean_top1 for rho = 1.0
        base_top1 = None
        for rho, mean_top1, avg_full, avg_tx, red, avg_comm in results[split_idx]:
            if abs(rho - 1.0) < 1e-8:
                base_top1 = mean_top1
                break

        if base_top1 is None or base_top1 == 0.0:
            print("Warning: no valid baseline (rho=1.0) Top-1 confidence for this split.")
            continue

        print("rho   | Top-1 prob | Rel (%)vs ρ= | "
              "Avg bytes full | Avg bytes UE→Net | Data reduction | Avg Comm Time (ms)")
        print("-" * 140)

        for rho, mean_top1, avg_full, avg_tx, red, avg_comm in results[split_idx]:
            rel = (mean_top1 / base_top1) * 100.0 if base_top1 > 0 else 0.0

            print(
                f"{rho:4.2f} | {mean_top1:15.4f} | {rel:14.2f}% | "
                f"{avg_full:23.0f} | {avg_tx:17.0f} | {red:8.2f}% | {avg_comm*1000:10.3f}"
            )

if __name__ == "__main__":
    main()

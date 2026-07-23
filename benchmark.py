import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity
)
from torchmetrics.regression import MeanAbsoluteError
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation
from model_file import GlassSegFormerRGBRadar  # <-- import your radar model
from data_loader import get_loaders        # <-- import your loader
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# CONFIG
# -----------------------------------
DATA_ROOT = "/home/astik/Downloads/test_dinesh"

RADAR_MODEL_PATH = "/home/astik/glass_seg/best_glass_seg_attention_stratified.pt"
SEGFORMER_MODEL_PATH = "/home/astik/glass_seg/best_segformer_baseline.pt"

IMG_SIZE = 402
BATCH_SIZE = 4

# -----------------------------------
# LOAD DATA
# -----------------------------------
_, _, test_loader = get_loaders(
    DATA_ROOT,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE
)

def save_test_results(loader, model, results_dir="results", threshold=0.5):
    model.eval()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for rgb, radar, mask, names in loader:
            rgb = rgb.to(device)
            radar = radar.to(device)

            # logits = model(rgb, radar)
            if isinstance(model, GlassSegFormerRGBRadar):
                logits = model(rgb, radar)
            else:
                outputs = model(pixel_values=rgb)
                logits = outputs.logits
            probs  = torch.sigmoid(logits)
            preds  = (probs > threshold).float()

            for i in range(rgb.size(0)):

                name = names[i]
                sample_dir = results_dir / name
                sample_dir.mkdir(parents=True, exist_ok=True)

                # -------------------------
                # RGB (3-channel)
                # -------------------------
                rgb_img = rgb[i].cpu()
                vutils.save_image(rgb_img, sample_dir / "rgb.png")

                # -------------------------
                # Radar (1-channel → convert to PIL)
                # -------------------------
                radar_img = radar[i, 0].cpu()
                TF.to_pil_image(radar_img).save(sample_dir / "radar.png")

                # -------------------------
                # Ground Truth
                # -------------------------
                gt_img = mask[i, 0].cpu()
                TF.to_pil_image(gt_img).save(sample_dir / "gt.png")

                # -------------------------
                # Binary Prediction
                # -------------------------
                pred_img = preds[i, 0].cpu()
                TF.to_pil_image(pred_img).save(sample_dir / "pred.png")

                # -------------------------
                # Probability Map (optional but recommended)
                # -------------------------
                prob_img = probs[i, 0].cpu()
                TF.to_pil_image(prob_img).save(sample_dir / "prob.png")

    print(f"Saved results to: {results_dir.resolve()}")

# -----------------------------------
# METRICS
# -----------------------------------
def get_metrics():
    return {
        "iou": BinaryJaccardIndex().to(device),
        "f1": BinaryF1Score().to(device),
        "recall": BinaryRecall().to(device),
        "specificity": BinarySpecificity().to(device),
        "mae": MeanAbsoluteError().to(device)
    }

# -----------------------------------
# EVALUATION FUNCTION
# -----------------------------------
def evaluate_model(model, use_radar=False):

    metrics = get_metrics()
    model.eval()

    with torch.no_grad():
        for rgb, radar, mask, _ in tqdm(test_loader):

            rgb = rgb.to(device)
            radar = radar.to(device)
            mask = mask.to(device)

            if use_radar:
                pred = model(rgb, radar)
            else:
                outputs = model(pixel_values=rgb)
                pred = outputs.logits

            pred = F.interpolate(
                pred,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            probs = torch.sigmoid(pred)
            preds_bin = (probs > 0.5).int()
            mask_int = mask.int()

            metrics["iou"].update(preds_bin, mask_int)
            metrics["f1"].update(preds_bin, mask_int)
            metrics["recall"].update(preds_bin, mask_int)
            metrics["specificity"].update(preds_bin, mask_int)
            metrics["mae"].update(probs, mask)

    # Compute
    iou = metrics["iou"].compute().item()
    f1 = metrics["f1"].compute().item()
    recall = metrics["recall"].compute().item()
    specificity = metrics["specificity"].compute().item()
    mae = metrics["mae"].compute().item()

    # Balanced Error Rate
    fpr = 1 - specificity
    fnr = 1 - recall
    ber = 0.5 * (fpr + fnr)

    return {
        "mIoU": iou,
        "F-score": f1,
        "MAE": mae,
        "BER": ber
    }

# -----------------------------------
# LOAD RADAR MODEL
# -----------------------------------
radar_model = GlassSegFormerRGBRadar().to(device)
radar_model.load_state_dict(torch.load(RADAR_MODEL_PATH, map_location=device))

# -----------------------------------
# LOAD BASELINE SEGFORMER
# -----------------------------------
segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True
).to(device)

segformer_model.load_state_dict(torch.load(SEGFORMER_MODEL_PATH, map_location=device))

# -----------------------------------
# RUN EVALUATION
# -----------------------------------
print("\nEvaluating RGB + Radar Model")
radar_results = evaluate_model(radar_model, use_radar=True)

print("\nEvaluating RGB-only SegFormer")
segformer_results = evaluate_model(segformer_model, use_radar=False)

# -----------------------------------
# PRINT RESULTS
# -----------------------------------
print("\n==============================")
print("BENCHMARK RESULTS")
print("==============================\n")

print("RGB + Radar Model:")
for k, v in radar_results.items():
    print(f"{k}: {v:.4f}")

print("\nRGB-only SegFormer:")
for k, v in segformer_results.items():
    print(f"{k}: {v:.4f}")

save_test_results(test_loader, radar_model, "dataset_22_radar")
save_test_results(test_loader, segformer_model, "dinesh_segformer")

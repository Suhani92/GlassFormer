"""Evaluate GlassFormer (RGB + radar) against the RGB-only SegFormer baseline.

Reports mIoU, F-measure, MAE and BER on the test split (Tables II and III),
and optionally dumps per-sample qualitative results (RGB / radar / GT / pred /
probability map).

Example::

    python -m glassformer.evaluate \
        --data-root /path/to/test_split \
        --radar-ckpt checkpoints/best_glassformer.pt \
        --segformer-ckpt checkpoints/best_segformer_baseline.pt \
        --save-dir results/
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.regression import MeanAbsoluteError

from glassformer.models.glassformer import GlassSegFormerRGBRadar
from glassformer.data.dataset import get_loaders

SEGFORMER_ID = "nvidia/segformer-b2-finetuned-ade-512-512"


def get_metrics(device):
    return {
        "iou": BinaryJaccardIndex().to(device),
        "f1": BinaryF1Score().to(device),
        "recall": BinaryRecall().to(device),
        "specificity": BinarySpecificity().to(device),
        "mae": MeanAbsoluteError().to(device),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, use_radar=False):
    metrics = get_metrics(device)
    model.eval()

    for rgb, radar, mask, _ in tqdm(loader):
        rgb, radar, mask = rgb.to(device), radar.to(device), mask.to(device)

        if use_radar:
            pred = model(rgb, radar)
        else:
            pred = model(pixel_values=rgb).logits

        pred = F.interpolate(
            pred, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )
        probs = torch.sigmoid(pred)
        preds_bin = (probs > 0.5).int()
        mask_int = mask.int()

        metrics["iou"].update(preds_bin, mask_int)
        metrics["f1"].update(preds_bin, mask_int)
        metrics["recall"].update(preds_bin, mask_int)
        metrics["specificity"].update(preds_bin, mask_int)
        metrics["mae"].update(probs, mask)

    recall = metrics["recall"].compute().item()
    specificity = metrics["specificity"].compute().item()
    ber = 0.5 * ((1 - specificity) + (1 - recall))  # Balanced Error Rate

    return {
        "mIoU": metrics["iou"].compute().item(),
        "F-score": metrics["f1"].compute().item(),
        "MAE": metrics["mae"].compute().item(),
        "BER": ber,
    }


@torch.no_grad()
def save_test_results(loader, model, device, results_dir, use_radar=True, threshold=0.5):
    model.eval()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for rgb, radar, mask, names in loader:
        rgb, radar = rgb.to(device), radar.to(device)

        if use_radar:
            logits = model(rgb, radar)
        else:
            logits = model(pixel_values=rgb).logits

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i in range(rgb.size(0)):
            sample_dir = results_dir / names[i]
            sample_dir.mkdir(parents=True, exist_ok=True)
            vutils.save_image(rgb[i].cpu(), sample_dir / "rgb.png")
            TF.to_pil_image(radar[i, 0].cpu()).save(sample_dir / "radar.png")
            TF.to_pil_image(mask[i, 0].cpu()).save(sample_dir / "gt.png")
            TF.to_pil_image(preds[i, 0].cpu()).save(sample_dir / "pred.png")
            TF.to_pil_image(probs[i, 0].cpu()).save(sample_dir / "prob.png")

    print(f"Saved results to: {results_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root of the test split.")
    parser.add_argument("--radar-ckpt", required=True, help="GlassFormer checkpoint (.pt).")
    parser.add_argument("--segformer-ckpt", help="RGB-only SegFormer baseline checkpoint (.pt).")
    parser.add_argument("--save-dir", default=None, help="If set, dump qualitative results here.")
    parser.add_argument("--img-size", type=int, default=402)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_loaders(
        args.data_root, batch_size=args.batch_size, img_size=args.img_size
    )

    radar_model = GlassSegFormerRGBRadar().to(device)
    radar_model.load_state_dict(torch.load(args.radar_ckpt, map_location=device))

    print("\nEvaluating RGB + Radar (GlassFormer)")
    radar_results = evaluate_model(radar_model, test_loader, device, use_radar=True)

    segformer_results = None
    if args.segformer_ckpt:
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_ID, num_labels=1, ignore_mismatched_sizes=True
        ).to(device)
        segformer_model.load_state_dict(torch.load(args.segformer_ckpt, map_location=device))
        print("\nEvaluating RGB-only SegFormer baseline")
        segformer_results = evaluate_model(segformer_model, test_loader, device, use_radar=False)

    print("\n" + "=" * 30)
    print("BENCHMARK RESULTS")
    print("=" * 30)
    print("\nGlassFormer (RGB + Radar):")
    for k, v in radar_results.items():
        print(f"  {k}: {v:.4f}")
    if segformer_results:
        print("\nRGB-only SegFormer:")
        for k, v in segformer_results.items():
            print(f"  {k}: {v:.4f}")

    if args.save_dir:
        save_test_results(test_loader, radar_model, device, args.save_dir, use_radar=True)


if __name__ == "__main__":
    main()

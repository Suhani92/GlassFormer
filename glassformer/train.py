"""Train GlassFormer (SegFormer-B2 + RadarAttention) on the RGB-D-radar dataset.

Example::

    python -m glassformer.train \
        --data-root /path/to/dataset \
        --epochs 20 --batch-size 4 --img-size 402 \
        --out checkpoints/best_glassformer.pt
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from glassformer.models.glassformer import GlassSegFormerRGBRadar
from glassformer.data.dataset import get_loaders
from glassformer.losses import seg_loss, iou_score


def train_epoch(loader, model, optimizer, device, epoch=None):
    model.train()
    total_loss = 0.0
    desc = f"Train Epoch {epoch}" if epoch is not None else "Train"
    pbar = tqdm(loader, desc=desc, total=len(loader), leave=False)

    for i, (rgb, radar, mask, _) in enumerate(pbar):
        rgb, radar, mask = rgb.to(device), radar.to(device), mask.to(device)

        pred = model(rgb, radar)
        loss = seg_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (i + 1))

    return total_loss / len(loader)


@torch.no_grad()
def validate(loader, model, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    for rgb, radar, mask, _ in loader:
        rgb, radar, mask = rgb.to(device), radar.to(device), mask.to(device)
        pred = model(rgb, radar)
        total_loss += seg_loss(pred, mask).item()
        total_iou += iou_score(pred, mask)
    return total_loss / len(loader), total_iou / len(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root of the prepared dataset.")
    parser.add_argument("--out", default="checkpoints/best_glassformer.pt",
                        help="Where to save the best checkpoint.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=402)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test", action="store_true",
                        help="Evaluate the best checkpoint on the test split at the end.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(
        args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    model = GlassSegFormerRGBRadar().to(device)
    # Optimizer created once (not per epoch) so AdamW momentum persists.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    for epoch in range(args.epochs):
        tr_loss = train_epoch(train_loader, model, optimizer, device, epoch)
        val_loss, val_iou = validate(val_loader, model, device)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | "
              f"val loss {val_loss:.4f} | val IoU {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ saved best model (val IoU {best_iou:.4f}) → {out_path}")

    print(f"\nTraining done. Best val IoU: {best_iou:.4f}")

    if args.test:
        model.load_state_dict(torch.load(out_path, map_location=device))
        _, test_iou = validate(test_loader, model, device)
        print(f"Test IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()

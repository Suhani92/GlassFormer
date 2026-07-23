"""Segmentation losses and metrics for GlassFormer."""

import torch
import torch.nn as nn

_bce = nn.BCEWithLogitsLoss()


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension w.r.t. sorted errors (Alg. 1 in the paper)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover the 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """Lovasz hinge on flattened logits/labels.

    Args:
        logits: (N,) raw logits.
        labels: (N,) values in {0, 1}.
    """
    labels = labels.float()
    signs = 2.0 * labels - 1.0        # +1 for positive, -1 for negative
    errors = 1.0 - logits * signs     # hinge errors
    errors_sorted, perm = torch.sort(errors, descending=True)
    grad = lovasz_grad(labels[perm])
    return torch.dot(torch.relu(errors_sorted), grad)


def lovasz_hinge(logits, labels, per_image=True):
    """Lovasz hinge for (B, 1, H, W) logits and {0,1} labels."""
    if per_image:
        losses = [
            lovasz_hinge_flat(logit.view(-1), label.view(-1))
            for logit, label in zip(logits, labels)
        ]
        return torch.mean(torch.stack(losses))
    return lovasz_hinge_flat(logits.view(-1), labels.view(-1))


def seg_loss(pred, target):
    """Combined BCE + Lovasz hinge loss on raw logits (Eq. 8)."""
    target = target.float()
    return _bce(pred, target) + lovasz_hinge(pred, target)


@torch.no_grad()
def iou_score(pred, target):
    """Binary IoU from raw logits (threshold 0.5)."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter + 1e-6
    return (inter / union).item()

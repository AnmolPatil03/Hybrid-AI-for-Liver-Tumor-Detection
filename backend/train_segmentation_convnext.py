"""
Train the ConvNeXt-Tiny + U-Net liver tumor segmentation model on the full
LiTS dataset (131 volumes, 58,638 slices).

Pipeline
    1. Volume-level 70/15/15 split (src/lits_dataset.py), 2.5D input
       ([-1, 0, +1] neighbouring slices stacked as 3 channels).
    2. ConvNeXt-Tiny encoder (ImageNet-pretrained) + U-Net decoder.
    3. AdamW, BCE(pos_weight) + Tversky loss, ReduceLROnPlateau,
       early stopping on validation Dice.
    4. One evaluation pass on the held-out test volumes:
         - pixel-level Dice and IoU
         - slice-level tumor detection precision / recall / accuracy
       written to outputs/metrics/segmentation_metrics.txt, plus loss,
       Dice, ROC and confusion-matrix plots in outputs/plots/.

Data: download https://www.kaggle.com/datasets/andrewmvd/lits-png and
extract it to backend/data/lits_png/.

Usage
    python train_segmentation_convnext.py
"""

import os

# Lets unsupported ops fall back to CPU on Apple Silicon. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.segmentation_model_convnext import create_segmentation_model, CombinedLoss
from src.lits_dataset import get_lits_dataloaders_25d


CONFIG = {
    # Effective batch = micro_batch_size * grad_accum_steps = 32.
    # Gradient accumulation keeps peak memory low enough for a laptop GPU.
    "micro_batch_size": 8,
    "grad_accum_steps": 4,
    "learning_rate": 1e-4,
    "weight_decay": 2e-2,
    # Tversky with beta > alpha penalises false negatives more than false
    # positives, which helps recall on small tumors.
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "use_focal_tversky": False,
    "focal_gamma": 4.0 / 3.0,
    "bce_pos_weight": 4.0,
    "bce_weight": 0.5,
    "region_weight": 0.5,
    "epochs": 100,
    "patience": 15,
    "num_workers": 4,
    "save_dir": "outputs",
    "root": os.path.join("data", "lits_png", "dataset_6", "dataset_6"),
    "split_path": os.path.join("outputs", "splits", "segmentation_split.json"),
    "elastic_alpha": 20.0,
    "elastic_sigma": 4.0,
    "elastic_prob": 0.5,
    # A slice counts as "tumor predicted" when at least this many pixels are above threshold.
    "slice_min_tumor_pixels": 1,
    "threshold": 0.5,
    # Path to a "latest" or "best" checkpoint to resume training from.
    "resume_from": None,
    "seed": 42,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _patch_batchnorm_for_mps():
    """
    BatchNorm2d backward fails on MPS when its input is non-contiguous
    (happens after the skip-connection concat in the decoder). Making the
    input contiguous avoids it without changing the result.
    """
    original_forward = nn.BatchNorm2d.forward

    def forward_contiguous(self, x):
        return original_forward(self, x.contiguous())

    nn.BatchNorm2d.forward = forward_contiguous


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_score(pred_logits, target, threshold=0.5, smooth=1.0):
    pred = (torch.sigmoid(pred_logits) > threshold).float().reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps=1, log_every=50):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(loader)

    optimizer.zero_grad()
    for step, (images, masks) in enumerate(loader, start=1):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        (loss / accum_steps).backward()

        if step % accum_steps == 0 or step == num_batches:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_dice += dice_score(outputs, masks).item()

        if step % log_every == 0 or step == num_batches:
            print(f"    [batch {step:4d}/{num_batches}] "
                  f"loss={total_loss / step:.4f} dice={total_dice / step:.4f}")

    return total_loss / num_batches, total_dice / num_batches


def validate(model, loader, criterion, device, threshold=0.5):
    """Return (mean loss, global Dice) where Dice is computed from TP/FP/FN over all pixels."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()

            preds = (torch.sigmoid(outputs) > threshold).float()
            tp += int(((preds == 1) & (masks == 1)).sum().item())
            fp += int(((preds == 1) & (masks == 0)).sum().item())
            fn += int(((preds == 0) & (masks == 1)).sum().item())

    denom = 2 * tp + fp + fn
    return total_loss / len(loader), (2 * tp / denom if denom else 0.0)


def evaluate_test_split(model, loader, device, threshold=0.5, min_pixels=1):
    """
    Single pass over the test split.

    Pixel level: TP/FP/FN for Dice and IoU.
    Slice level: each slice is labelled tumor / no tumor from its ground
    truth mask and from the prediction, giving the detection confusion
    matrix. The max tumor probability per slice is kept for the ROC curve.
    """
    model.eval()
    px = {"tp": 0, "fp": 0, "fn": 0}
    sl = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    slice_scores, slice_labels = [], []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            probs = torch.sigmoid(model(images))
            preds = (probs > threshold).float()

            px["tp"] += int(((preds == 1) & (masks == 1)).sum().item())
            px["fp"] += int(((preds == 1) & (masks == 0)).sum().item())
            px["fn"] += int(((preds == 0) & (masks == 1)).sum().item())

            gt_pos = masks.flatten(1).sum(dim=1) > 0
            pred_pos = preds.flatten(1).sum(dim=1) >= min_pixels
            sl["tp"] += int((gt_pos & pred_pos).sum().item())
            sl["fp"] += int((~gt_pos & pred_pos).sum().item())
            sl["fn"] += int((gt_pos & ~pred_pos).sum().item())
            sl["tn"] += int((~gt_pos & ~pred_pos).sum().item())

            slice_scores.append(probs.flatten(1).max(dim=1).values.cpu().numpy())
            slice_labels.append(gt_pos.cpu().numpy().astype(np.int64))

    return px, sl, np.concatenate(slice_scores), np.concatenate(slice_labels)


def compute_metrics(px, sl):
    dice_denom = 2 * px["tp"] + px["fp"] + px["fn"]
    iou_denom = px["tp"] + px["fp"] + px["fn"]
    total = sum(sl.values())
    return {
        "dice_coefficient": 2 * px["tp"] / dice_denom if dice_denom else 0.0,
        "iou_jaccard": px["tp"] / iou_denom if iou_denom else 0.0,
        "precision": sl["tp"] / (sl["tp"] + sl["fp"]) if (sl["tp"] + sl["fp"]) else 0.0,
        "recall_sensitivity": sl["tp"] / (sl["tp"] + sl["fn"]) if (sl["tp"] + sl["fn"]) else 0.0,
        "accuracy": (sl["tp"] + sl["tn"]) / total if total else 0.0,
    }


def write_metrics_file(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("LIVER TUMOR SEGMENTATION — CONVNEXT-TINY U-NET — TEST SET RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"dice_coefficient (primary overlap metric) : {metrics['dice_coefficient']:.4f}\n")
        f.write(f"iou_jaccard                               : {metrics['iou_jaccard']:.4f}\n")
        f.write(f"precision                                 : {metrics['precision']:.4f}\n")
        f.write(f"recall / sensitivity                      : {metrics['recall_sensitivity']:.4f}\n")
        f.write(f"accuracy                                  : {metrics['accuracy']:.4f}\n")
        f.write("\n--- for reference, not independent evidence ---\n")
        f.write(f"f1_score (== dice_coefficient, same TP/FP/FN) : {metrics['dice_coefficient']:.4f}\n")
    print(f"[INFO] Metrics saved to: {save_path}")


def plot_loss_curve(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], lw=1.8, label="Training loss")
    ax.plot(epochs, history["val_loss"], lw=1.8, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE + Tversky loss")
    ax.set_title("ConvNeXt-Tiny 2.5D + U-Net: Loss Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dice_curve(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(history["val_dice"]) + 1)
    ax.plot(epochs, history["train_dice"], lw=1.8, label="Training Dice")
    ax.plot(epochs, history["val_dice"], lw=1.8, label="Validation Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim([0.0, 1.0])
    ax.set_title("ConvNeXt-Tiny 2.5D + U-Net: Dice", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(labels, scores, save_path):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ConvNeXt (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ConvNeXt-Tiny 2.5D + U-Net: ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(sl, save_path):
    cm = np.array([[sl["tp"], sl["fn"]], [sl["fp"], sl["tn"]]])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], ["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1], ["Actual Positive", "Actual Negative"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)
    ax.set_title("ConvNeXt-Tiny 2.5D + U-Net", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = get_device()
    print(f"[SETUP] Device: {device} | seed: {cfg['seed']}")
    if device.type == "mps":
        _patch_batchnorm_for_mps()

    plots_dir = os.path.join(cfg["save_dir"], "plots")
    models_dir = os.path.join(cfg["save_dir"], "models")
    metrics_dir = os.path.join(cfg["save_dir"], "metrics")
    for d in (plots_dir, models_dir, metrics_dir):
        os.makedirs(d, exist_ok=True)

    print("\n[DATA] Loading full LiTS dataset (2.5D)...")
    train_loader, val_loader, test_loader, info = get_lits_dataloaders_25d(
        root=cfg["root"],
        split_path=cfg["split_path"],
        batch_size=cfg["micro_batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        elastic_alpha=cfg["elastic_alpha"],
        elastic_sigma=cfg["elastic_sigma"],
        elastic_prob=cfg["elastic_prob"],
    )
    print(f"[DATA] {info['total_slices']} slices | "
          f"train {info['train_slices']} ({info['train_volumes']} vols) | "
          f"val {info['val_slices']} ({info['val_volumes']} vols) | "
          f"test {info['test_slices']} ({info['test_volumes']} vols)")
    print(f"[DATA] Effective batch size: {cfg['micro_batch_size'] * cfg['grad_accum_steps']}")

    model = create_segmentation_model(pretrained=True).to(device)

    pos_weight = torch.tensor(cfg["bce_pos_weight"]) if cfg["bce_pos_weight"] is not None else None
    criterion = CombinedLoss(
        bce_weight=cfg["bce_weight"], region_weight=cfg["region_weight"],
        alpha=cfg["tversky_alpha"], beta=cfg["tversky_beta"],
        pos_weight=pos_weight, use_focal=cfg["use_focal_tversky"], gamma=cfg["focal_gamma"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_path = os.path.join(models_dir, "best_segmentation_convnext_model.pth")
    latest_path = os.path.join(models_dir, "latest_segmentation_convnext_model.pth")

    start_epoch = 1
    best_val_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    resume_path = cfg.get("resume_from")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", ckpt.get("val_dice", 0.0))
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", history)
        print(f"[RESUME] From epoch {start_epoch} (best val Dice {best_val_dice:.4f})")
    elif resume_path:
        print(f"[WARN] Checkpoint '{resume_path}' not found, starting from scratch.")

    print(f"\n[TRAIN] Epochs {start_epoch}-{cfg['epochs']}")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        start = time.time()
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, accum_steps=cfg["grad_accum_steps"],
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device, cfg["threshold"])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
              f"val loss {val_loss:.4f} dice {val_dice:.4f} | {time.time() - start:.0f}s")

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_dice": val_dice,
            "best_val_dice": best_val_dice,
            "patience_counter": patience_counter,
            "history": history,
        }
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"  saved best model (val dice {val_dice:.4f})")

        if patience_counter >= cfg["patience"]:
            print(f"\n[EARLY STOP] No improvement for {cfg['patience']} epochs.")
            break

    if history["train_loss"]:
        plot_loss_curve(history, os.path.join(plots_dir, "segmentation_loss_curve.png"))
        plot_dice_curve(history, os.path.join(plots_dir, "segmentation_dice_curve.png"))

    print("\n[EVAL] Evaluating best checkpoint on the test split...")
    eval_path = best_path if os.path.exists(best_path) else latest_path
    ckpt = torch.load(eval_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    px, sl, slice_scores, slice_labels = evaluate_test_split(
        model, test_loader, device, cfg["threshold"], cfg["slice_min_tumor_pixels"],
    )
    metrics = compute_metrics(px, sl)
    write_metrics_file(metrics, os.path.join(metrics_dir, "segmentation_metrics.txt"))
    plot_confusion_matrix(sl, os.path.join(plots_dir, "segmentation_confusion_matrix.png"))
    plot_roc_curve(slice_labels, slice_scores, os.path.join(plots_dir, "segmentation_roc_curve.png"))

    print("\nTEST RESULTS")
    for key, value in metrics.items():
        print(f"  {key:20s}: {value:.4f}")


if __name__ == "__main__":
    main()

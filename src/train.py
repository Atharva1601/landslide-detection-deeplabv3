import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import LandslideDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.model import get_model
from src.utils import BCEDiceLoss, iou_score, dice_score
import os


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = LandslideDataset(
        "data/splits/train.txt",
        "data/processed/images",
        "data/processed/masks",
        transforms=get_train_transforms(256),
    )

    val_ds = LandslideDataset(
        "data/splits/val.txt",
        "data/processed/images",
        "data/processed/masks",
        transforms=get_val_transforms(256),
    )

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
    )

    model = get_model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = BCEDiceLoss()

    os.makedirs("outputs/checkpoints", exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(10):
        # -------- TRAIN --------
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/10 [Train]")

        for imgs, masks in loop:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            preds = model(imgs)["out"]
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        total_iou = 0
        total_dice = 0

        loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/10 [Val]")

        with torch.no_grad():
            for imgs, masks in loop:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                preds = model(imgs)["out"]
                loss = criterion(preds, masks)

                val_loss += loss.item()
                total_iou += iou_score(preds, masks).item()
                total_dice += dice_score(preds, masks).item()

                loop.set_postfix(val_loss=loss.item())

        val_loss /= len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_dice = total_dice / len(val_loader)

        print(
            f"\nEpoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"IoU={avg_iou:.4f}, "
            f"Dice={avg_dice:.4f}"
        )

        # -------- SAVE BEST --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/checkpoints/best_model.pth")
            print("✅ Best model saved")

    print("Training complete 🚀")

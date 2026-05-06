import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.dataset import LandslideDataset
from src.transforms import get_val_transforms
from src.model import get_model
from src.utils import iou_score, dice_score


def evaluate(save_viz=False, num_samples=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = LandslideDataset(
        "data/splits/test.txt",
        "data/processed/images",
        "data/processed/masks",
        transforms=get_val_transforms(256),
    )

    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
    )

    model = get_model().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
    model.eval()

    total_iou = 0
    total_dice = 0

    os.makedirs("outputs/predictions", exist_ok=True)

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(loader, desc="Testing")):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            preds = model(imgs)["out"]

            total_iou += iou_score(preds, masks).item()
            total_dice += dice_score(preds, masks).item()

            # ---------- OPTIONAL VISUALIZATION ----------
            if save_viz and i < num_samples:
                for j in range(len(imgs)):
                    pred = torch.sigmoid(preds[j])
                    pred = (pred > 0.5).float()

                    img = imgs[j].permute(1, 2, 0).cpu().numpy()
                    gt = masks[j].squeeze().cpu().numpy()
                    pr = pred.squeeze().cpu().numpy()

                    plt.figure(figsize=(10, 3))

                    plt.subplot(1, 3, 1)
                    plt.imshow(img)
                    plt.title("Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(gt, cmap="gray")
                    plt.title("GT")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(pr, cmap="gray")
                    plt.title("Pred")
                    plt.axis("off")

                    plt.savefig(f"outputs/predictions/sample_{i}_{j}.png")
                    plt.close()

    avg_iou = total_iou / len(loader)
    avg_dice = total_dice / len(loader)

    print("\n===== TEST RESULTS =====")
    print(f"IoU  : {avg_iou:.4f}")
    print(f"Dice : {avg_dice:.4f}")

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.model import get_model
from src.transforms import get_val_transforms


def predict(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- LOAD MODEL ----------
    model = get_model().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
    model.eval()

    # ---------- LOAD IMAGE ----------
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---------- TRANSFORM ----------
    transform = get_val_transforms(256)

    augmented = transform(image=image_rgb)
    img = augmented["image"].unsqueeze(0).to(device)

    # ---------- PREDICTION ----------
    with torch.no_grad():
        pred = model(img)["out"]

        prob = torch.sigmoid(pred)

        # confidence score
        confidence = prob.max().item()

        pred = (prob > 0.5).float()

    mask = pred.squeeze().cpu().numpy()

    # ---------- CLASSIFICATION ----------
    landslide_pixels = mask.sum()

    if landslide_pixels > 50:
        label = "LANDSLIDE DETECTED"
    else:
        label = "NO LANDSLIDE"

    print(f"\nPrediction : {label}")
    print(f"Confidence : {confidence:.4f}")

    # ---------- OVERLAY ----------
    mask_resized = cv2.resize(
        mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    overlay = image_rgb.copy()

    # red overlay
    overlay[mask_resized > 0] = [255, 0, 0]

    # ---------- VISUALIZATION ----------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_resized, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"{label}\nConf: {confidence:.2f}")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

import torch
import cv2
import os
import matplotlib.pyplot as plt
from src.dataset import LandslideDataset
from src.transforms import get_val_transforms
from src.model import get_model


def visualize(num_samples=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LandslideDataset(
        "data/splits/test.txt",
        "data/processed/images",
        "data/processed/masks",
        transforms=get_val_transforms(256),
    )

    model = get_model().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
    model.eval()

    os.makedirs("outputs/predictions", exist_ok=True)

    for i in range(num_samples):
        img, mask = dataset[i]

        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_input)["out"]
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

        pred = pred.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()

        # plot
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred, cmap="gray")
        plt.axis("off")

        save_path = f"outputs/predictions/sample_{i}.png"
        plt.savefig(save_path)
        plt.close()

    print("✅ Visualization saved in outputs/predictions/")

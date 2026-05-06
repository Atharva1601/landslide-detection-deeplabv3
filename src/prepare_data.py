import os
import cv2
from tqdm import tqdm

RAW_PATH = "data/raw/Bijie-landslide-dataset"
OUT_IMG = "data/processed/images"
OUT_MASK = "data/processed/masks"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

# ---------- LANDSLIDE ----------
ls_img_path = os.path.join(RAW_PATH, "landslide", "image")
ls_mask_path = os.path.join(RAW_PATH, "landslide", "mask")

ls_images = sorted(os.listdir(ls_img_path))

for i, name in enumerate(tqdm(ls_images, desc="Landslide")):
    img = cv2.imread(os.path.join(ls_img_path, name))
    mask = cv2.imread(os.path.join(ls_mask_path, name), 0)

    new_name = f"ls_{i:04d}.png"

    cv2.imwrite(os.path.join(OUT_IMG, new_name), img)
    cv2.imwrite(os.path.join(OUT_MASK, new_name), mask)


# ---------- NON-LANDSLIDE ----------
nls_img_path = os.path.join(RAW_PATH, "non-landslide", "image")

nls_images = sorted(os.listdir(nls_img_path))

for i, name in enumerate(tqdm(nls_images, desc="Non-Landslide")):
    img = cv2.imread(os.path.join(nls_img_path, name))

    h, w, _ = img.shape
    mask = 0 * img[:, :, 0]  # all zeros

    new_name = f"nls_{i:04d}.png"

    cv2.imwrite(os.path.join(OUT_IMG, new_name), img)
    cv2.imwrite(os.path.join(OUT_MASK, new_name), mask)

print("DONE ✅")

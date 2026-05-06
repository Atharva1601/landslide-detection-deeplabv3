import os
import gdown
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from src.model import get_model
from src.transforms import get_val_transforms


# ---------- DOWNLOAD MODEL ----------
MODEL_PATH = "outputs/checkpoints/best_model.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs("outputs/checkpoints", exist_ok=True)

    file_id = "1T-MgN3Zmx7LNLp-5XgZo8vdfhAqvQo_m"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, MODEL_PATH, quiet=False)


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Landslide Detection", layout="wide")

st.title("🌍 Landslide Detection using DeepLabV3")


# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():

    model = get_model().to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.eval()

    return model


model = load_model()


# ---------- FILE UPLOADER ----------
uploaded_file = st.file_uploader("Upload Satellite Image", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    # ---------- READ IMAGE ----------
    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    st.subheader("Uploaded Image")

    st.image(image_np, use_container_width=True)

    # ---------- TRANSFORM ----------
    transform = get_val_transforms(256)

    augmented = transform(image=image_np)

    img_tensor = augmented["image"].unsqueeze(0).to(device)

    # ---------- PREDICTION ----------
    with torch.no_grad():
        pred = model(img_tensor)["out"]

        prob = torch.sigmoid(pred)

        pred = (prob > 0.5).float()

    mask = pred.squeeze().cpu().numpy()

    # ---------- LANDSLIDE RATIO ----------
    landslide_ratio = mask.sum() / mask.size

    # ---------- CLASSIFICATION ----------
    if landslide_ratio > 0.01:
        label = "LANDSLIDE DETECTED"

        confidence = min(landslide_ratio * 100 * 5, 99)

    else:
        label = "NO LANDSLIDE"

        confidence = (1 - landslide_ratio) * 100

    # ---------- RESIZE MASK ----------
    mask_resized = cv2.resize(
        mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # ---------- OVERLAY ----------
    overlay = image_np.copy()

    overlay[mask_resized > 0] = [255, 0, 0]

    # ---------- RESULTS ----------
    st.subheader("Prediction Result")

    if label == "LANDSLIDE DETECTED":
        st.error(label)
    else:
        st.success(label)

    st.write(f"### Confidence: {confidence:.2f}%")

    # ---------- DISPLAY ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Mask")
        st.image(mask_resized)

    with col2:
        st.subheader("Overlay Detection")
        st.image(overlay)

    # ---------- EXTRA INFO ----------
    st.write(f"Detected Landslide Area Ratio: {landslide_ratio:.4f}")

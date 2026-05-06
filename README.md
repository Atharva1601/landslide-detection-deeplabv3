# 🌍 DeepLand: Landslide Detection & Segmentation using DeepLabV3

## 🚀 Live Demo

🔗 [https://atharva1601-landslide-detection-deeplabv3-app-vflxfy.streamlit.app/](https://atharva1601-landslide-detection-deeplabv3-app-vflxfy.streamlit.app/)

---

# 📌 Overview

DeepLand is an end-to-end deep learning system for landslide detection and semantic segmentation using DeepLabV3-ResNet50.

The project takes satellite imagery as input and:

* Detects whether a landslide is present
* Segments landslide regions pixel-by-pixel
* Generates visual overlays
* Provides prediction confidence scores
* Supports real-time inference through a deployed Streamlit web application

This project was built using PyTorch and deployed using Streamlit Community Cloud.

---

# 🧠 Problem Statement

Landslides are one of the major natural disasters affecting mountainous and hilly regions worldwide. Rapid and accurate landslide detection from satellite imagery can help:

* Disaster management teams
* Geospatial analysts
* Environmental monitoring systems
* Remote sensing applications
* Early warning systems

Traditional manual interpretation of satellite images is slow and labor-intensive.

DeepLand automates this process using semantic segmentation with DeepLabV3.

---

# 🎯 Project Objectives

* Build an end-to-end landslide segmentation system
* Train DeepLabV3 on satellite imagery
* Perform pixel-level segmentation of landslide regions
* Deploy the model as an interactive web application
* Provide visual overlays and prediction confidence
* Create a production-style deep learning workflow

---

# 🏗️ System Architecture

```text
Satellite Image
       ↓
Image Preprocessing
       ↓
DeepLabV3-ResNet50
       ↓
Segmentation Mask
       ↓
Landslide Classification Logic
       ↓
Confidence Estimation
       ↓
Visualization + Overlay
       ↓
Streamlit Deployment
```

---

# 📂 Dataset

Dataset Used:

Bijie Landslide Dataset

Source:
[https://www.kaggle.com/datasets/hanstankman/bijie-landslidedataset/data](https://www.kaggle.com/datasets/hanstankman/bijie-landslidedataset/data)

## Dataset Structure

```text
Bijie-landslide-dataset/
├── landslide/
│   ├── dem/
│   ├── image/
│   ├── mask/
│   └── polygon_coordinate/
└── non-landslide/
    ├── dem/
    └── image/
```

## Dataset Information

* RGB satellite imagery
* Binary segmentation masks
* DEM data available
* Both landslide and non-landslide samples

---

# 🛠️ Tech Stack

## Deep Learning

* PyTorch
* Torchvision
* DeepLabV3

## Computer Vision

* OpenCV
* Albumentations

## Visualization

* Matplotlib

## Deployment

* Streamlit
* gdown

## Utilities

* NumPy
* tqdm
* scikit-learn

---

# 🧩 Model Architecture

## DeepLabV3

This project uses:

```text
DeepLabV3-ResNet50
```

### Why DeepLabV3?

DeepLabV3 is a powerful semantic segmentation architecture that:

* Captures multi-scale contextual information
* Uses Atrous Spatial Pyramid Pooling (ASPP)
* Produces accurate segmentation boundaries
* Performs well on remote sensing tasks

### Backbone

```text
ResNet50
```

### Output

The classifier head was modified for binary segmentation:

```python
nn.Conv2d(256, 1, kernel_size=1)
```

---

# ⚙️ Training Pipeline

## Data Preprocessing

Images were:

* Resized to 256×256
* Normalized
* Converted to tensors

## Data Augmentation

Albumentations was used for:

* Horizontal Flip
* Vertical Flip
* Rotation
* Normalization

## Loss Function

Combined loss:

```text
BCE Loss + Dice Loss
```

This improves:

* Class imbalance handling
* Segmentation quality
* Boundary learning

## Optimizer

```text
AdamW
```

## Batch Size

```text
8
```

## Epochs

```text
10
```

---

# 📊 Evaluation Metrics

The model was evaluated using:

## IoU (Intersection over Union)

Measures overlap between prediction and ground truth.

## Dice Score

Measures segmentation similarity.

---

# 📈 Results

## Validation Performance

| Metric     | Score       |
| ---------- | ----------- |
| IoU        | 0.62 – 0.68 |
| Dice Score | 0.75 – 0.79 |

These results demonstrate strong segmentation performance on the dataset.

---

# 🔍 Features

## ✅ Landslide Detection

Classifies whether a landslide exists in the image.

## ✅ Semantic Segmentation

Performs pixel-level segmentation.

## ✅ Overlay Visualization

Highlights landslide regions in red.

## ✅ Confidence Score

Displays prediction confidence.

## ✅ Streamlit Deployment

Interactive web-based inference system.

---

# 🖼️ Sample Workflow

## Input

Satellite image uploaded by user.

## Prediction

Model predicts segmentation mask.

## Output

* Landslide / No Landslide
* Confidence score
* Segmentation mask
* Overlay visualization

---

# 🚀 Streamlit Deployment

The application was deployed using Streamlit Community Cloud.

## Features of Web App

* Upload satellite image
* Real-time prediction
* Visual segmentation masks
* Overlay visualization
* Confidence estimation

## Deployment Pipeline

```text
GitHub Repository
        ↓
Streamlit Cloud
        ↓
Automatic Model Download (Google Drive)
        ↓
Inference Pipeline
```

---

# 📁 Project Structure

```text
landslide-segmentation/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── processed/
│   ├── raw/
│   └── splits/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
│
├── notebooks/
│   └── exploration.ipynb
│
└── src/
    ├── dataset.py
    ├── eval.py
    ├── inference.py
    ├── model.py
    ├── train.py
    ├── transforms.py
    ├── utils.py
    └── __init__.py
```

---

# ⚡ Installation

## Clone Repository

```bash
git clone https://github.com/Atharva1601/landslide-detection-deeplabv3.git
```

## Create Environment

```bash
conda create -n dlls python=3.10
conda activate dlls
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## Train Model

```bash
python main.py --mode train
```

## Evaluate Model

```bash
python main.py --mode eval
```

## Evaluate + Visualization

```bash
python main.py --mode eval --viz
```

## Predict on New Image

```bash
python main.py --mode predict --image path/to/image.png
```

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🔬 Future Improvements

Potential future upgrades:

* DEM + RGB multimodal fusion
* Attention-enhanced DeepLabV3+
* Better confidence calibration
* Lightweight deployment models
* MobileNet backbone
* Quantization and pruning
* Cloud deployment
* Real-time monitoring system
* Temporal satellite analysis

---

# 📚 Research Relevance

This project relates to:

* Computer Vision
* Semantic Segmentation
* Remote Sensing
* Geospatial AI
* Disaster Intelligence Systems
* Environmental Monitoring

---

# 👨‍💻 Author

Atharva

GitHub:
[https://github.com/Atharva1601](https://github.com/Atharva1601)

---

# ⭐ Acknowledgements

* PyTorch
* Streamlit
* Kaggle Dataset Contributors
* DeepLabV3 Research Community

---

# 📜 License

This project is intended for educational and research purposes.

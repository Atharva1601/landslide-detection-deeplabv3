import torch.nn as nn
import torchvision.models as models


def get_model():
    model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")

    # change classifier to 1 output channel
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    return model

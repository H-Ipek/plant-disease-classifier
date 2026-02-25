import torch.nn as nn
from torchvision import models


def build_model(num_classes: int):
    """
    EfficientNet-B0 tabanlı transfer learning modeli.
    ImageNet ağırlıkları yüklenir, son katman veri setine göre ayarlanır.
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Tüm katmanları dondur (feature extraction için)
    for param in model.parameters():
        param.requires_grad = False

    # Son sınıflandırıcı katmanını değiştir
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
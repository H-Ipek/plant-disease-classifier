import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# ── Sınıf isimlerini yükle ────────────────────────────────────────
with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)

# ── Cihaz ayarı ───────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Modeli yükle ──────────────────────────────────────────────────
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval().to(device)

# ── Transform ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Tahmin fonksiyonu ─────────────────────────────────────────────
def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    top5 = probs.topk(5)
    return {CLASS_NAMES[i]: float(p)
            for i, p in zip(top5.indices, top5.values)}

# ── Arayüz ────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Bitki yaprağı fotoğrafı yükle"),
    outputs=gr.Label(num_top_classes=5, label="Tahmin"),
    title="🌿 Plant Disease Detector",
    description="Bir bitki yaprağı fotoğrafı yükle, model hastalık durumunu tahmin etsin. 38 farklı sınıf — %97 doğruluk.",
)

demo.launch()
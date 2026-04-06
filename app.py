
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASSES      = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
MODEL_PATH   = 'best_robust_model.pth'

# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class RobustCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stage1      = nn.Sequential(ResBlock(32, 32),   ResBlock(32, 32))
        self.downsample1 = ResBlock(32, 64, stride=2)
        self.stage2      = nn.Sequential(ResBlock(64, 64),   ResBlock(64, 64))
        self.downsample2 = ResBlock(64, 128, stride=2)
        self.stage3      = nn.Sequential(ResBlock(128, 128), ResBlock(128, 128))
        self.avg_pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout     = nn.Dropout(0.3)
        self.fc          = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ─────────────────────────────────────────────
# ADVERSARIAL DETECTOR
# ─────────────────────────────────────────────
class AdversarialDetector:
    def __init__(self, model):
        self.model = model
        self.mean  = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(DEVICE)
        self.std   = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(DEVICE)

    def _bit_reduce(self, x, bits):
        d = x * self.std + self.mean
        q = 2 ** bits - 1
        return (torch.round(d * q) / q - self.mean) / self.std

    def _smooth(self, x, k=3):
        pad = k // 2
        return F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode='reflect'),
            k, stride=1
        )

    def get_pred(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x).argmax(1).item()

    def detect(self, tensor):
        """
        tensor: 3D (C,H,W) or 4D (1,C,H,W)
        Returns True if adversarial, False if clean
        """
        self.model.eval()
        img = tensor.to(DEVICE)
        if img.dim() == 3:
            img = img.unsqueeze(0)   # always make 4D

        orig = self.get_pred(img)
        squeezed_preds = [
            self.get_pred(self._bit_reduce(img, 5)),
            self.get_pred(self._bit_reduce(img, 4)),
            self.get_pred(self._bit_reduce(img, 3)),
            self.get_pred(self._smooth(img, 3)),
            self.get_pred(self._smooth(img, 5)),
        ]
        changes = sum(p != orig for p in squeezed_preds)
        return changes >= 2


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model    = RobustCNN().to(DEVICE)
detector = None

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    detector = AdversarialDetector(model)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Device: {DEVICE}")
else:
    print(f"⚠  Model not found: {MODEL_PATH}")
    print("   Place best_robust_model.pth in the same folder as app.py")


# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ─────────────────────────────────────────────
# HELPER — decode base64 → 3D tensor (C,H,W)
# ─────────────────────────────────────────────
def decode_image(b64_string):
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return transform(img).to(DEVICE)   # (3, 32, 32)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': os.path.exists(MODEL_PATH),
        'device':       str(DEVICE),
        'classes':      CLASSES,
        'accuracy': {
            'clean': 91.20,
            'fgsm':  76.58,
            'pgd20': 73.29,
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not loaded. Place best_robust_model.pth in app folder.'}), 500
    try:
        data   = request.get_json()
        tensor = decode_image(data['image'])        # (3, 32, 32)

        model.eval()
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))     # (1, 3, 32, 32)
            probs  = F.softmax(logits, dim=1)[0]    # (10,)

        top_idx   = probs.argmax().item()
        top_class = CLASSES[top_idx]
        top_conf  = round(probs[top_idx].item() * 100, 2)
        all_probs = {CLASSES[i]: round(probs[i].item() * 100, 2) for i in range(10)}

        is_adversarial = detector.detect(tensor) if detector else False

        return jsonify({
            'prediction':     top_class,
            'confidence':     top_conf,
            'all_probs':      all_probs,
            'is_adversarial': is_adversarial,
            'status':         'adversarial' if is_adversarial else 'clean',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/simulate_attack', methods=['POST'])
def simulate_attack():
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not loaded.'}), 500
    try:
        data   = request.get_json()
        tensor = decode_image(data['image'])        # (3, 32, 32)

        # ── Clean prediction ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))
            probs  = F.softmax(logits, dim=1)[0]

        clean_pred  = CLASSES[probs.argmax().item()]
        clean_conf  = round(probs.max().item() * 100, 2)
        clean_label = torch.tensor([probs.argmax().item()]).to(DEVICE)

        # ── FGSM attack ───────────────────────────────────────────────────────
        eps       = 8 / 255
        # unsqueeze to 4D for model forward, then squeeze grad back to 3D
        adv_input = tensor.unsqueeze(0).clone().detach().requires_grad_(True)  # (1,3,32,32)

        model.train()
        loss = nn.CrossEntropyLoss()(model(adv_input), clean_label)
        loss.backward()

        with torch.no_grad():
            grad_sign  = adv_input.grad.data.squeeze(0).sign()          # (3,32,32)
            adv_tensor = (tensor + eps * grad_sign).clamp(0, 1)         # (3,32,32)

        model.eval()

        # ── Adversarial prediction ─────────────────────────────────────────────
        with torch.no_grad():
            adv_logits = model(adv_tensor.unsqueeze(0))
            adv_probs  = F.softmax(adv_logits, dim=1)[0]

        adv_pred = CLASSES[adv_probs.argmax().item()]
        adv_conf = round(adv_probs.max().item() * 100, 2)

        # ── Detection ─────────────────────────────────────────────────────────
        is_adv_detected = detector.detect(adv_tensor) if detector else False

        return jsonify({
            'clean': {
                'prediction': clean_pred,
                'confidence': clean_conf,
            },
            'adversarial': {
                'prediction': adv_pred,
                'confidence': adv_conf,
                'detected':   is_adv_detected,
            },
            'attack_success': clean_pred != adv_pred,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)

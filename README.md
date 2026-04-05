# C002 — Adversarial Shield Website

A full-stack web application for the C002 Adversarially Robust AI System.

## Project Structure

```
c002_website/
├── app.py                  ← Flask backend server
├── requirements.txt        ← Python dependencies
├── best_robust_model.pth   ← Your trained ML model (place here!)
├── README.md
└── static/
    └── index.html          ← Frontend (HTML + CSS + JS)
```

---

## Setup Instructions

### Step 1 — Place your model file
Copy `best_robust_model.pth` (downloaded from Colab) into this folder:
```
c002_website/best_robust_model.pth
```

### Step 2 — Install dependencies
```bash
pip install flask flask-cors torch torchvision pillow numpy
```

### Step 3 — Run the server
```bash
cd c002_website
python app.py
```

### Step 4 — Open the website
Open your browser and go to:
```
http://127.0.0.1:5000
```

---

## Features

- Upload any image and classify it into 10 CIFAR-10 categories
- See confidence scores for all 10 classes with animated bars
- Adversarial detection — flags suspicious inputs at runtime
- Simulate FGSM attack and see if the model defends itself
- Beautiful cyberpunk UI with particle animations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/status` | GET | Model status + accuracy stats |
| `/api/predict` | POST | Classify uploaded image |
| `/api/simulate_attack` | POST | Run FGSM attack simulation |

## Model Info

- Architecture: RobustCNN (ResNet-style, 1M parameters)
- Dataset: CIFAR-10 (10 classes)
- Training: 60 epochs adversarial training on A100 GPU
- Clean Accuracy: 91.20%
- FGSM Accuracy: 76.58%
- PGD-20 Accuracy: 73.29%

## Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

# app.py - 100% Pure-Python Flask Web App (Bypasses Windows DLL & Pandas blocks)
import io
import json
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 1. Load Class Names
with open("class_names.json", "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# 2. Define the Exact Same Model Architecture
class CropDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(CropDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.features(x))

# 3. Load the Trained Weights (.pth file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropDiseaseCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("crop_disease_cnn.pth", map_location=device, weights_only=True))
model.eval()

# 4. Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Embedded HTML Template (No extra HTML/CSS files needed)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>SDG 2 - Crop Disease Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 650px; margin: 40px auto; text-align: center; background: #f4f7f6; padding: 20px; }
        .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h2 { color: #2e7d32; margin-top: 0; }
        input[type="file"] { margin: 15px 0; }
        button { background: #2e7d32; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1b5e20; }
        .result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 8px; }
        img { max-width: 240px; border-radius: 8px; margin-top: 15px; border: 2px solid #2e7d32; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Crop Disease Detection Prototype</h2>
        <p><b>UN SDG 2: Zero Hunger</b> | PyTorch CNN Model</p>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="leaf_image" accept="image/*" required><br>
            <button type="submit">Predict Disease</button>
        </form>

        {% if prediction %}
        <div class="result">
            <img src="data:image/jpeg;base64,{{ img_data }}" alt="Uploaded Leaf">
            <h3>Prediction: <span style="color:#1b5e20;">{{ prediction }}</span></h3>
            <p><b>Confidence:</b> {{ confidence }}%</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, img_data = None, None, None
    if request.method == "POST":
        file = request.files.get("leaf_image")
        if file:
            raw_bytes = file.read()
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            
            # Preprocess & Predict
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                
            prediction = class_names[pred_idx.item()]
            confidence = f"{conf.item() * 100:.2f}"
            img_data = base64.b64encode(raw_bytes).decode("utf-8")
            
    return render_template_string(HTML_PAGE, prediction=prediction, confidence=confidence, img_data=img_data)

if __name__ == "__main__":
    print("Starting Offline Server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
import os
import torch
import torchvision.models as models
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from torch.autograd import Variable
from flask_cors import CORS
import uuid
from io import BytesIO


app = Flask(__name__)
CORS(app)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"C:\Users\chand\Desktop\FlaskExtension\flask_backend\model\modelaug.pth"
    num_classes = 2
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
    exit(1)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']

    try:
        image_bytes = image_file.read()
        print("Received image size (bytes):", len(image_bytes))
        image_file.seek(0)

        image = Image.open(image_file.stream).convert('RGB')
    except UnidentifiedImageError:
        print("Unidentified image file")
        return jsonify({"error": "Unidentified image file"}), 400
    except Exception as e:
        print("Error opening image:", e)
        return jsonify({"error": str(e)}), 500

    try:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        print("Image tensor shape:", image_tensor.shape)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)
            print("Prediction result:", predicted_class.item())
        label = "fake" if predicted_class.item() == 0 else "genuine"
        
        return jsonify({"result": label})

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

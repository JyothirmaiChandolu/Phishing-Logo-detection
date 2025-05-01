import numpy as np 
import pandas as pd 
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd

# Define dataset path
dataset_path = "/kaggle/input/logos-dataset/dataset/train"

# Initialize lists
image_paths = []
labels = []

# Assign labels based on folder name
for label, category in enumerate(["fake", "genuine"]):  # 0 = Fake, 1 = Real
    category_path = os.path.join(dataset_path, category)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Add image path and label to lists
        image_paths.append(img_path)
        labels.append(label)

# Convert to a DataFrame (for visualization or saving)
df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.head())  # View first few labeled samples

# Optional: Save labeled data to a CSV file
df.to_csv("labeled_dataset.csv", index=False)
import os
import pandas as pd

# Define dataset path
dataset_path = "/kaggle/input/logos-dataset/dataset/test"

# Initialize lists
image_paths = []
labels = []

# Assign labels based on folder name
for label, category in enumerate(["fake", "genuine"]):  # 0 = Fake, 1 = Real
    category_path = os.path.join(dataset_path, category)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Add image path and label to lists
        image_paths.append(img_path)
        labels.append(label)

# Convert to a DataFrame (for visualization or saving)
df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.head())  # View first few labeled samples

# Optional: Save labeled data to a CSV file
df.to_csv("labeled_test_dataset.csv", index=False)

dataset_path = "/kaggle/input/logos-dataset/dataset/test"

# Initialize lists
image_paths = []
labels = []

# Assign labels based on folder name
for label, category in enumerate(["fake", "genuine"]):  # 0 = Fake, 1 = Real
    category_path = os.path.join(dataset_path, category)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Add image path and label to lists
        image_paths.append(img_path)
        labels.append(label)

# Convert to a DataFrame (for visualization or saving)
df = pd.DataFrame({"image_path": image_paths, "label": labels})
print(df.head())  # View first few labeled samples

# Optional: Save labeled data to a CSV file
df.to_csv("labeled_test_dataset.csv", index=False)

# Load pretrained ResNet-50 model
resnet = models.resnet50(weights=True)

# Modify the last layer for your number of classes
num_classes = 2  # Change according to your dataset
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_labels = []
        self.image_paths = []

        # Read labeled images (assuming format: "label_imageName.jpg")
        for file_name in os.listdir(img_dir):
            label = 0 if "fake" in file_name else 1  # Example: filenames contain "fake" or "real"
            self.image_labels.append(label)
            self.image_paths.append(os.path.join(img_dir, file_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.image_labels[idx]
        return image, label

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    transforms.RandomAdjustSharpness(2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CustomImageDataset(img_dir="/kaggle/input/logos-dataset/dataset/train", transform=train_transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define image transformations for ResNet
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    transforms.RandomAdjustSharpness(2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define dataset path (Ensure structure is: train/FAKE/, train/REAL/)
dataset_path = "/kaggle/input/logos-dataset/dataset/train"

# Load dataset using ImageFolder (Auto-labels FAKE as 0, REAL as 1)
train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Print class-to-index mapping
print(train_dataset.class_to_idx)  # Output: {'FAKE': 0, 'REAL': 1}

# Test loading one batch
images, labels = next(iter(train_loader))
print(f"Batch image shape: {images.shape}, Batch labels: {labels}")

import cv2
import os
import numpy as np
input_dir = "/kaggle/input/logos-dataset/dataset" 
output_dir = "/kaggle/working/preprocssed_data"
folders = ["train/fake", "train/genuine", "test/fake", "test/genuine"]
for folder in folders:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

def unsharp_mask(image):
    """Apply unsharp masking for mild blur correction"""
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened
def laplacian_sharpen(image):
    """Apply Laplacian filter for edge enhancement"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened = cv2.subtract(image, laplacian.astype(np.uint8))
    return sharpened
for folder in folders:
    input_folder = os.path.join(input_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        sharpened_img = unsharp_mask(img)
        sharpened_img = laplacian_sharpen(sharpened_img)
        cv2.imwrite(output_path, sharpened_img)
print("Image preprocessing completed! Processed images saved in 'processed_dataset'.")

import torch.optim as optim
import torch.nn as nn

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    resnet.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

print("Training complete!")


resnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")


def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = train_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    print(output)

    return "Fake" if predicted.item() == 0 else "Real"

# Test on a new image
print(predict_image("/kaggle/input/layscvbnm/received_test.jpg", resnet))

resnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    #probs = torch.nn.functional.softmax(outputs, dim=1)
    #print("Prediction Probabilities:", probs)

        #print(f"labels:{labels}")

print(f"Accuracy: {100 * correct / total:.2f}%")

# Define image transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet input
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define dataset path (Ensure structure is: train/FAKE/, train/REAL/)
dataset_path = "/kaggle/input/logos-dataset/dataset/test"

# Load dataset using ImageFolder (Auto-labels FAKE as 0, REAL as 1)
test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Print class-to-index mapping
print(test_dataset.class_to_idx)  # Output: {'FAKE': 0, 'REAL': 1}

# Test loading one batch
images, labels = next(iter(test_loader))
print(f"Batch image shape: {images.shape}, Batch labels: {labels}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet input
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define dataset path (Ensure structure is: train/FAKE/, train/REAL/)
dataset_path = "/kaggle/input/logos-dataset/dataset/test"

# Load dataset using ImageFolder (Auto-labels FAKE as 0, REAL as 1)
test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Print class-to-index mapping
print(test_dataset.class_to_idx)  # Output: {'FAKE': 0, 'REAL': 1}

# Test loading one batch
images, labels = next(iter(test_loader))
print(f"Batch image shape: {images.shape}, Batch labels: {labels}")

actual_labels = []
predicted_labels = []
count=0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
    
        actual_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

for i in range(len(predicted_labels)):
    if(actual_labels[i]!=predicted_labels[i]):
        count+=1
print(count)


from sklearn.metrics import confusion_matrix, classification_report

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["genuine", "fake"]))

import torch
import torchvision.models as models

# Load the trained model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modify according to your classes
torch.save(model.state_dict(), "modelaug.pth")
model.load_state_dict(torch.load("modelaug.pth", map_location=torch.device('cpu')))
model.eval()

# Check if the model loads correctly
print("✅ Model loaded successfully!")

dummy_input = torch.randn(1, 3, 224, 224)  # Batch size = 1, 3 color channels, 224x224 image

torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",  # Output ONNX filename
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Enable dynamic batch size
    opset_version=11  # Ensure compatibility with ONNX Runtime
)

print("✅ Model successfully converted to ONNX format!")

import onnx

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid!")

pip install onnxruntime

import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Generate a dummy input for testing
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_name = session.get_inputs()[0].name

# Run inference
output = session.run(None, {input_name: input_data})
print("🔍 Model Output:", output)

import onnx

model = onnx.load("model.onnx")
for inp in model.graph.input:
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    print(f"Input: {inp.name}, Shape: {shape}")


import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_path = "/kaggle/working/model.onnx"
quantized_model_path = "/kaggle/working/model_quantized.onnx"

# Reduce size using dynamic quantization
quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QInt8)

print("✅ Quantization complete! New model saved at:", quantized_model_path)






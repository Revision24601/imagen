import os
import torch
import tensorflow as tf
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load pre-trained PyTorch model (ResNet50)
pytorch_model = models.resnet50(pretrained=True)
pytorch_model.eval()

# Load pre-trained TensorFlow model (MobileNetV2)
tf_model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Define PyTorch preprocessing
def preprocess_pytorch(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Define TensorFlow preprocessing
def preprocess_tf(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)
    return image

# Load ImageNet class labels manually
imagenet_classes_path = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
import requests

response = requests.get(imagenet_classes_path)
imagenet_classes = response.text.splitlines()
imagenet_classes = {idx: label for idx, label in enumerate(imagenet_classes)}

def predict_pytorch(image):
    input_tensor = preprocess_pytorch(image)
    with torch.no_grad():
        outputs = pytorch_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)  # Get top 5 predictions

    results = [
        {"label": imagenet_classes[catid.item()], "confidence": prob.item()}
        for prob, catid in zip(top5_prob, top5_catid)
    ]
    return results

def predict_tf(image):
    input_tensor = preprocess_tf(image)
    predictions = tf_model.predict(input_tensor)
    class_labels = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)  # Get top 5

    results = [
        {"label": label, "confidence": float(confidence)}
        for (_, label, confidence) in class_labels[0]
    ]
    return results

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(file.stream).convert('RGB')
    model_type = request.form.get('model', 'pytorch')

    if model_type == 'pytorch':
        results = predict_pytorch(image)
    else:
        results = predict_tf(image)

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
import torch
from data.preprocess import extract_features
from model.cnn import AudioClassifier
import utils

# Load trained model
model = AudioClassifier(num_classes=10)
utils.load_model(model, "audio_classifier.pth")
model.eval()

# Prediction function
def predict(model, file_path):
    features = extract_features(file_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(features)
        prediction = torch.argmax(output, 1).item()
    
    return prediction

# Test inference
file_path = "path/to/audio/file.wav"
predicted_class = predict(model, file_path)
print(f"Predicted Class: {predicted_class}")

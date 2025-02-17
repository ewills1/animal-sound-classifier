import torch
from torch.utils.data import DataLoader
from data.dataset import AudioDataset
from model.cnn import AudioClassifier
import utils

# Load test dataset
test_files = [...]  # List of test file paths
test_labels = [...]  # Corresponding test labels
test_dataset = AudioDataset(test_files, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = AudioClassifier(num_classes=10)
utils.load_model(model, "audio_classifier.pth")
model.eval()

# Evaluation
def evaluate(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

evaluate(model, test_loader)

import torch
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from model.cnn import CNN
import matplotlib.pyplot as plt
import seaborn as sns
from train import AudioDataset

# Load the trained model
model = CNN()  
model.load_state_dict(torch.load('audio_classifier.pth'))
model.eval()  

# Load your test dataset
test_dataset = AudioDataset()  # Replace with your dataset class
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

all_predictions = []
all_labels = []

with torch.no_grad():  # No need to calculate gradients during evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Forward pass
        _, preds = torch.max(outputs, 1)  # Get predicted classes
        all_predictions.extend(preds.numpy())  # Store predictions
        all_labels.extend(labels.numpy())  # Store true labels

accuracy = np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2", "Class 3"]))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2", "Class 3"], yticklabels=["Class 0", "Class 1", "Class 2", "Class 3"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

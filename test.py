import torch
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from data.parse import Parse
from model.cnn import CNN
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from train import AudioDataset

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parse = Parse()
test_df = parse.get_test_dataframe()

test_labels = test_df['label'].values

# Convert string representations of lists to actual lists
test_features = test_df['features'].apply(eval).tolist()

scaler = MinMaxScaler()

# Flatten the mel spectrograms for normalization
test_features_flat = [np.array(feature).flatten() for feature in test_features]
test_features_flat = np.array([resample(x, 128) for x in test_features_flat])

# Reshape to (batch_size, 1, 128, 1) without an extra dimension
test_features_normalized = scaler.fit_transform(test_features_flat)
test_features_normalized = np.array(test_features_normalized).reshape(len(test_features_normalized), 1, 128, 1)

# Load your test dataset
test_dataset = AudioDataset(test_features_normalized, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = CNN(num_classes=5)  
model.load_state_dict(torch.load('audio_classifier.pth'))
model.to(device)
model.eval()  

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_predictions.extend(preds.cpu().numpy())  # Convert to numpy
        all_labels.extend(labels.cpu().numpy())

accuracy = np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], yticklabels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

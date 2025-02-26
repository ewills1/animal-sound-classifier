import torch
import torch.optim as optim
import torch.nn as nn
from data.parse import Parse
from torch.utils.data import DataLoader, Dataset
from model.cnn import CNN
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import utils

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are LONG for CrossEntropyLoss

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

parse = Parse()
train_df = parse.get_train_dataframe()
test_df = parse.get_test_dataframe()

train_labels = train_df['label'].values
test_labels = test_df['label'].values

train_features = train_df['features'].tolist()
test_features = test_df['features'].tolist()

scaler = MinMaxScaler()

# Flatten the mel spectrograms for normalization
train_features_flat = [feature.flatten() for feature in train_features]
train_features_flat = np.array([resample(x, 128) for x in train_features_flat])

test_features_flat = [feature.flatten() for feature in test_features]
test_features_flat = np.array([resample(x, 128) for x in test_features_flat])

# Reshape to (batch_size, 1, 128, 1) without an extra dimension
train_features_normalized = scaler.fit_transform(train_features_flat)
train_features_normalized = np.array(train_features_normalized).reshape(len(train_features_normalized), 1, 128, 1)

test_features_normalized = scaler.fit_transform(test_features_flat)
test_features_normalized = np.array(test_features_normalized).reshape(len(test_features_normalized), 1, 128, 1)

print(train_features_normalized.shape)  # Should be (batch_size, 1, 128, 1)
train_labels = train_labels - train_labels.min()
test_labels = test_labels - test_labels.min()

print("Max label:", train_labels.max())
print("Min label:", train_labels.min())
print("Number of classes:", len(np.unique(train_labels)))


# Create an instance of the custom dataset
audio_dataset = AudioDataset(train_features_normalized, train_labels)

# Create a DataLoader
train_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(AudioDataset(test_features_normalized, test_labels), batch_size=32, shuffle=False)

# Model, loss, optimizer
train_model = CNN(num_classes=len(np.unique(train_labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(train_model.parameters(), lr=0.0001)

test_model = CNN(num_classes=len(np.unique(test_labels)))
optimizer = optim.Adam(test_model.parameters(), lr=0.0001)

# Training function
def train(model, dataloader, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


# Train model
train(train_model, train_loader, criterion, optimizer, epochs=30)

# Evaluate model
evaluate(train_model, test_loader)

# Save trained model
utils.save_model(train_model, "audio_classifier.pth")



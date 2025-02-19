import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model.cnn import CNN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import utils
from ast import literal_eval

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def process_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Convert string features to numpy arrays
    # Convert the features lists back to numpy arrays
    train_features = np.array(train_df['features'])
    test_features = np.array(test_df['features'])
    
    # Check the shapes after extraction
    print("Extracted train features shape:", train_features.shape)  # Should be (num_samples, feature_length)
    print("Extracted test features shape:", test_features.shape)

    train_labels = train_df['label']
    test_labels = test_df['label'] 

    # Normalize features
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    train_features = train_features.reshape(train_features.shape[0], 1, 128, 1)  # This keeps the 1 channel and adds a width dimension.
    test_features = test_features.reshape(test_features.shape[0], 1, 128, 1)  # This keeps the 1 channel and adds a width dimension.

    
    return (train_features, train_labels), (test_features, test_labels)

# Load data
(train_features, train_labels), (test_features, test_labels) = process_data('train_features.csv', 'test_features.csv')


# Create datasets and dataloaders
train_dataset = AudioDataset(train_features, train_labels)
test_dataset = AudioDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = CNN(num_classes=len(np.unique(train_labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

# Train model
train(model, train_loader, criterion, optimizer, epochs=10)

# Save trained model
utils.save_model(model, "audio_classifier.pth")



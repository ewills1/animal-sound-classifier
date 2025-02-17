import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import AudioDataset
from model.cnn import AudioClassifier
import numpy as np
import os
import librosa
import utils

def load_data_with_labels(directory):
    data = []
    labels = []
    for class_name in os.listdir(directory):  # Class names (dog, cat, cow)
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory
            for file_name in os.listdir(class_path):
                if file_name.endswith('.ogg'):  # Only process .ogg files
                    file_path = os.path.join(class_path, file_name)
                    audio, _ = librosa.load(file_path, sr=None)  # Load audio
                    data.append(audio)
                    labels.append(class_name)  # Assign label from folder name
    return np.array(data), np.array(labels)

train_dataset = AudioDataset(data, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Model, loss, optimizer
# model = AudioClassifier(num_classes=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training function
# def train(model, dataloader, criterion, optimizer, epochs=10):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in dataloader:
#             inputs = inputs.unsqueeze(1)  # Add channel dimension
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
        
#         print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

# # Train model
# train(model, train_loader, criterion, optimizer, epochs=10)

# # Save trained model
# utils.save_model(model, "audio_classifier.pth")



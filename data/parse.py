import os
import shutil
from sklearn.model_selection import train_test_split
import kagglehub
import pandas as pd
import librosa
import numpy as np

# Load dataset
path = kagglehub.dataset_download("ouaraskhelilrafik/tp-02-audio")

print("Path to dataset files:", path)

# Define the paths
dataset_dir = os.path.join(path, "data/data")  # Adjust to reach actual class folders  # Update with the path to your dataset

data = [] # List to store (filepath, label)
features = [] # List to store features

# Create train and test directories for each class
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):  # Only process directories
        class_id = int(class_name.split(" - ")[0])  # Extract numeric ID

        for root, _, filenames in os.walk(class_path):
            for filename in filenames:
                if filename.endswith('.ogg'):
                    file_path = os.path.join(root, filename)
                    data.append((file_path, class_name))  # Store filename and label
                    
                    # Load audio file
                    y, sr = librosa.load(file_path, sr=None)

                    # Extract Mel Spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    
                    # Append features and label
                    features.append((mel_spectrogram_db.tolist(), class_id))



df = pd.DataFrame(data, columns=["filepath", "label"])
df.to_csv("file_labels.csv", index=False)

df_mels = pd.DataFrame(features, columns=["features", "label"])

# Split into train and test sets
train_df, test_df = train_test_split(df_mels, test_size=0.2, random_state=42, stratify=df_mels["label"])

# Save to CSV
train_df.to_csv("train_features.csv", index=False)
test_df.to_csv("test_features.csv", index=False)

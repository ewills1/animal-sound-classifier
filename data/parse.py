import os
import shutil
from sklearn.model_selection import train_test_split
import kagglehub
import pandas as pd
import librosa
import numpy as np

class Parse:
    def __init__(self):
        # Load dataset
        self.path = kagglehub.dataset_download("ouaraskhelilrafik/tp-02-audio")
        print("Path to dataset files:", self.path)

        # Define the paths
        self.dataset_dir = os.path.join(self.path, "data/data")  # Adjust to reach actual class folders

        self.data = []  # List to store (filepath, label)
        self.features = []  # List to store features

        # Process dataset and extract features
        self.load_data()

    def load_data(self):
        # Create train and test directories for each class
        for class_name in os.listdir(self.dataset_dir):
            class_path = os.path.join(self.dataset_dir, class_name)
            if os.path.isdir(class_path):  # Only process directories
                class_id = int(class_name.split(" - ")[0])  # Extract numeric ID

                for root, _, filenames in os.walk(class_path):
                    for filename in filenames:
                        if filename.endswith('.ogg'):
                            file_path = os.path.join(root, filename)
                            self.data.append((file_path, class_name))  # Store filename and label

                            # Load audio file
                            y, sr = librosa.load(file_path, sr=None)

                            # Extract Mel Spectrogram
                            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                            # Append features and label
                            self.features.append((mel_spectrogram_db, class_id))

        self.df = pd.DataFrame(self.data, columns=["filepath", "label"])
        self.df.to_csv("file_labels.csv", index=False)

        self.df_mels = pd.DataFrame(self.features, columns=["features", "label"])

        # Split into train and test sets
        self.train_df, self.test_df = train_test_split(self.df_mels, test_size=0.2, random_state=42, stratify=self.df_mels["label"])

        # Save to CSV
        self.train_df.to_csv("train_features.csv", index=False)
        self.test_df.to_csv("test_features.csv", index=False)

    def get_dataframe(self):
        return self.df

    def get_train_dataframe(self):
        return self.train_df

    def get_test_dataframe(self):
        return self.test_df

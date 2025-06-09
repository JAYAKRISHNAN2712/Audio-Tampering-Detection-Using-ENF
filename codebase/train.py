import os
import sys
import time
import json
import math
import wave
import argparse

import numpy as np
import matplotlib.pyplot as plt
import librosa
from PIL import Image
from scipy import signal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    preprocess_audio,
    extract_frame_wise_features,
    extract_spatial_features,
    extract_temporal_features,
    enf_processing,
)
from models import CNN_BiLSTM_Attention_MLP


def train(folder_path: str, batch_size: int, num_epochs: int, model_store_path: str) -> None:
    """
    Trains a CNN-BiLSTM-Attention-MLP model for audio tampering detection based on ENF signal features.

    This function processes a directory of audio files, extracts spatial and temporal ENF features,
    and trains a deep learning model to classify the audio as tampered or untampered.

    Parameters:
    ----------
    folder_path : str
        Path to the directory containing labeled .wav or .mp3 audio files.
        Files with names ending in 'e' are labeled as tampered; others as untampered.

    batch_size : int
        Number of samples per batch used during training.

    num_epochs : int
        Number of complete training epochs over the dataset.

    model_store_path : str
        Directory where the trained model will be saved as 'model.pth'.

    Returns:
    -------
    None
        The function saves the trained model to disk and prints training progress but does not return any values.
    """
    if not os.path.isdir(folder_path):
        print("❌ Provided folder path does not exist.")
        return

    print("============= EXTRACTING ENF COMPONENTS ===============")
    labels = []
    enf_list = []

    for audio_file in os.listdir(folder_path):
        if os.path.splitext(audio_file)[1].lower() in ['.wav', '.mp3']:
            audio_file_path = os.path.join(folder_path, audio_file)
            if os.path.splitext(audio_file)[0].endswith('e'):
                labels.append([1.0, 0.0])  # Tampered
            else:
                labels.append([0.0, 1.0])  # Untampered

            signal0, fs = librosa.load(audio_file_path, sr=None)
            enf_list.append(preprocess_audio(signal0, fs))

    spatial_features, temporal_features = enf_processing(enf_list)

    # Prepare DataLoader
    spatial_features = torch.tensor(np.array(spatial_features), dtype=torch.float32)
    padded_temporal_features = torch.tensor(np.array(temporal_features), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(spatial_features, padded_temporal_features, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('model_store/params.json', 'r') as f:
        params = json.load(f)

    model = CNN_BiLSTM_Attention_MLP(**params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0

        for spatial_batch, temporal_batch, label_batch in train_loader:
            spatial_batch = spatial_batch.to(device)
            temporal_batch = temporal_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            outputs = model(spatial_batch, temporal_batch)

            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch Loss: {total_loss:.4f}")

    # Save model
    os.makedirs(model_store_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_store_path, 'model.pth'))
    print(f"\n✅ Model saved to {model_store_path}/model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN-BiLSTM-Attention model on ENF features.")
    parser.add_argument('-i', '--input_directory', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('-e', '--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-chpkt', '--model_store_path', type=str, default='model_store/', help='Path to save the model')

    args = parser.parse_args()
    train(args.input_directory, args.batch_size, args.num_epochs, args.model_store_path)

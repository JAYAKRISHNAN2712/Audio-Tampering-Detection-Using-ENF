import os
import json
import argparse

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scipy import signal
from PIL import Image

from utils import (
    preprocess_audio,
    extract_frame_wise_features,
    extract_spatial_features,
    extract_temporal_features,
    enf_processing,
)
from models import CNN_BiLSTM_Attention_MLP


def inference(audio_file_path: str, model_store: str = 'model_store') -> None:
    """
    Performs inference on a single audio file to detect tampering using a trained CNN-BiLSTM-Attention-MLP model.

    This function loads a trained model, extracts ENF-based spatial and temporal features from the input audio file,
    and predicts whether the audio is tampered or untampered.

    Parameters:
    ----------
    audio_file_path : str
        Full path to the input audio file (.wav or .mp3) to be analyzed.

    model_store : str, optional
        Directory where the trained model ('model.pth') and the model parameters ('params.json') are stored.
        Defaults to 'model_store'.

    Returns:
    -------
    None
        The function prints the raw logits, softmax probabilities, and the final prediction label ("tampered" or "untampered").
    """
    if not os.path.isfile(audio_file_path):
        print(f"❌ File not found: {audio_file_path}")
        return

    # Label mapping
    idx_to_label = {0: "tampered", 1: "untampered"}

    # Load audio and extract ENF features
    signal0, fs = librosa.load(audio_file_path, sr=None)
    enf_features = preprocess_audio(signal0, fs)
    spatial_features, temporal_features = enf_processing([enf_features])

    # Prepare tensors
    spatial_features = torch.tensor(np.array(spatial_features), dtype=torch.float32)
    temporal_features = torch.tensor(np.array(temporal_features), dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model parameters
    with open(os.path.join(model_store, 'params.json'), 'r') as f:
        params = json.load(f)

    model = CNN_BiLSTM_Attention_MLP(**params)
    model.load_state_dict(torch.load(os.path.join(model_store, 'model.pth'), map_location=device))
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        spatial_features = spatial_features.to(device)
        temporal_features = temporal_features.to(device)

        logits = model(spatial_features, temporal_features)
        probs = torch.sigmoid(logits)
        predicted_index = torch.argmax(probs, dim=1).item()
        predicted_label = idx_to_label[predicted_index]

    # Output results
    print("Raw logits:", logits.cpu().numpy())
    print("Probabilities:", probs.cpu().numpy())
    print("Prediction:", predicted_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a single audio file.")
    parser.add_argument('-i', '--audio_path', type=str, required=True, help="Path to input audio file")
    parser.add_argument('-m', '--model_store', type=str, default='model_store', help="Path to trained model directory")

    args = parser.parse_args()
    inference(args.audio_path, args.model_store)

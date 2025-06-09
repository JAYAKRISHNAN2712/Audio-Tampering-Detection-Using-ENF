import json
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

class DeepCNN(nn.Module):
    """
    A simple deep convolutional neural network block with one Conv2D layer,
    followed by ReLU activation and MaxPooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels:int, out_channels:int):
        super(DeepCNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3)
        self.relu = nn.ReLU()
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network block.

        :param:
            batch (torch.Tensor): Input tensor of shape (batch_size,
            number of input channels, number of frames per unit frame, number of frames per unit frame).

        :return:
            torch.Tensor: Output tensor after convolution, ReLU, and max pooling.
        """
        feature_map = self.conv_layer_1(batch)
        relued_feature_map = self.relu(feature_map)
        max_pooled_feature_map = self.max_pool_1(relued_feature_map)
        return max_pooled_feature_map

class DeepBiLSTM(nn.Module):
    """
    A deep bidirectional LSTM model with layer normalization and ReLU activation.

    Args:
        input_size (int): Number of frames (f_n).
        hidden_size (int): number of phase points per unit frame (p_n).
        num_layers (int): Number of recurrent layers.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(DeepBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.layerNorm = nn.LayerNorm(2 * hidden_size)
        self.relu = nn.ReLU()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepBiLSTM model.

        :param:
            batch (torch.Tensor): Input Tensor of shape (batch_size, p_n, f_n).

        :returns:
            torch.Tensor: Output tensor after LSTM, layer normalization, and ReLU.
        """
        hidden_state_output, _ = self.lstm(batch)
        layer_normalized_output = self.layerNorm(hidden_state_output)
        return self.relu(layer_normalized_output)


class FC(nn.Module):
    """
    A simple fully connected (linear) layer wrapper.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """
    def __init__(self, in_features: int, out_features: int):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, batch: torch.Tensor):
        """
        Forward pass through the fully connected layer.

        Args:
            batch (torch.Tensor): Input tensor of shape (batch_size, *, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *, out_features).
        """
        linear_feature_map = self.fc(batch)
        return linear_feature_map

class CNN_BiLSTM_Attention_MLP(nn.Module):
    """
       Combined CNN + BiLSTM + Attention + MLP architecture for spatiotemporal learning.
    """

    def __init__(
            self,
            cnn_in_channels: List[int],
            cnn_out_channels: List[int],
            cnn_kernel_size: List[int],
            cnn_output_dim: int,
            lstm_input_size_layer_1: int,
            lstm_hidden_size_layer_1: int,
            lstm_num_layers_layer_1: int,
            lstm_input_size_layer_2: int,
            lstm_hidden_size_layer_2: int,
            lstm_num_layers_layer_2: int,
            lstm_output_dim: int,
            attn_compression_param: int
    ) -> None:

        super(CNN_BiLSTM_Attention_MLP, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CNN block
        self.cnn_block = nn.Sequential(
            DeepCNN(cnn_in_channels[0], cnn_out_channels[0]).to(device),
            DeepCNN(cnn_in_channels[1], cnn_out_channels[1]).to(device),
            DeepCNN(cnn_in_channels[2], cnn_out_channels[2]).to(device)
        )

        # FC block
        self.fc_block_1 = nn.Sequential(
            FC(cnn_output_dim, 1024).to(device),
            FC(1024, 256).to(device)
        )

        # BiLSTM block
        self.bilstm_block = nn.Sequential(
            DeepBiLSTM(lstm_input_size_layer_1, lstm_hidden_size_layer_1, lstm_num_layers_layer_1).to(device),
            DeepBiLSTM(lstm_input_size_layer_2, lstm_hidden_size_layer_2, lstm_num_layers_layer_2).to(device)
        )

        # FC Block
        self.fc_block_2 = nn.Sequential(
            FC(lstm_output_dim, 512).to(device),
            FC(512, 256).to(device)
        )

        # Attention Module
        self.attn_model = nn.Sequential(
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(512, 512 // 8),
            nn.ReLU(),
            nn.Linear(512 // 8, 512),
            nn.Sigmoid()
        )

        # DNN Classifier
        self.mlp_model = nn.Sequential(
            nn.Linear(512, 400),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(400, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 2),  # For softmax binary classification
            nn.Softmax(dim=1)  # Use dim=1 for batch-based outputs
        )

    def forward(self, spatial_feature_batch: torch.Tensor, temporal_feature_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            spatial_feature_batch (torch.Tensor): Input spatial data, expected shape (B, 1, 45, 45).
            temporal_feature_batch (torch.Tensor): Input temporal data, expected shape(B, 1, 85, 25).

        Returns:
            torch.Tensor: Output probabilities for binary classification.
        """
        # Process spatial features through CNN
        spatial_feature_batch = spatial_feature_batch.view(-1, 1, 45,45).float()
        cnn_out = self.cnn_block(spatial_feature_batch)
        cnn_out_flattened = cnn_out.reshape(cnn_out.size(0), -1)
        cnn_fc_out = self.fc_block_1(cnn_out_flattened)

        # Process temporal features though Bi-LSTM
        temporal_feature_batch = np.stack(temporal_feature_batch)
        temporal_feature_batch = torch.from_numpy(temporal_feature_batch).float()
        bilstm_out = self.bilstm_block(temporal_feature_batch)
        bilstm_out_flattened = bilstm_out[:, -1, :]
        bilstm_fc_out = self.fc_block_2(bilstm_out_flattened)
        bilstm_fc_out =  bilstm_fc_out.view(bilstm_fc_out.size(0), 256)

        # Combine CNN and BiLSTM features
        spatio_temporal_features = torch.cat([cnn_fc_out, bilstm_fc_out], dim=1)

        # Attention mechanism
        attn_weights = self.attn_model(spatio_temporal_features)
        weighted_spatio_temporal_features = spatio_temporal_features * attn_weights

        # MLP Prediction
        predict_proba = self.mlp_model(weighted_spatio_temporal_features)
        return predict_proba
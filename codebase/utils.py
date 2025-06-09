import json
import math
from typing import List, Tuple, Dict

import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt, get_window, resample

def preprocess_audio(audio_signal, orig_fs, target_fs=1000, enf_freq=50):
    """
    Preprocess the audio signal to extract the ENFC signal.

    :param audio_signal: 1D NumPy array of raw audio data
    :param orig_fs: Original sampling frequency of the audio
    :param target_fs: Target down-sampling frequency (default 1000 Hz)
    :param enf_freq: Standard ENF frequency (50 Hz or 60 Hz)
    :return: Extracted ENFC signal
    """
    # Step 1: Down-sample the signal
    num_samples = int(len(audio_signal) * target_fs / orig_fs)
    downsampled_signal = resample(audio_signal, num_samples)

    # Step 2: Apply Band-pass Filtering
    low_cut = enf_freq - 1  # Lower bound (e.g., 49 Hz for 50 Hz ENF)
    high_cut = enf_freq + 1  # Upper bound (e.g., 51 Hz for 50 Hz ENF)
    nyquist = 0.5 * target_fs  # Nyquist frequency
    low = low_cut / nyquist
    high = high_cut / nyquist

    b, a = butter(N=4, Wn=[low, high], btype='band')  # 4th order band-pass filter
    enfc_signal = filtfilt(b, a, downsampled_signal)  # Zero-phase filtering

    return enfc_signal


def extract_frame_wise_features(ENF_signal, fd, enf_freq=50, N_DFT=1024):
    """
    Extracts frame-wise ψ₀ and ψ₁ values using DFT1.

    :param ENF_signal: 1D array of ENF values
    :param fd: Down-sampling frequency
    :param enf_freq: Standard ENF frequency (50 Hz or 60 Hz)
    :param N_DFT: Number of DFT points
    :return: Lists of ψ₀, ψ₁, and f_DFT1 for each frame
    """
    ENF_signal = np.asarray(ENF_signal).flatten()  # Flatten the signal if needed

    # Frame and overlap calculation
    frame_length = int(10 * fd / enf_freq)  # 10 ENF periods
    frame_shift = int(fd / enf_freq)  # 1 ENF period
    num_frames = (len(ENF_signal) - frame_length) // (frame_shift + 1)

    # Initialize lists for frame-wise values
    psi_0_list = []
    psi_1_list = []
    f_DFT1_list = []

    for i in range(num_frames):
        start = i * frame_shift
        frame = ENF_signal[start: start + frame_length]

        # Compute first-order derivative
        u_ENFC = np.diff(frame) * fd  # First derivative

        # Apply Hanning window
        window = get_window('hann', len(frame))
        U_N = frame * window  # [:-1]  # Apply window (adjusting length)
        U_N_prime = u_ENFC * window[:-1]

        # Compute DFT
        U = fft(U_N, N_DFT)
        U_prime = fft(U_N_prime, N_DFT)

        # Find peak index
        k_peak = np.argmax(np.abs(U))
        U_peak = U[k_peak]
        U_prime_peak = U_prime[k_peak]

        # Compute ψ₀
        psi_0 = np.angle(U_peak)

        # Estimate f_DFT1
        epsilon = 1e-9  # Small constant to avoid division by zero
        f_DFT1 = (1 / (2 * np.pi)) * (np.abs(U_prime_peak) / (np.abs(U_peak) + epsilon))

        # Compute ψ₁
        omega_0 = 2 * np.pi * f_DFT1 / fd
        k_DFT_1 = f_DFT1 * N_DFT / fd
        k_high = math.ceil(k_DFT_1)
        k_low = math.floor(k_DFT_1)

        # Ensure index bounds
        k_high = min(k_high, len(U) - 1)
        k_low = max(k_low, 0)

        theta_low = np.angle(U[k_low])
        theta_high = np.angle(U[k_high])

        # Linear Interpolation for θ
        if k_high != k_low:
            theta = (k_DFT_1 - k_low) * ((theta_high - theta_low) / (k_high - k_low)) + theta_low
        else:
            theta = theta_low

        # Compute ψ₁
        psi_1 = np.arctan(
            (np.tan(theta) * (1 - np.cos(omega_0) + np.sin(omega_0))) /
            (1 - np.cos(omega_0) - np.tan(theta) * np.sin(omega_0))
        )

        # Choose ψ₁ closest to ψ₀
        psi_1 = psi_1 if abs(psi_1 - psi_0) < abs(psi_1 + psi_0) else -psi_1

        # Store frame-wise values
        psi_0_list.append(psi_0)
        psi_1_list.append(psi_1)
        f_DFT1_list.append(f_DFT1)

    return psi_0_list, psi_1_list, f_DFT1_list

def extract_spatial_features(psi_1_list, n:int):
    """
    Extracts spatial features from phase sequence features ψ1.

    Parameters:
    psi_1_list (list of arrays): List of phase sequence features ψ1 from different audio files.

    Returns:
    list of np.ndarray: List of spatial feature matrices P(n x n) for each ψ1.
    """
    # Step 1: Calculate the longest phase sequence length
    len_psi_DFT1 = max(len(psi_1) for psi_1 in psi_1_list)

    # Step 2: Determine the number of phase points per unit frame (n)
    X = math.ceil(math.sqrt(len_psi_DFT1))
    #n = math.ceil(math.sqrt(X))
    #n=45
    spatial_features = []  # Store feature matrices for all phase sequences

    for psi_1 in psi_1_list:
        # Step 3: Calculate the frame shift (overlap)
        overlap = n - math.ceil((n - 1) / X - n)

        # Split the phase sequence into overlapping frames
        frames = []
        for i in range(0, len(psi_1) - n + 1, overlap):
            frames.append(psi_1[i:i + n])

        # Pad frames if necessary to make them of size (n x n)
        while len(frames) < n:
            frames.append(np.zeros(n))  # Pad with zeros if needed

        # Convert frames into an n x n matrix
        P_n_n = np.array(frames[:n])  # Take first 'n' frames

        spatial_features.append(P_n_n)

    return spatial_features



def extract_temporal_features(psi_1_list: list, p_n: int, f_n: int):
    """
    Extracts temporal ENF features from psi_1 sequences,
    reshaping each into a (p_n, f_n) matrix (85x25).

    :param psi_1_list: List of 1D NumPy arrays (psi_1 sequences).
    :param p_n: Number of phase points per frame.
    :param f_n: Number of frames.
    :return: List of 2D NumPy arrays of shape (p_n, f_n).
    """
    total_length = p_n * f_n
    temporal_features = []

    for psi_1 in psi_1_list:
        seq_len = len(psi_1)

        # Pad or trim to match required total length
        if seq_len < total_length:
            pad_len = total_length - seq_len
            psi_1_padded = np.pad(psi_1, (0, pad_len), mode='constant')
        else:
            psi_1_padded = psi_1[:total_length]

        # Reshape into (p_n, f_n)
        reshaped = psi_1_padded.reshape((f_n, p_n)).T  # Transpose to (p_n, f_n)
        temporal_features.append(reshaped)

    return temporal_features


def enf_processing(enf_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Processes a list of ENF components and extracts spatial and temporal features.

    Args:
        enf_list (List[np.ndarray]): List of 1D ENF signals.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - spatial_features: List of spatial feature matrices P (n x n) for each ψ₁.
            - temporal_features: List of temporal feature matrices X (p_n x f_n) for each ψ₁.
    """
    ENF = np.asarray(enf_list, dtype=object)

    psi_0_list = []
    psi_1_list = []

    for enf_signal in ENF:
        psi_0, psi_1, _ = extract_frame_wise_features(
            enf_signal, fd=1000, enf_freq=50, N_DFT=1024
        )
        psi_0_list.append(np.array(psi_0))
        psi_1_list.append(np.array(psi_1))

    spatial_features = extract_spatial_features(psi_1_list, n=45)
    temporal_features = extract_temporal_features(psi_1_list, p_n=85, f_n=25)

    return spatial_features, temporal_features
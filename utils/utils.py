

import os
import random
import subprocess

import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def frozen_model(model):
    if model is None:
        return None
    for param in model.parameters():
        param.requires_grad = False
    model = model
    return model

def set_seed(seed=10086):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # Disable hash randomization for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If using multiple GPUs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_signal(X,max_cutoff_frequency=None,min_cutoff_frequency=0,std_freq=100):
    if max_cutoff_frequency is None:
        max_cutoff_frequency = std_freq

    fft_signal = np.fft.fft(X.squeeze().numpy(),axis=1)
    n = (X.shape[1])  # Signal length
    frequencies = np.fft.fftfreq(n, 1/std_freq)  # Frequency components
    frequencies = np.fft.fftshift(frequencies)  # Shift zero frequency to center

    fft_signal_filtered = np.fft.fftshift(fft_signal)  # Center the spectrum
    fft_signal_filtered[:,( (np.abs(frequencies) < max_cutoff_frequency) &(min_cutoff_frequency <= np.abs(frequencies)))] = 0  # Apply band-stop (remove selected band)
    fft_signal_filtered = np.fft.ifftshift(fft_signal_filtered)  # Restore spectrum

    # Inverse FFT back to time domain
    filtered_signal = np.fft.ifft(fft_signal_filtered).real  # Take real part
    return filtered_signal


def get_folder_size(folder_path):
    """Get folder size using system command"""
    try:
        result = subprocess.run(
            ['du', '-sb', folder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # Parse command output and extract size
        size_in_bytes = int(result.stdout.split()[0])
        return size_in_bytes
    except Exception as e:
        print(f"Error retrieving folder size: {e}")
        return None


# def str2list(s:str):
#     return list(map(str,s[1:-1].split(', ')))

if __name__ == '__main__':
    print('end')

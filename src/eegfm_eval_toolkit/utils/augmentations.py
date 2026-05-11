import numpy as np
import torch
from scipy.signal import butter, lfilter

def add_gaussian_noise(x, snr_db):
    """
    Adds Gaussian noise to the signal to achieve a target SNR (dB).
    x: (Channels, Time)
    """
    # Calculate signal power per channel
    # axis=1 means time dimension
    x_power = np.mean(x ** 2, axis=1, keepdims=True)
    
    # Calculate required noise power
    # SNR_db = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / 10^(SNR/10)
    noise_power = x_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.normal(0, 1, x.shape)
    
    # Scale noise to required power
    # We need to scale the standard normal (std=1) to have variance = noise_power
    # Multiply by sqrt(noise_power)
    noise_scaled = noise * np.sqrt(noise_power)
    
    return (x + noise_scaled).astype(np.float32)

def add_emg_noise(x, fs, target_ch_idx, snr_db=5, burst_prob=0.5):
    """
    Simulates EMG (Muscle) noise: High-frequency, bursty noise on specific channels.
    x: (Channels, Time)
    fs: Sampling frequency
    """
    if len(target_ch_idx) == 0:
        return x
        
    n_ch, n_time = x.shape
    x_noisy = x.copy()
    
    # EMG is typically high frequency (e.g., >20Hz). 
    # We simulate it with bandpassed white noise.
    nyq = 0.5 * fs
    b, a = butter(4, [20 / nyq], btype='highpass') # > 20Hz
    
    for ch in target_ch_idx:
        # Determine if this trial has an EMG burst (burst_prob)
        if np.random.rand() < burst_prob:
            # Generate White Noise
            noise = np.random.normal(0, 1, n_time)
            
            # Filter to make it look like EMG
            noise_emg = lfilter(b, a, noise)
            
            # Scale to target SNR relative to the specific channel's power
            sig_power = np.mean(x[ch] ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            
            # Apply a temporal envelope (burstiness)
            # Create a window (e.g., Hanning) located randomly
            burst_len = np.random.randint(int(0.1*fs), int(0.5*fs)) # 100ms to 500ms
            start = np.random.randint(0, n_time - burst_len)
            
            envelope = np.zeros(n_time)
            envelope[start:start+burst_len] = np.hanning(burst_len)
            
            noise_scaled = noise_emg * np.sqrt(noise_power) * envelope * 5.0 # *5 factor for burst intensity
            
            x_noisy[ch] += noise_scaled.astype(np.float32)
            
    return x_noisy

def add_eog_noise(x, fs, target_ch_idx, amplitude_factor=2.0):
    """
    Simulates EOG (Eye) noise: Low-frequency, high amplitude drifts/blinks on frontal channels.
    x: (Channels, Time)
    """
    if len(target_ch_idx) == 0:
        return x

    n_ch, n_time = x.shape
    x_noisy = x.copy()
    t = np.arange(n_time) / fs

    for ch in target_ch_idx:
        # Simulate Blink: Low freq sine wave or Gaussian pulse
        # Using a very slow sine wave (< 1Hz) to simulate drift/blink
        
        if np.random.rand() < 0.5: # 50% chance of EOG artifact
            freq = np.random.uniform(0.1, 3.0) # 0.1 to 3 Hz
            phase = np.random.uniform(0, 2*np.pi)
            
            # EOG is usually much larger than EEG
            # We estimate channel std and multiply
            scale = np.std(x[ch]) * amplitude_factor
            
            artifact = np.sin(2 * np.pi * freq * t + phase) * scale
            x_noisy[ch] += artifact.astype(np.float32)
            
    return x_noisy
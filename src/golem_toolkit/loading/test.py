#%%
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Create a test signal
fs = 1000
t = np.linspace(0, 2, 2*fs)
# Signal with changing frequency
freq = 50 + 30 * t  # Frequency increases from 50 to 110 Hz
signal_data = np.sin(2 * np.pi * freq * t)

# Compute STFT
frequencies_stft, times_stft, Zxx = signal.stft(signal_data, fs=fs, nperseg=256)

# Compute Spectrogram
frequencies_spec, times_spec, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=256)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# STFT Magnitude
axes[0,0].pcolormesh(times_stft, frequencies_stft, np.abs(Zxx), shading='gouraud')
axes[0,0].set_title('STFT Magnitude')
axes[0,0].set_ylabel('Frequency [Hz]')

# STFT Phase
axes[0,1].pcolormesh(times_stft, frequencies_stft, np.angle(Zxx), shading='gouraud')
axes[0,1].set_title('STFT Phase')
axes[0,1].set_ylabel('Frequency [Hz]')

# Spectrogram
axes[1,0].pcolormesh(times_spec, frequencies_spec, Sxx, shading='gouraud')
axes[1,0].set_title('Spectrogram (Power)')
axes[1,0].set_xlabel('Time [s]')
axes[1,0].set_ylabel('Frequency [Hz]')

# Spectrogram (log scale)
axes[1,1].pcolormesh(times_spec, frequencies_spec, 10*np.log10(Sxx), shading='gouraud')
axes[1,1].set_title('Spectrogram (dB)')
axes[1,1].set_xlabel('Time [s]')
axes[1,1].set_ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wfdb

# Download MIT-BIH data
print("Downloading ECG data...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
ecg_signal = record.p_signal[:, 0]  # First channel
fs = record.fs  # Sampling frequency (360 Hz)

# Normalize signal
ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

# Animation parameters
WINDOW_SIZE = 5  # seconds of data to display
UPDATE_INTERVAL = 50  # milliseconds between frames
PLAYBACK_SPEED = 1.0  # 1.0 = real-time, 2.0 = 2x speed

# Calculate samples
samples_per_window = int(WINDOW_SIZE * fs)
samples_per_frame = int((UPDATE_INTERVAL / 1000) * fs * PLAYBACK_SPEED)

# Setup figure with white background
fig = plt.figure(figsize=(12, 6), facecolor='white')

# Main ECG plot
ax_ecg = fig.add_subplot(111)
ax_ecg.set_facecolor('white')
ax_ecg.set_ylim(-2, 7)
ax_ecg.set_xlim(0, WINDOW_SIZE)
ax_ecg.set_ylabel('Amplitude (mV)', fontsize=10)
ax_ecg.set_title('Real-Time ECG Monitor', fontsize=14, pad=10)

# Add ECG grid
ax_ecg.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax_ecg.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.3)
ax_ecg.minorticks_on()
ax_ecg.tick_params(labelsize=8)

# Initialize plot line (red color)
line, = ax_ecg.plot([], [], color='red', linewidth=1.5)
sweep_line = ax_ecg.axvline(x=0, color='blue', linewidth=2, alpha=0.7)

# Data buffer
current_idx = 0
time_axis = np.linspace(0, WINDOW_SIZE, samples_per_window)

def init():
    """Initialize animation"""
    line.set_data([], [])
    return line, sweep_line

def animate(frame):
    """Update function for animation"""
    global current_idx
    
    # Get current window of data
    start_idx = current_idx
    end_idx = current_idx + samples_per_window
    
    # Loop back to start if we reach the end
    if end_idx >= len(ecg_signal):
        current_idx = 0
        start_idx = 0
        end_idx = samples_per_window
    
    # Extract current window
    current_window = ecg_signal[start_idx:end_idx]
    
    # Update plot
    line.set_data(time_axis, current_window)
    
    # Calculate sweep line position (simulates the moving cursor)
    sweep_position = ((frame * samples_per_frame) % samples_per_window) / fs
    sweep_line.set_xdata([sweep_position, sweep_position])
    
    # Advance index
    current_idx += samples_per_frame
    
    return line, sweep_line

# Create animation
print("Starting animation...")
anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=None, interval=UPDATE_INTERVAL,
                             blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

print("Animation complete!")

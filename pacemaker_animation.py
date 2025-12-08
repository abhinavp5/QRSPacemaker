import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wfdb

from Qrs_detect.r_peak import run_pantompkins
from SILPackemaker.Pacemaker import Pacemaker

# Download MIT-BIH data. # op shit right here

print("Downloading ECG data...")
record = wfdb.rdrecord('100', pn_dir='mitdb')




#############################################################################################################################


# i changed this to both channels
ecg_signal_ch1 = record.p_signal[:, 0]  # Channel 1 (Primary)
ecg_signal_ch2 = record.p_signal[:, 1]  # Channel 2 (Secondary)
fs = record.fs  # Sampling frequency (360 Hz)

ecg_signal_ch1 = (ecg_signal_ch1 - np.mean(ecg_signal_ch1)) / np.std(ecg_signal_ch1)
ecg_signal_ch2 = (ecg_signal_ch2 - np.mean(ecg_signal_ch2)) / np.std(ecg_signal_ch2)

# Run R-peak detection on both channels
print("Running R-peak detection...")
# Convert to int32 for Pan-Tompkins (as it expects integer input)
ecg_ch1_int = (ecg_signal_ch1 * 1000).astype(np.int32)  #need to check input type later
ecg_ch2_int = (ecg_signal_ch2 * 1000).astype(np.int32)

r_peaks_ch1 = run_pantompkins(ecg_ch1_int)
r_peaks_ch2 = run_pantompkins(ecg_ch2_int)

print(f"Channel 1: {np.sum(r_peaks_ch1)} R-peaks detected")
print(f"Channel 2: {np.sum(r_peaks_ch2)} R-peaks detected")


#############################################################################################################################



#########################
#VISUALIZATION CODE
#########################




# Find R-peak indices for visualization
r_peak_indices_ch1 = np.where(r_peaks_ch1 == 1)[0]
r_peak_indices_ch2 = np.where(r_peaks_ch2 == 1)[0]

# Initialize Pacemaker (VVIR mode - Ventricular paced, Ventricular sensed, Inhibited, Rate responsive)
print("Initializing Pacemaker...")
pacemaker = Pacemaker(NBG_code="VVIR", sampling_rate=int(fs))

# Simulate pacemaker operation on Channel 1 (primary)
print("Running Pacemaker simulation...")
pace_outputs = []
activity_level = 30  # Simulate low activity level (0-100 scale)

for i in range(len(ecg_signal_ch1)):
    # Use normalized signal for pacemaker (it expects voltage-like values)
    ecg_sample = ecg_signal_ch1[i]
    pace_decision = pacemaker.step(ecg_sample, activity_level)
    pace_outputs.append(pace_decision)

pace_outputs = np.array(pace_outputs)
pace_indices = np.where(pace_outputs == True)[0]  # Find where pacemaker would fire

print(f"Pacemaker would fire {len(pace_indices)} times")
print(f"Natural heartbeats detected: {np.sum(r_peaks_ch1)}")

# storing original
ecg_signal = ecg_signal_ch1

# Animation parameters
WINDOW_SIZE = 10  # seconds of data to display
UPDATE_INTERVAL = 50  # milliseconds between frames
PLAYBACK_SPEED = 0.5  # 1.0 = real-time, 2.0 = 2x speed

# Calculate samples
samples_per_window = int(WINDOW_SIZE * fs)
samples_per_frame = int((UPDATE_INTERVAL / 1000) * fs * PLAYBACK_SPEED)

# Setup figure with white background for dual-channel display
fig = plt.figure(figsize=(14, 10), facecolor='white')

# Channel 1 (Primary) - Top subplot
ax_ch1 = fig.add_subplot(311)
ax_ch1.set_facecolor('white')
ax_ch1.set_ylim(-3, 3)
ax_ch1.set_xlim(0, WINDOW_SIZE)
ax_ch1.set_ylabel('Channel 1 (mV)', fontsize=10)
ax_ch1.set_title('Dual-Channel ECG Monitor with Pacemaker', fontsize=14, pad=10)
ax_ch1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax_ch1.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.3)
ax_ch1.minorticks_on()
ax_ch1.tick_params(labelsize=8)

# Channel 2 (Secondary) - Middle subplot  
ax_ch2 = fig.add_subplot(312)
ax_ch2.set_facecolor('white')
ax_ch2.set_ylim(-3, 3)
ax_ch2.set_xlim(0, WINDOW_SIZE)
ax_ch2.set_ylabel('Channel 2 (mV)', fontsize=10)
ax_ch2.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax_ch2.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.3)
ax_ch2.minorticks_on()
ax_ch2.tick_params(labelsize=8)

# Pacemaker Activity - Bottom subplot
ax_pace = fig.add_subplot(313)
ax_pace.set_facecolor('white')
ax_pace.set_ylim(-0.5, 1.5)
ax_pace.set_xlim(0, WINDOW_SIZE)
ax_pace.set_ylabel('Pacemaker', fontsize=10)
ax_pace.set_xlabel('Time (seconds)', fontsize=10)
ax_pace.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax_pace.set_yticks([0, 1])
ax_pace.set_yticklabels(['OFF', 'PACE'])
ax_pace.tick_params(labelsize=8)

# Initialize plot lines
line_ch1, = ax_ch1.plot([], [], color='red', linewidth=1.5, label='Primary')
line_ch2, = ax_ch2.plot([], [], color='blue', linewidth=1.5, label='Secondary')

# Initialize R-peak markers (empty scatter plots)
r_peaks_plot_ch1 = ax_ch1.scatter([], [], color='red', marker='o', s=80, zorder=5, alpha=0.8, label='R-peaks')
r_peaks_plot_ch2 = ax_ch2.scatter([], [], color='blue', marker='o', s=80, zorder=5, alpha=0.8, label='R-peaks')

# Initialize Pacemaker markers
pace_plot_ch1 = ax_ch1.scatter([], [], color='orange', marker='^', s=100, zorder=6, alpha=0.9, label='Pacemaker')
pace_line, = ax_pace.plot([], [], color='orange', linewidth=2, label='Pacing Signal')

# Add legends
ax_ch1.legend(loc='upper right', fontsize=8)
ax_ch2.legend(loc='upper right', fontsize=8)
ax_pace.legend(loc='upper right', fontsize=8)

sweep_line_ch1 = ax_ch1.axvline(x=0, color='green', linewidth=2, alpha=0.7)
sweep_line_ch2 = ax_ch2.axvline(x=0, color='green', linewidth=2, alpha=0.7)
sweep_line_pace = ax_pace.axvline(x=0, color='green', linewidth=2, alpha=0.7)

# Data buffer
current_idx = 0
time_axis = np.linspace(0, WINDOW_SIZE, samples_per_window)

def init():
    """Initialize animation"""
    line_ch1.set_data([], [])
    line_ch2.set_data([], [])
    pace_line.set_data([], [])
    r_peaks_plot_ch1.set_offsets(np.empty((0, 2)))
    r_peaks_plot_ch2.set_offsets(np.empty((0, 2)))
    pace_plot_ch1.set_offsets(np.empty((0, 2)))
    return line_ch1, line_ch2, pace_line, r_peaks_plot_ch1, r_peaks_plot_ch2, pace_plot_ch1, sweep_line_ch1, sweep_line_ch2, sweep_line_pace

def animate(frame):
    """Update function for animation"""
    global current_idx
    
    # Get current window of data
    start_idx = current_idx
    end_idx = current_idx + samples_per_window
    
    # Loop back to start if we reach the end
    if end_idx >= len(ecg_signal_ch1):
        current_idx = 0
        start_idx = 0
        end_idx = samples_per_window
    
    # Extract current window for both channels
    current_window_ch1 = ecg_signal_ch1[start_idx:end_idx]
    current_window_ch2 = ecg_signal_ch2[start_idx:end_idx]
    current_window_pace = pace_outputs[start_idx:end_idx].astype(float)
    
    # Update both ECG plots
    line_ch1.set_data(time_axis, current_window_ch1)
    line_ch2.set_data(time_axis, current_window_ch2)
    
    # Update pacemaker signal
    pace_line.set_data(time_axis, current_window_pace)
    
    # Find R-peaks in current window
    window_r_peaks_ch1 = r_peak_indices_ch1[(r_peak_indices_ch1 >= start_idx) & (r_peak_indices_ch1 < end_idx)]
    window_r_peaks_ch2 = r_peak_indices_ch2[(r_peak_indices_ch2 >= start_idx) & (r_peak_indices_ch2 < end_idx)]
    
    # Find pacemaker events in current window
    window_pace_events = pace_indices[(pace_indices >= start_idx) & (pace_indices < end_idx)]
    
    # Convert sample indices to time for display
    if len(window_r_peaks_ch1) > 0:
        peak_times_ch1 = (window_r_peaks_ch1 - start_idx) / fs
        peak_amplitudes_ch1 = ecg_signal_ch1[window_r_peaks_ch1]
        peak_coords_ch1 = np.column_stack([peak_times_ch1, peak_amplitudes_ch1])
        r_peaks_plot_ch1.set_offsets(peak_coords_ch1)
    else:
        r_peaks_plot_ch1.set_offsets(np.empty((0, 2)))
    
    if len(window_r_peaks_ch2) > 0:
        peak_times_ch2 = (window_r_peaks_ch2 - start_idx) / fs
        peak_amplitudes_ch2 = ecg_signal_ch2[window_r_peaks_ch2]
        peak_coords_ch2 = np.column_stack([peak_times_ch2, peak_amplitudes_ch2])
        r_peaks_plot_ch2.set_offsets(peak_coords_ch2)
    else:
        r_peaks_plot_ch2.set_offsets(np.empty((0, 2)))
    
    # Show pacemaker events on Channel 1
    if len(window_pace_events) > 0:
        pace_times = (window_pace_events - start_idx) / fs
        pace_amplitudes = ecg_signal_ch1[window_pace_events]
        pace_coords = np.column_stack([pace_times, pace_amplitudes])
        pace_plot_ch1.set_offsets(pace_coords)
    else:
        pace_plot_ch1.set_offsets(np.empty((0, 2)))
    
    # Calculate sweep line position (simulates the moving cursor)
    sweep_position = ((frame * samples_per_frame) % samples_per_window) / fs
    sweep_line_ch1.set_xdata([sweep_position, sweep_position])
    sweep_line_ch2.set_xdata([sweep_position, sweep_position])
    sweep_line_pace.set_xdata([sweep_position, sweep_position])
    
    # Advance index
    current_idx += samples_per_frame
    
    return line_ch1, line_ch2, pace_line, r_peaks_plot_ch1, r_peaks_plot_ch2, pace_plot_ch1, sweep_line_ch1, sweep_line_ch2, sweep_line_pace



##################################################
# Create animation
##################################################
print("Starting animation...")
anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=None, interval=UPDATE_INTERVAL,
                             blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

print("Animation complete!")

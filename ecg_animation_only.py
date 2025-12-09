import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wfdb

from Qrs_detect.r_peak import run_pantompkins

# Download MIT-BIH data. # op sh
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

# storing original
ecg_signal = ecg_signal_ch1

# Animation parameters
WINDOW_SIZE = 5  # seconds of data to display
UPDATE_INTERVAL = 50  # milliseconds between frames
PLAYBACK_SPEED = 0.5  # 1.0 = real-time, 2.0 = 2x speed

# Calculate samples
samples_per_window = int(WINDOW_SIZE * fs)
samples_per_frame = int((UPDATE_INTERVAL / 1000) * fs * PLAYBACK_SPEED)

# Setup figure with white background for dual-channel display
fig = plt.figure(figsize=(14, 8), facecolor='white')

# Channel 1 (Primary) - Top subplot
ax_ch1 = fig.add_subplot(211)
ax_ch1.set_facecolor('white')
ax_ch1.set_ylim(-3, 3)
ax_ch1.set_xlim(0, WINDOW_SIZE)
ax_ch1.set_ylabel('Channel 1 (mV)', fontsize=10)
ax_ch1.set_title('Dual-Channel ECG Monitor - Pacemaker View', fontsize=14, pad=10)
ax_ch1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax_ch1.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.3)
ax_ch1.minorticks_on()
ax_ch1.tick_params(labelsize=8)

# Channel 2 (Secondary) - Bottom subplot  
ax_ch2 = fig.add_subplot(212)
ax_ch2.set_facecolor('white')
ax_ch2.set_ylim(-3, 3)
ax_ch2.set_xlim(0, WINDOW_SIZE)
ax_ch2.set_ylabel('Channel 2 (mV)', fontsize=10)
ax_ch2.set_xlabel('Time (seconds)', fontsize=10)
ax_ch2.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax_ch2.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.3)
ax_ch2.minorticks_on()
ax_ch2.tick_params(labelsize=8)

# Initialize plot lines
line_ch1, = ax_ch1.plot([], [], color='red', linewidth=1.5, label='Primary')
line_ch2, = ax_ch2.plot([], [], color='blue', linewidth=1.5, label='Secondary')

# Initialize R-peak markers (empty scatter plots)
r_peaks_plot_ch1 = ax_ch1.scatter([], [], color='red', marker='o', s=80, zorder=5, alpha=0.8, label='R-peaks')
r_peaks_plot_ch2 = ax_ch2.scatter([], [], color='blue', marker='o', s=80, zorder=5, alpha=0.8, label='R-peaks')

# Add legends
ax_ch1.legend(loc='upper right', fontsize=8)
ax_ch2.legend(loc='upper right', fontsize=8)

sweep_line_ch1 = ax_ch1.axvline(x=0, color='green', linewidth=2, alpha=0.7)
sweep_line_ch2 = ax_ch2.axvline(x=0, color='green', linewidth=2, alpha=0.7)

# Add time display text
time_text = ax_ch1.text(0.02, 0.95, '', transform=ax_ch1.transAxes, fontsize=12, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightblue', alpha=0.8))

# Data buffer
current_idx = 0
time_axis = np.linspace(0, WINDOW_SIZE, samples_per_window)

def init():
    """Initialize animation"""
    line_ch1.set_data([], [])
    line_ch2.set_data([], [])
    r_peaks_plot_ch1.set_offsets(np.empty((0, 2)))
    r_peaks_plot_ch2.set_offsets(np.empty((0, 2)))
    time_text.set_text('')
    return line_ch1, line_ch2, r_peaks_plot_ch1, r_peaks_plot_ch2, sweep_line_ch1, sweep_line_ch2, time_text

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
    
    # Update both plots
    line_ch1.set_data(time_axis, current_window_ch1)
    line_ch2.set_data(time_axis, current_window_ch2)
    
    # Find R-peaks in current window
    # Get R-peaks that fall within current window
    window_r_peaks_ch1 = r_peak_indices_ch1[(r_peak_indices_ch1 >= start_idx) & (r_peak_indices_ch1 < end_idx)]
    window_r_peaks_ch2 = r_peak_indices_ch2[(r_peak_indices_ch2 >= start_idx) & (r_peak_indices_ch2 < end_idx)]
    
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
    
    # Calculate sweep line position (simulates the moving cursor)
    sweep_position = ((frame * samples_per_frame) % samples_per_window) / fs
    sweep_line_ch1.set_xdata([sweep_position, sweep_position])
    sweep_line_ch2.set_xdata([sweep_position, sweep_position])
    
    # Update time display - show actual time in recording
    current_time_sec = start_idx / fs
    current_time_min = int(current_time_sec // 60)
    current_time_sec_remainder = current_time_sec % 60
    time_text.set_text(f'Recording Time: {current_time_min:02d}:{current_time_sec_remainder:05.2f}\n'
                       f'Window: {current_time_sec:.1f} - {(start_idx + samples_per_window)/fs:.1f}s')
    
    # Advance index
    current_idx += samples_per_frame
    
    return line_ch1, line_ch2, r_peaks_plot_ch1, r_peaks_plot_ch2, sweep_line_ch1, sweep_line_ch2, time_text



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

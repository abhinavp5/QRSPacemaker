"""
Real-time ECG Visualization with VVIR Pacemaker Simulation and Heart Video
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec                if len(bmp_data) == len(time_data):
                    bmp_line.set_data(time_data, bpm_data)mport GridSpec
import imageio.v2 as imageio

# Load the ECG data from MIT-BIH database
with open('./Assests/mit-bih-database-sample.txt', 'r') as f:
    ecg_samples = [float(line.strip()) for line in f]

# Simulation parameters
SAMPLE_RATE = 360  # Hz
DATA_LENGTH = len(ecg_samples)
WINDOW_SIZE = 10  # seconds
samples_in_window = WINDOW_SIZE * SAMPLE_RATE

def load_media():
    """Load video and image assets"""
    print("Loading heart video...")
    heart_reader = imageio.get_reader('./Assests/heart2.mp4', 'ffmpeg')
    return heart_reader

def plot_ecg_with_pacemaker():
    """Create the main plotting function with real-time animation"""
    heart_reader = load_media()
    
    # Create main figure with black background
    fig = plt.figure(figsize=(16, 8), facecolor='black')
    
    # Create a 3x2 grid layout: left side for ECG/pacemaker/BPM, right side for heart video
    gs = GridSpec(3, 2, figure=fig, width_ratios=[3, 1], hspace=0.3, wspace=0.2)
    
    # ECG plot panel (top-left)
    ax_ecg = fig.add_subplot(gs[0, 0])
    ax_ecg.set_xlim(0, WINDOW_SIZE)
    ax_ecg.set_ylim(-2, 3)
    ax_ecg.set_ylabel("Amplitude (mV)", color="white")
    ax_ecg.set_title("Real-time ECG with VVIR Pacing", color="white")
    ax_ecg.set_facecolor('black')
    ax_ecg.tick_params(colors='white')
    ax_ecg.grid(True, color='gray', alpha=0.3)
    
    ecg_line, = ax_ecg.plot([], [], lw=2, color="blue", label="ECG")
    pace_markers, = ax_ecg.plot([], [], 'ro', markersize=8, label="Pace Markers")
    sweep_line_ecg = ax_ecg.axvline(0, color="green", lw=2, alpha=0.7)
    ax_ecg.legend(loc="upper right")
    
    # Pacemaker status panel (middle-left)
    ax_pm = fig.add_subplot(gs[1, 0])
    ax_pm.set_title("Pacemaker Status", color="white")
    ax_pm.axis("off")
    ax_pm.set_facecolor('black')
    
    # Create text for pacemaker status
    pm_status_text = ax_pm.text(0.5, 0.7, "IDLE", fontsize=16, ha='center', va='center', 
                                color="white", transform=ax_pm.transAxes)
    pm_mode_text = ax_pm.text(0.5, 0.5, "Mode: VVIR", fontsize=12, ha='center', va='center',
                              color="white", transform=ax_pm.transAxes)
    pm_rate_text = ax_pm.text(0.5, 0.3, "Rate: 60 BPM", fontsize=12, ha='center', va='center',
                              color="white", transform=ax_pm.transAxes)
    
    # BPM tracking panel (bottom-left)
    ax_bpm = fig.add_subplot(gs[2, 0])
    ax_bpm.set_xlim(0, WINDOW_SIZE)
    ax_bpm.set_ylim(40, 120)
    ax_bpm.set_ylabel("Heart Rate (BPM)", color="white")
    ax_bpm.set_xlabel("Time (s)", color="white")
    ax_bpm.set_title("Real-time Heart Rate", color="white")
    ax_bpm.set_facecolor('black')
    ax_bpm.tick_params(colors='white')
    ax_bpm.grid(True, color='gray', alpha=0.3)
    
    bpm_line, = ax_bpm.plot([], [], lw=2, color="red", label="Instantaneous BPM")
    target_bpm_line, = ax_bpm.plot([], [], lw=2, color="purple", linestyle='--', label="Target BPM")
    sweep_line_bpm = ax_bpm.axvline(0, color="green", lw=2, alpha=0.7)
    ax_bpm.legend(loc="upper right")
    
    # Heart video panel (right side, spanning all rows)
    ax_heart = fig.add_subplot(gs[:, 1])
    ax_heart.set_title("Heart Video", color="white")
    ax_heart.axis("off")
    ax_heart.set_facecolor('black')
    
    # Initialize heart frame with the first frame of the video
    first_frame = heart_reader.get_data(0)
    heart_im = ax_heart.imshow(first_frame)
    
    # Animation data
    time_data = []
    ecg_data = []
    bpm_data = []
    pace_times = []
    pace_amplitudes = []
    frame_counter = 0
    video_frame_counter = 0
    
    # Import the simulation module
    try:
        from simulate import run_vvir_simulation
        print("Simulation module loaded successfully")
    except ImportError:
        print("Warning: simulate.py not found. Using dummy data.")
        run_vvir_simulation = None
    
    def animate(frame):
        nonlocal frame_counter, video_frame_counter
        
        current_time = frame_counter / SAMPLE_RATE
        
        if frame_counter < DATA_LENGTH - 1:
            # Get ECG sample
            ecg_value = ecg_samples[frame_counter]
            
            # Update heart video (continuous loop, playing at original speed)
            if frame % 1 == 0:  # Update every frame for original speed
                video_frame_idx = video_frame_counter % heart_reader.count_frames()
                try:
                    new_frame = heart_reader.get_data(video_frame_idx)
                    heart_im.set_array(new_frame)
                except:
                    # Reset to beginning if we reach the end
                    video_frame_idx = 0
                    new_frame = heart_reader.get_data(video_frame_idx)
                    heart_im.set_array(new_frame)
                video_frame_counter += 1
            
            # Update time and ECG data
            time_data.append(current_time)
            ecg_data.append(ecg_value)
            
            # Keep only the last WINDOW_SIZE seconds
            if len(time_data) > samples_in_window:
                time_data.pop(0)
                ecg_data.pop(0)
            
            # Update ECG plot
            ecg_line.set_data(time_data, ecg_data)
            
            # Update sweep line position
            sweep_line_ecg.set_xdata([current_time % WINDOW_SIZE])
            sweep_line_bpm.set_xdata([current_time % WINDOW_SIZE])
            
            # Simulate pacemaker logic
            if run_vvir_simulation:
                try:
                    # Run simulation for current sample
                    result = run_vvir_simulation(frame_counter, ecg_value)
                    if result:
                        paced, bpm, target_bpm = result
                        
                        # Update BPM data
                        bpm_data.append(bpm)
                        if len(bpm_data) > samples_in_window:
                            bpm_data.pop(0)
                        
                        # Update BPM plot
                        if len(bpm_data) == len(time_data):
                            bpm_line.set_data(time_data, bpm_data)
                            target_bpm_line.set_data(time_data, [target_bpm] * len(time_data))
                        
                        # Update pacemaker status
                        if paced:
                            pm_status_text.set_text("PACING")
                            pm_status_text.set_color("red")
                            pace_times.append(current_time)
                            pace_amplitudes.append(ecg_value)
                            
                            # Keep only recent pace markers
                            if len(pace_times) > 100:
                                pace_times.pop(0)
                                pace_amplitudes.pop(0)
                        else:
                            pm_status_text.set_text("SENSING")
                            pm_status_text.set_color("green")
                        
                        pm_rate_text.set_text(f"Rate: {int(bpm)} BPM")
                        
                        # Update pace markers
                        visible_pace_times = []
                        visible_pace_amps = []
                        for t, a in zip(pace_times, pace_amplitudes):
                            if current_time - WINDOW_SIZE <= t <= current_time:
                                visible_pace_times.append(t)
                                visible_pace_amps.append(a)
                        
                        pace_markers.set_data(visible_pace_times, visible_pace_amps)
                        
                except Exception as e:
                    print(f"Simulation error: {e}")
            else:
                # Dummy data if simulation unavailable
                dummy_bpm = 70 + 10 * np.sin(current_time * 0.1)
                bpm_data.append(dummy_bpm)
                if len(bpm_data) > samples_in_window:
                    bpm_data.pop(0)
                
                if len(bpm_data) == len(time_data):
                    bpm_line.set_data(time_data, bmp_data)
                    target_bpm_line.set_data(time_data, [70] * len(time_data))
            
            frame_counter += 1
        
        return [ecg_line, pace_markers, sweep_line_ecg, sweep_line_bpm, 
                bpm_line, target_bpm_line, heart_im, pm_status_text, pm_rate_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=DATA_LENGTH,
        interval=50, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    print("Starting ECG visualization with heart video...")
    anim = plot_ecg_with_pacemaker()

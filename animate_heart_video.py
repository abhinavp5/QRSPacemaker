import numpy as np
import matplotlib
# Try to use Qt5Agg if available, otherwise fallback is automatic usually, but user requested it.
try:
    matplotlib.use('Qt5Agg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2
import os
from simulate import simulate_vvir_on_mitbih

def animate_simulation_with_video(fs, ecg, sensed_beats, paced_beats, rate_trace, t):
    # --- Constants from plot_pacemaker_heart_QT5.py ---
    WINDOW_SIZE = 10
    UPDATE_INTERVAL = 50
    PLAYBACK_SPEED = 2

    N = len(ecg)
    samples_per_window = int(WINDOW_SIZE * fs)
    samples_per_frame = int((UPDATE_INTERVAL / 1000) * fs * PLAYBACK_SPEED)
    time_axis = np.linspace(0, WINDOW_SIZE, samples_per_window)

    # --- Configuration ---
    INITIAL_X = 837 # X Position of overlay
    INITIAL_Y = 54 # Y Position of overlay
    INITIAL_SCALE = 13.3 # Percentage scale of overlay
    INITIAL_THRESH = 30 # Background threshold for overlay transparency
    OVERLAY_HOLD_FRAMES = 5 # How many frames to keep the overlay visible after firing
    ALWAYS_SHOW_OVERLAY = False # Set to True to position the overlay
    SHOW_SLIDERS = False # Set to True to show adjustment sliders

    # --- Setup Video ---
    video_path = os.path.join("Assests", "heart2.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Load Overlay
    image_path = os.path.join("Assests", "Gemini_Generated_Image_8ycphr8ycphr8ycp.png")
    if not os.path.exists(image_path):
        image_path = os.path.join("Assests", "Gemini_Generated_Image_gq2kj1gq2kj1gq2k.png")
    
    overlay_img_orig = None
    if os.path.exists(image_path):
        overlay_img_orig = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if overlay_img_orig is not None:
            # Ensure alpha channel
            if overlay_img_orig.shape[2] == 3:
                overlay_img_orig = cv2.cvtColor(overlay_img_orig, cv2.COLOR_BGR2BGRA)
    
    # --- Setup Figures ---
    
    # Figure 1: Graphs (Copied from plot_pacemaker_heart_QT5.py)
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.canvas.manager.set_window_title("ECG & Pacemaker Analysis")

    # ECG panel (top)
    ax_ecg = fig.add_subplot(311)
    ax_ecg.set_xlim(0, WINDOW_SIZE)
    ax_ecg.set_ylim(-3, 3)
    ax_ecg.set_title("ECG Signal with R-peak Detection")
    ax_ecg.set_ylabel("ECG (normalized)")
    ax_ecg.grid(True)

    ecg_line, = ax_ecg.plot([], [], lw=1.5, color="black", label="ECG")
    intrinsic_scatter = ax_ecg.scatter([], [], s=80, color="red", label="R-peaks", marker='o')
    sweep_line_ecg = ax_ecg.axvline(0, color="green", lw=2, alpha=0.7)
    
    # Pacemaker panel (middle)
    ax_pace = fig.add_subplot(312)
    ax_pace.set_xlim(0, WINDOW_SIZE)
    ax_pace.set_ylim(-0.2, 1.2)
    ax_pace.set_ylabel("Pacemaker")
    ax_pace.set_title("Pacemaker Firing Events")
    ax_pace.grid(True)
    ax_pace.set_yticks([0, 1])
    ax_pace.set_yticklabels(['OFF', 'PACE'])

    pace_line, = ax_pace.plot([], [], lw=3, color="orange", label="Pacing Pulses")
    pace_scatter = ax_pace.scatter([], [], s=100, color="orange", marker='^', label="Pace Events")
    sweep_line_pace = ax_pace.axvline(0, color="green", lw=2, alpha=0.7)

    # BPM panel (bottom)
    ax_bpm = fig.add_subplot(313)
    ax_bpm.set_xlim(0, WINDOW_SIZE)
    ax_bpm.set_ylim(40, 120)
    ax_bpm.set_ylabel("Heart Rate (BPM)")
    ax_bpm.set_xlabel("Time (s)")
    ax_bpm.set_title("Real-time Heart Rate")
    ax_bpm.grid(True)
    
    bpm_line, = ax_bpm.plot([], [], lw=2, color="red", label="Instantaneous BPM")
    target_bpm_line, = ax_bpm.plot([], [], lw=2, color="purple", linestyle='--', label="Target BPM")
    sweep_line_bpm = ax_bpm.axvline(0, color="green", lw=2, alpha=0.7)
    
    # Add legends and time display
    ax_ecg.legend(loc='upper right', fontsize=8)
    ax_pace.legend(loc='upper right', fontsize=8)
    ax_bpm.legend(loc='upper right', fontsize=8)
    
    time_text = ax_ecg.text(0.02, 0.95, '', transform=ax_ecg.transAxes, fontsize=10, 
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='lightblue', alpha=0.8))

    # Figure 2: Video with Controls
    fig_video = plt.figure(figsize=(8, 8))
    fig_video.canvas.manager.set_window_title("Pacemaker Visualization & Controls")
    
    # Video Axis
    ax_video = fig_video.add_axes([0.05, 0.3, 0.9, 0.65]) # Left, Bottom, Width, Height
    ax_video.axis('off')
    
    # Sliders
    slider_x = None
    slider_y = None
    slider_scale = None
    slider_thresh = None

    if SHOW_SLIDERS:
        ax_x = fig_video.add_axes([0.2, 0.2, 0.6, 0.03])
        ax_y = fig_video.add_axes([0.2, 0.15, 0.6, 0.03])
        ax_scale = fig_video.add_axes([0.2, 0.1, 0.6, 0.03])
        ax_thresh = fig_video.add_axes([0.2, 0.05, 0.6, 0.03])
        
        slider_x = Slider(ax_x, 'X Pos', 0, video_width, valinit=INITIAL_X)
        slider_y = Slider(ax_y, 'Y Pos', 0, video_height, valinit=INITIAL_Y)
        slider_scale = Slider(ax_scale, 'Scale %', 1, 200, valinit=INITIAL_SCALE)
        slider_thresh = Slider(ax_thresh, 'BG Thresh', 0, 255, valinit=INITIAL_THRESH)
    else:
        # Adjust video axis to take up more space if sliders are hidden
        ax_video.set_position([0.05, 0.05, 0.9, 0.9])

    # Initial video frame
    ret, frame_0 = cap.read()
    im_display = None
    if ret:
        frame_rgb = cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB)
        im_display = ax_video.imshow(frame_rgb)
    
    status_text = ax_video.text(0.02, 0.95, '', transform=ax_video.transAxes, color='white', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))


    current_index = {"idx": 0, "overlay_frames_remaining": 0}

    def init():
        ecg_line.set_data([], [])
        intrinsic_scatter.set_offsets(np.empty((0, 2)))
        pace_line.set_data([], [])
        pace_scatter.set_offsets(np.empty((0, 2)))
        bpm_line.set_data([], [])
        target_bpm_line.set_data([], [])
        time_text.set_text('')
        return ecg_line, intrinsic_scatter, pace_line, pace_scatter, bpm_line, target_bpm_line, sweep_line_ecg, sweep_line_pace, sweep_line_bpm, time_text, im_display, status_text

    def update(frame):
        # --- Graph Update Logic ---
        idx = current_index["idx"]
        start = idx
        end = idx + samples_per_window

        if end >= N:
            current_index["idx"] = 0
            start = 0
            end = samples_per_window
            idx = 0

        window_ecg = ecg[start:end]
        window_paced = paced_beats[start:end]
        window_rate = rate_trace[start:end]

        # Update ECG
        ecg_line.set_data(time_axis, window_ecg)

        # Update intrinsic beats (R-peaks)
        intrinsic_idx = np.flatnonzero(sensed_beats[start:end]) + start
        if len(intrinsic_idx) > 0:
            intrinsic_times = (intrinsic_idx - start) / fs
            intrinsic_vals = ecg[intrinsic_idx]
            intrinsic_scatter.set_offsets(np.column_stack([intrinsic_times, intrinsic_vals]))
        else:
            intrinsic_scatter.set_offsets(np.empty((0, 2)))

        # Update pacemaker events
        paced_idx = np.flatnonzero(paced_beats[start:end])
        pace_signal = window_paced.astype(float)
        pace_line.set_data(time_axis, pace_signal)
        
        if len(paced_idx) > 0:
            pace_times = paced_idx / fs
            pace_vals = np.ones(len(paced_idx))
            pace_scatter.set_offsets(np.column_stack([pace_times, pace_vals]))
        else:
            pace_scatter.set_offsets(np.empty((0, 2)))

        # Calculate and update BPM with smoothing
        all_events = np.logical_or(sensed_beats[start:end], paced_beats[start:end])
        event_indices = np.where(all_events)[0]
        
        window_size_bpm = int(5 * fs)
        smoothed_bpm = np.full(len(window_ecg), 60.0)
        
        if len(event_indices) > 2:
            intervals = np.diff(event_indices) / fs
            bpms = 60.0 / intervals
            for i, event_idx in enumerate(event_indices[1:], 1):
                if i < len(bpms):
                    start_avg = max(0, i-2)
                    avg_bpm = np.mean(bpms[start_avg:i+1])
                    avg_bpm = np.clip(avg_bpm, 30, 150)
                    fill_start = max(0, event_idx - window_size_bpm//10)
                    fill_end = min(len(smoothed_bpm), event_idx + window_size_bpm//10)
                    smoothed_bpm[fill_start:fill_end] = avg_bpm

        bpm_line.set_data(time_axis, smoothed_bpm)
        target_bpm_line.set_data(time_axis, window_rate)

        # Update sweep lines
        sweep_x = ((frame * samples_per_frame) % samples_per_window) / fs
        sweep_line_ecg.set_xdata([sweep_x, sweep_x])
        sweep_line_pace.set_xdata([sweep_x, sweep_x])
        sweep_line_bpm.set_xdata([sweep_x, sweep_x])

        # Update time display
        current_time_sec = start / fs
        current_time_min = int(current_time_sec // 60)
        current_time_sec_remainder = current_time_sec % 60
        intrinsic_count = len(intrinsic_idx)
        paced_count = np.sum(window_paced)
        total_bpm = (intrinsic_count + paced_count) / WINDOW_SIZE * 60 if WINDOW_SIZE > 0 else 0
        
        time_text.set_text(f'Time: {current_time_min:02d}:{current_time_sec_remainder:05.2f}\n'
                          f'Intrinsic: {intrinsic_count} | Paced: {paced_count}\n'
                          f'Total BPM: {total_bpm:.0f}')

        # --- Video Update Logic ---
        sim_time = end / fs
        video_frame_idx = int(sim_time * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx % total_video_frames)
        ret, vid_frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
            
            check_start = max(0, end - samples_per_frame)
            check_end = end
            
            is_pacing = False
            if check_end > check_start:
                if np.any(paced_beats[check_start:check_end]):
                    is_pacing = True
            
            if is_pacing:
                current_index["overlay_frames_remaining"] = OVERLAY_HOLD_FRAMES

            show_overlay = False
            if current_index["overlay_frames_remaining"] > 0:
                show_overlay = True
                current_index["overlay_frames_remaining"] -= 1
            
            if ALWAYS_SHOW_OVERLAY:
                show_overlay = True

            if show_overlay and overlay_img_orig is not None:
                # Get slider values or defaults
                if SHOW_SLIDERS:
                    x_pos = int(slider_x.val)
                    y_pos = int(slider_y.val)
                    scale = slider_scale.val
                    thresh = slider_thresh.val
                else:
                    x_pos = int(INITIAL_X)
                    y_pos = int(INITIAL_Y)
                    scale = INITIAL_SCALE
                    thresh = INITIAL_THRESH
                
                # Process Overlay
                # 1. Resize
                new_width = int(overlay_img_orig.shape[1] * scale / 100)
                new_height = int(overlay_img_orig.shape[0] * scale / 100)
                if new_width > 0 and new_height > 0:
                    overlay_resized = cv2.resize(overlay_img_orig, (new_width, new_height))
                    
                    # 2. Background Removal (Simple Thresholding on Brightness)
                    # Assuming white background -> high brightness is transparent
                    # Or black background -> low brightness is transparent
                    # Let's assume white background for now based on "Gemini Generated" usually being full rect
                    
                    # Convert to grayscale for thresholding
                    gray = cv2.cvtColor(overlay_resized, cv2.COLOR_BGRA2GRAY)
                    
                    # Create mask: Pixels DARKER than thresh are TRANSPARENT (0)
                    # Pixels BRIGHTER than thresh are OPAQUE (255)
                    # Adjust logic based on image. If it's a black background, we want to keep bright parts.
                    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                    
                    # Update Alpha Channel
                    overlay_resized[:, :, 3] = mask
                    
                    # 3. Overlay
                    # Ensure bounds
                    h, w = overlay_resized.shape[:2]
                    
                    # Clip to video frame
                    y1 = max(0, y_pos)
                    x1 = max(0, x_pos)
                    y2 = min(video_height, y_pos + h)
                    x2 = min(video_width, x_pos + w)
                    
                    # Corresponding overlay coordinates
                    oy1 = max(0, -y_pos)
                    ox1 = max(0, -x_pos)
                    oy2 = oy1 + (y2 - y1)
                    ox2 = ox1 + (x2 - x1)
                    
                    if y2 > y1 and x2 > x1:
                        roi = frame_rgb[y1:y2, x1:x2]
                        overlay_crop = overlay_resized[oy1:oy2, ox1:ox2]
                        
                        # Convert overlay to RGB for blending
                        overlay_rgb = cv2.cvtColor(overlay_crop, cv2.COLOR_BGRA2RGB)
                        alpha = overlay_crop[:, :, 3][:, :, np.newaxis] / 255.0
                        
                        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
                        frame_rgb[y1:y2, x1:x2] = blended

                status_text.set_text("PACEMAKER: FIRING")
                status_text.set_color("red")
            else:
                status_text.set_text("PACEMAKER: MONITORING")
                status_text.set_color("lime")
            
            im_display.set_data(frame_rgb)

        current_index["idx"] += samples_per_frame

        return ecg_line, intrinsic_scatter, pace_line, pace_scatter, bpm_line, target_bpm_line, sweep_line_ecg, sweep_line_pace, sweep_line_bpm, time_text, im_display, status_text

    # We attach the animation to the main figure, but it updates both
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=UPDATE_INTERVAL,
        blit=True,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()
    cap.release()

if __name__ == "__main__":
    print("Running simulation...")
    fs, ecg, sensed_beats, paced_beats, rate_trace, t = simulate_vvir_on_mitbih()
    print("Simulation complete. Starting animation...")
    animate_simulation_with_video(fs, ecg, sensed_beats, paced_beats, rate_trace, t)

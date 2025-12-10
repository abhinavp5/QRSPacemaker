import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# matplotlib.use('Qt5Agg')  # Commented out - use default backend

from simulate import simulate_vvir_on_mitbih


def animate_simulation(fs, ecg, sensed_beats, paced_beats, rate_trace, t):
    WINDOW_SIZE = 10
    UPDATE_INTERVAL = 50
    PLAYBACK_SPEED = 2

    N = len(ecg)
    samples_per_window = int(WINDOW_SIZE * fs)
    samples_per_frame = int((UPDATE_INTERVAL / 1000) * fs * PLAYBACK_SPEED)
    time_axis = np.linspace(0, WINDOW_SIZE, samples_per_window)

    fig = plt.figure(figsize=(14, 10), facecolor="white")

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

    current_index = {"idx": 0}

    def init():
        ecg_line.set_data([], [])
        intrinsic_scatter.set_offsets(np.empty((0, 2)))
        pace_line.set_data([], [])
        pace_scatter.set_offsets(np.empty((0, 2)))
        bpm_line.set_data([], [])
        target_bpm_line.set_data([], [])
        time_text.set_text('')
        return ecg_line, intrinsic_scatter, pace_line, pace_scatter, bpm_line, target_bpm_line, sweep_line_ecg, sweep_line_pace, sweep_line_bpm, time_text

    def update(frame):
        idx = current_index["idx"]
        start = idx
        end = idx + samples_per_window

        if end >= N:
            current_index["idx"] = 0
            start = 0
            end = samples_per_window

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
        # Method 1: Instantaneous BPM (creates rectangles - current issue)
        all_events = np.logical_or(sensed_beats[start:end], paced_beats[start:end])
        event_indices = np.where(all_events)[0]
        
        # Method 2: Smoothed BPM using moving average
        window_size_bpm = int(5 * fs)  # 5-second smoothing window
        smoothed_bpm = np.full(len(window_ecg), 60.0)  # Default 60 BPM
        
        if len(event_indices) > 2:
            # Calculate intervals between consecutive beats
            intervals = np.diff(event_indices) / fs  # Convert to seconds
            bpms = 60.0 / intervals  # Convert intervals to BPM
            
            # Smooth the BPM values using a moving average
            for i, event_idx in enumerate(event_indices[1:], 1):
                if i < len(bpms):
                    # Take average of last few BPM values for smoothing
                    start_avg = max(0, i-2)  # Look back 2 beats
                    avg_bpm = np.mean(bpms[start_avg:i+1])
                    # Clamp to reasonable range
                    avg_bpm = np.clip(avg_bpm, 30, 150)
                    
                    # Fill a smaller region around the beat with smoothed BPM
                    fill_start = max(0, event_idx - window_size_bpm//10)
                    fill_end = min(len(smoothed_bpm), event_idx + window_size_bpm//10)
                    smoothed_bpm[fill_start:fill_end] = avg_bpm

        bpm_line.set_data(time_axis, smoothed_bpm)

        # Target BPM (this should be smooth already)
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

        current_index["idx"] += samples_per_frame

        return ecg_line, intrinsic_scatter, pace_line, pace_scatter, bpm_line, target_bpm_line, sweep_line_ecg, sweep_line_pace, sweep_line_bpm, time_text

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


if __name__ == "__main__":
    fs, ecg, sensed_beats, paced_beats, rate_trace, t = simulate_vvir_on_mitbih()
    animate_simulation(fs, ecg, sensed_beats, paced_beats, rate_trace, t)
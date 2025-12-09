import numpy as np
import wfdb

from Qrs_detect.r_peak import run_pantompkins
from SILPackemaker.PacingLogic import PacingLogic
from SILPackemaker.RateModulator import RateModulator


def simulate_vvir_on_mitbih(record_name="100", nbg_code="VVIR"):
    """
    Runs Pan–Tompkins + Pacemaker Simulation on MIT-BIH record.
    Returns fs, ecg_norm, sensed_beats, paced_beats, rate_trace.
    """
    print(f"Loading record {record_name}...")
    record = wfdb.rdrecord(record_name, pn_dir="mitdb")

    fs = record.fs
    ecg = record.p_signal[:, 0]
    N = len(ecg)
    t = np.arange(N) / fs

    # Normalize + convert for Pan–Tompkins
    ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)
    ecg_int = (ecg_norm * 1000).astype(np.int32)

    # --- Pan–Tompkins ---
    print("Running Pan–Tompkins...")
    det_raw = run_pantompkins(ecg_int)
    pad = N - len(det_raw)
    if pad < 0:
        raise RuntimeError("Pan-Tompkins output too long — FS/DELAY mismatch.")

    det_full = np.concatenate([np.zeros(pad, dtype=np.int32), det_raw])
    sensed_beats = det_full.astype(bool)

    # --- Pacemaker Components ---
    logic = PacingLogic(nbg_code, sampling_rate=fs)
    use_adaptive = (len(nbg_code) > 3) and (nbg_code[3] == "R")
    rate_mod = RateModulator(adaptive=use_adaptive, sampling_rate=fs)

    # Activity profile
    sensor_series = np.zeros(N)
    sensor_series[int(0.3*N):int(0.6*N)] = 60  # simulated moderate activity

    paced_beats = np.zeros(N, dtype=bool)
    rate_trace = np.zeros(N)

    print("Simulating pacemaker...")
    for n in range(N):
        hb = sensed_beats[n]
        rate_trace[n] = rate_mod.step(sensor_series[n])
        paced_beats[n] = logic.step(hb, rate_trace[n])

    # Print summary
    print("\n===== SUMMARY =====")
    print("Intrinsic beats:", sensed_beats.sum())
    print("Paced beats:", paced_beats.sum())
    print("Mean target rate:", rate_trace.mean())

    return fs, ecg_norm, sensed_beats, paced_beats, rate_trace, t


if __name__ == "__main__":
    simulate_vvir_on_mitbih()
import numpy as np
from Qrs_detect.r_peak import run_pantompkins  # your C wrapper

class PanTompkins:
    def __init__(self, sampling_rate: int = 360, delay: int = 22):
        self.fs = sampling_rate
        self.delay = delay
        self._detections = None  # bool array aligned to full ECG
        self._idx = 0

    def prepare_from_full_signal(self, ecg_array_int: np.ndarray):
        """
        Run the C Pan–Tompkins on the entire episode and cache detections.
        ecg_array_int must be int32 and match FS used in the C code.
        """
        ecg_array_int = np.asarray(ecg_array_int, dtype=np.int32)
        N = ecg_array_int.shape[0]

        det = run_pantompkins(ecg_array_int)  # length ≈ N - delay
        pad = N - det.shape[0]
        if pad < 0:
            raise ValueError("PanTompkins output longer than input; check FS/DELAY.")

        det_aligned = np.concatenate([np.zeros(pad, dtype=np.int32), det])
        self._detections = det_aligned.astype(bool)
        self._idx = 0

    def reset(self):
        self._idx = 0

    def step(self, ecg_sample) -> bool:
        """
        Called once per sample by Pacemaker.
        ecg_sample is not used here because detection was precomputed.
        """
        if self._detections is None:
            raise RuntimeError("Call prepare_from_full_signal() before using step().")

        if self._idx >= len(self._detections):
            return False

        flag = bool(self._detections[self._idx])
        self._idx += 1
        return flag
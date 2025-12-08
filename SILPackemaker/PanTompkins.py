import numpy as np
from Qrs_detect.r_peak import run_pantompkins

class PanTompkins:
    def __init__(self, sampling_rate: int = 360, delay: int = 22):
        self.fs = sampling_rate
        self.delay = delay
        self._detections = None  # np.ndarray of 0/1
        self._idx = 0            # current sample index

    def prepare(self, ecg_array: np.ndarray):
        """
        Pre-run the Pan–Tompkins C code on the full ECG for this episode.
        ecg_array: 1D numpy array of ECG samples for the entire simulation.
        """
        ecg_array = np.asarray(ecg_array, dtype=np.int32)
        det = run_pantompkins(ecg_array)  # length ~ N - delay

        # Align to original ECG length by padding at the front with zeros
        N = ecg_array.shape[0]
        pad = N - det.shape[0]
        if pad < 0:
            raise ValueError("PanTompkins output longer than input, check DELAY & C code.")
        det_aligned = np.concatenate([np.zeros(pad, dtype=np.int32), det])

        self._detections = det_aligned.astype(bool)
        self._idx = 0

    def step(self, ecg_sample) -> bool:
        """
        Called once per sample by the Pacemaker.
        ecg_sample is not used here because detection was done offline in prepare().
        Returns True only at samples corresponding to detected R-peaks.
        """
        if self._detections is None:
            raise RuntimeError("PanTompkins.prepare(ecg_array) must be called before step().")

        if self._idx >= len(self._detections):
            # Past end of signal → no more beats
            return False

        is_beat = bool(self._detections[self._idx])
        self._idx += 1
        return is_beat
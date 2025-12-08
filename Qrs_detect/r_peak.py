import os
import tempfile
import numpy as np
import ctypes


lib = ctypes.CDLL("./Qrs_detect/libpantompkins.so")

lib.init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.init.restype  = None

lib.panTompkins.argtypes = []
lib.panTompkins.restype  = None

def run_pantompkins(ecg_samples: np.ndarray) -> np.ndarray:
    """
    ecg_samples: 1D NumPy array of int (matches dataType in C).
    Returns: 1D NumPy array of 0/1 indicating detected R-peaks per sample.
    """

    
     #need to check input type later
    ecg_samples = np.asarray(ecg_samples, dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = os.path.join(tmpdir, "ecg_in.txt")
        out_path = os.path.join(tmpdir, "ecg_out.txt")

        
        np.savetxt(in_path, ecg_samples, fmt="%d")

        lib.init(in_path.encode("utf-8"), out_path.encode("utf-8"))
        lib.panTompkins()

        detections = np.loadtxt(out_path, dtype=np.int32)

    return detections


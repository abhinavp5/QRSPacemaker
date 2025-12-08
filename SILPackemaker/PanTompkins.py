'''
Sahil Code for QRS peak detection - Real-time wrapper for Pan-Tompkins algorithm
'''

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Qrs_detect'))
from r_peak import run_pantompkins


class PanTompkins:
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.sample_buffer = []
        self.detection_buffer = []
        self.sample_count = 0
        
        # Process in chunks to balance accuracy vs latency
        self.chunk_size = max(sampling_rate // 2, 180)  # 0.5 seconds minimum
        
    def step(self, ecg_sample: float) -> bool:
        '''
        Real-time QRS detection step-by-step processing
        :param ecg_sample: Single voltage value from ECG signal
        :return: True if QRS complex/heartbeat is detected at this sample, False otherwise
        '''
        
        # Add sample to buffer
        self.sample_buffer.append(ecg_sample)
        self.sample_count += 1
        
        # Need minimum samples for reliable detection
        if len(self.sample_buffer) < self.chunk_size:
            return False
        
        # Process when we have enough samples
        if len(self.sample_buffer) == self.chunk_size:
            try:
                # Prepare data for Pan-Tompkins (expects integers)
                buffer_array = np.array(self.sample_buffer, dtype=np.float32)
                
                # Normalize and convert to integer format
                if np.std(buffer_array) > 1e-8:  # Avoid division by zero
                    buffer_normalized = (buffer_array - np.mean(buffer_array)) / np.std(buffer_array)
                    buffer_int = (buffer_normalized * 1000).astype(np.int32)
                else:
                    buffer_int = np.zeros_like(buffer_array, dtype=np.int32)
                
                # Run Pan-Tompkins detection
                detection_results = run_pantompkins(buffer_int)
                
                # Store detection results
                self.detection_buffer.extend(detection_results)
                
                # Keep only recent results (avoid memory growth)
                max_buffer_size = self.sampling_rate * 10  # 10 seconds max
                if len(self.detection_buffer) > max_buffer_size:
                    excess = len(self.detection_buffer) - max_buffer_size
                    self.detection_buffer = self.detection_buffer[excess:]
                
                # Clear sample buffer for next chunk
                self.sample_buffer = []
                
            except Exception as e:
                print(f"Pan-Tompkins detection error: {e}")
                self.sample_buffer = []  # Reset on error
                return False
        
        # Return detection result for current sample
        # We look at the detection buffer at the corresponding position
        detection_index = self.sample_count - 1
        
        if detection_index < len(self.detection_buffer):
            result = self.detection_buffer[detection_index]
            return bool(result == 1)
        
        return False
    

import wfdb


# Download MIT-BIH data. # op sh
print("Downloading ECG data...")
record = wfdb.rdrecord('100', pn_dir='mitdb')

# Also get the annotations (true R-peak locations)
try:
    annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
    true_r_peaks = annotation.sample  # True R-peak sample locations
    print(f"True R-peaks in first 30s: {np.sum(true_r_peaks < fs * 30)} beats")
    
    # Show first few true R-peaks for comparison
    first_peaks = true_r_peaks[true_r_peaks < fs * 30][:10]
    print("First 10 true R-peak locations:")
    for peak in first_peaks:
        print(f"  Sample {peak} (time: {peak/fs:.2f}s)")
except Exception as e:
    print(f"Could not load annotations: {e}")
    true_r_peaks = None






ecg_signal_ch1 = record.p_signal[:, 0]  # Channel 1 (Primary)
ecg_signal_ch2 = record.p_signal[:, 1]  # Channel 2 (Secondary)
fs = record.fs  # Sampling frequency (360 Hz)

# Also get the annotations (true R-peak locations)
try:
    annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
    true_r_peaks = annotation.sample  # True R-peak sample locations
    print(f"True R-peaks in first 30s: {np.sum(true_r_peaks < fs * 30)} beats")
    
    # Show first few true R-peaks for comparison
    first_peaks = true_r_peaks[true_r_peaks < fs * 30][:10]
    print("First 10 true R-peak locations:")
    for peak in first_peaks:
        print(f"  Sample {peak} (time: {peak/fs:.2f}s)")
except Exception as e:
    print(f"Could not load annotations: {e}")
    true_r_peaks = None

print(f"ECG data length: {len(ecg_signal_ch1)} samples")
print(f"Duration: {len(ecg_signal_ch1) / fs:.1f} seconds")

ecg_signal_ch1 = (ecg_signal_ch1 - np.mean(ecg_signal_ch1)) / np.std(ecg_signal_ch1)
ecg_signal_ch2 = (ecg_signal_ch2 - np.mean(ecg_signal_ch2)) / np.std(ecg_signal_ch2)

# Run R-peak detection on both channels
print("Running R-peak detection...")
# Convert to int32 for Pan-Tompkins (as it expects integer input)
ecg_ch1_int = (ecg_signal_ch1 * 1000).astype(np.int32)  #need to check input type later
detector = PanTompkins(360)

# Process only first 10 seconds for testing (remove this line to process all data)
max_samples = min(len(ecg_ch1_int), fs * 30)  # 30 seconds max for testing
print(f"Processing {max_samples} samples ({max_samples/fs:.1f} seconds)")

for i, sample in enumerate(ecg_ch1_int[:max_samples]):
    is_peak = detector.step(sample)
    if is_peak:
        print(f"R-peak detected at sample {i} (time: {i/fs:.2f}s)")

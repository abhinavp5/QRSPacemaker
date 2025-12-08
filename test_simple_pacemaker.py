#!/usr/bin/env python3
"""
Simple Pacemaker Test - Using synthetic ECG data
This tests your pacemaker logic without external dependencies
"""

import numpy as np
import sys
import os
sys.path.append('Qrs_detect')
from r_peak import run_pantompkins
from SILPackemaker.RateModulator import RateModulator
from SILPackemaker.PacingLogic import PacingLogic

class SimplePacemakerTest:
    def __init__(self):
        self.sampling_rate = 360
        
    def create_synthetic_ecg(self, duration=30, heart_rate_bpm=45):
        """
        Create synthetic ECG with controllable heart rate
        """
        fs = self.sampling_rate
        total_samples = int(fs * duration)
        
        # Create time axis
        t = np.arange(total_samples) / fs
        
        # Calculate beat interval
        beat_interval = 60 / heart_rate_bpm  # seconds between beats
        
        # Create ECG signal
        ecg = np.zeros(total_samples)
        
        # Add R-peaks (simple spikes)
        beat_times = np.arange(0, duration, beat_interval)
        
        for beat_time in beat_times:
            peak_idx = int(beat_time * fs)
            if peak_idx < total_samples - 10:
                # Create R-peak (triangular spike)
                for i in range(10):
                    if peak_idx + i < total_samples:
                        ecg[peak_idx + i] = 1.0 * (1 - i/10)  # Decreasing spike
        
        # Add some noise
        noise = 0.1 * np.random.randn(total_samples)
        ecg += noise
        
        return ecg, beat_times
        
    def test_pacemaker_logic_only(self):
        """
        Test just the pacemaker logic components without ECG detection
        """
        print("=" * 50)
        print("PACEMAKER LOGIC TEST")
        print("=" * 50)
        
        # Test different scenarios
        scenarios = [
            {"name": "Normal Heart Rate", "rr_intervals": [1.0, 1.0, 1.0, 1.0], "expected": "No pacing"},
            {"name": "Bradycardia (Slow)", "rr_intervals": [2.0, 2.5, 2.0, 2.2], "expected": "Pacing needed"},
            {"name": "Missing Beats", "rr_intervals": [1.0, 3.0, 1.0, 2.8], "expected": "Pacing needed"},
            {"name": "Very Fast", "rr_intervals": [0.4, 0.4, 0.4, 0.4], "expected": "No pacing"},
        ]
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Initialize components
            rate_modulator = RateModulator(adaptive=True, sampling_rate=self.sampling_rate)
            pacing_logic = PacingLogic("VVIR", self.sampling_rate)
            
            activity_level = 30  # Low activity
            pace_count = 0
            beat_count = 0
            
            # Simulate RR intervals
            current_time = 0
            for rr_interval in scenario['rr_intervals']:
                # Simulate samples during this RR interval
                samples_in_interval = int(rr_interval * self.sampling_rate)
                
                for sample_idx in range(samples_in_interval):
                    # Only heartbeat at the beginning of interval
                    is_heartbeat = (sample_idx == 0)
                    if is_heartbeat:
                        beat_count += 1
                    
                    # Get target rate
                    target_rate = rate_modulator.step(activity_level)
                    
                    # Get pacing decision
                    pace_decision = pacing_logic.step(is_heartbeat, target_rate)
                    if pace_decision:
                        pace_count += 1
                    
                    current_time += 1/self.sampling_rate
            
            duration = current_time
            print(f"  Duration: {duration:.1f}s")
            print(f"  Natural beats: {beat_count} ({beat_count/duration*60:.1f} BPM)")
            print(f"  Pacing events: {pace_count} ({pace_count/duration*60:.1f} paces/min)")
            print(f"  Expected: {scenario['expected']}")
            
            if pace_count > 0:
                print("  ✅ PACEMAKER ACTIVE")
            else:
                print("  ⭕ PACEMAKER MONITORING")
    
    def test_complete_system(self):
        """
        Test complete system with synthetic ECG
        """
        print("\n" + "=" * 50)
        print("COMPLETE SYSTEM TEST")
        print("=" * 50)
        
        # Test scenarios with different heart rates
        test_cases = [
            {"heart_rate": 45, "description": "Bradycardia (Should pace)"},
            {"heart_rate": 75, "description": "Normal (Should monitor)"},
            {"heart_rate": 30, "description": "Severe bradycardia (Should pace heavily)"}
        ]
        
        for case in test_cases:
            print(f"\n--- {case['description']} ---")
            
            # Create synthetic ECG
            duration = 20  # seconds
            ecg_signal, true_beat_times = self.create_synthetic_ecg(duration, case['heart_rate'])
            
            print(f"Generated ECG: {duration}s, target {case['heart_rate']} BPM")
            print(f"True beats: {len(true_beat_times)} ({len(true_beat_times)/duration*60:.1f} BPM)")
            
            # Run R-peak detection
            try:
                ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
                ecg_int = (ecg_normalized * 1000).astype(np.int32)
                detection_results = run_pantompkins(ecg_int)
                detected_beats = np.sum(detection_results)
                
                print(f"Detected beats: {detected_beats} ({detected_beats/duration*60:.1f} BPM)")
                
                # Initialize pacemaker
                rate_modulator = RateModulator(adaptive=True, sampling_rate=self.sampling_rate)
                pacing_logic = PacingLogic("VVIR", self.sampling_rate)
                
                # Run pacemaker simulation
                pace_count = 0
                activity_level = 30
                
                for i in range(len(detection_results)):
                    is_heartbeat = detection_results[i] == 1
                    target_rate = rate_modulator.step(activity_level)
                    pace_decision = pacing_logic.step(is_heartbeat, target_rate)
                    
                    if pace_decision:
                        pace_count += 1
                
                print(f"Pacing events: {pace_count} ({pace_count/duration*60:.1f} paces/min)")
                print(f"Total cardiac events: {detected_beats + pace_count} ({(detected_beats + pace_count)/duration*60:.1f} BPM)")
                
                # Assessment
                if case['heart_rate'] < 60:  # Bradycardia
                    if pace_count > 0:
                        print("  ✅ CORRECT: Bradycardia detected, pacing activated")
                    else:
                        print("  ❌ ISSUE: Bradycardia but no pacing")
                else:  # Normal
                    if pace_count == 0:
                        print("  ✅ CORRECT: Normal rate, no pacing needed")
                    else:
                        print("  ⚠️  CHECK: Normal rate but pacing active")
                        
            except Exception as e:
                print(f"  ERROR: {e}")

def main():
    """
    Run all tests
    """
    print("Testing Pacemaker System Components")
    print("This test shows how your pacemaker logic responds to different scenarios")
    
    tester = SimplePacemakerTest()
    
    # Test 1: Just the logic components
    tester.test_pacemaker_logic_only()
    
    # Test 2: Complete system with synthetic ECG
    tester.test_complete_system()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("✅ This test shows whether your pacemaker logic:")
    print("   - Detects bradycardia correctly")
    print("   - Fires when appropriate")
    print("   - Stays inactive when not needed")
    print("   - Integrates properly with R-peak detection")

if __name__ == "__main__":
    main()

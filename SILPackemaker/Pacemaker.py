'''
Logic for taking in a chain of impulses and implementing Open Loop Pacemaker
For the sake of simplicity this implementation follow a Single - chamber Pacemaker,
This is also the the most common type of pacemaker
'''
from SILPackemaker.PacingLogic import PacingLogic
from SILPackemaker.PanTompkins import PanTompkins
from SILPackemaker.RateModulator import RateModulator

# class Pacemaker:
#     def __init__(self,NBG_code = "VVIR", sampling_rate = 360):
          
#         self.detector = PanTompkins(sampling_rate) # this detects QRS Peaks
#         self.logic  = PacingLogic(NBG_code, sampling_rate)
#         # Check index 3 (4th letter) for 'R' --> Rate Modulation Closed Loop system
#         use_adaptive_rate_modulation = (len(NBG_code) > 3) and (NBG_code[3] == "R")
#         self.rate_modulator = RateModulator(adaptive = use_adaptive_rate_modulation, sampling_rate = sampling_rate)

#     def step(self, ecg_signal, sensor_signal = None):
#         '''
#         :param ecg_signal: Voltage value from ecg signal
#         :param sensor_signal: For an adpative Closed loop system we also need a measurement, typically this would be
#         a sensor placed on the pacemaker but in our case this will be a made up value to indciate the level of activtiy 
#         that the patient. [0-100] where 0 is non activt such as sleeping and 100- is full activity such as running
#         '''

#         # flag for whether current time point is QRS complex/ heart beat
#         is_heart_beat = self.detector.step(ecg_signal) 

#         # modulate rate based on activity from sensor_signal
#         rate = self.rate_modulator.step(sensor_signal)

#         # decide whether we need pacing based on the pacemaker configuration
#         new_pace = self.logic.step(is_heart_beat,rate)
#         return new_pace

class Pacemaker:
    def __init__(self, NBG_code="VVIR", sampling_rate=360):
        self.detector = PanTompkins(sampling_rate)
        self.logic = PacingLogic(NBG_code, sampling_rate)
        use_adaptive_rate_modulation = (len(NBG_code) > 3) and (NBG_code[3] == "R")
        self.rate_modulator = RateModulator(adaptive=use_adaptive_rate_modulation,
                                            sampling_rate=sampling_rate)

    def prepare(self, ecg_array):
        # Precompute QRS detections for this episode
        self.detector.prepare(ecg_array)
        self._ecg_array = ecg_array
        self._idx = 0

    def step(self, sensor_signal=None):
        """
        One time step: uses precomputed ecg sample at current index.
        """
        if self._idx >= len(self._ecg_array):
            return False  # no more pacing

        ecg_sample = self._ecg_array[self._idx]
        self._idx += 1

        is_heart_beat = self.detector.step(ecg_sample)
        rate = self.rate_modulator.step(sensor_signal)
        new_pace = self.logic.step(is_heart_beat, rate)
        return new_pace

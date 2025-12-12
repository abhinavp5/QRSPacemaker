# Introduction
This project aims to design and evaluate a signal-processing pipeline capable of detecting irregular QRS complexes from raw ecg signals based on the paper by Pan-tompking1. The primary goal is to extract accurate R-peak locations and timing characteristics (RR intervals, variability patterns, and abnormal beat signatures) directly from raw ECG recordings. These timing metrics will then be mapped onto simplified pacemaker-relevant decision logic to simulate how a pacemaker interprets irregular cardiac rhythms, inhibits or triggers pacing, and maintains stable heart rhythm during abnormal events. Our plan is to implement preprocessing filters, develop a robust Pan–Tompkins QRS detector, compute irregularity metrics, and integrate these outputs into a pacing-decision model.  

## Process
We take in a raw electrocardiogram (ECG) response and apply the approach described in A Real Time QRS-Detection Algorithm to detect the onset and duration of QRS complexes (the depolarization of the ventricle of the heart post-contraction). The Pan-Tompkins algorithm applies a bandpass filter to the raw signal to place it in the ideal passband for detection, followed by differentiation and squaring to capture the rate of change and ensure only positive values appear. The next step involves a moving-average filter to smooth sharp edges, followed by a peak detector that detects a peak within a specific window. The second stage of our project implements the pacemaker project, which is designed as a closed-loop feedback system. For implementation, we can initialize a discrete-time counter t and count up towards a desired tdesired. If the heart has not naturally fired, we enter a discrete-time delta function to force the heart to contract and induce depolarization. Conversely, if the heart beats before the timer reaches the desired time limit for the intended frequency, then reset the timer. We plan to test this open-source ECG data from Kaggle and other sources.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QRSPacemaker
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

To run the main simulation on MIT-BIH ECG data:
```bash
python simulate.py
```

## TODO

- [ ] Add Code for more NBG Codes
- [ ] Include Simulink Control Diagram
- [ ] Hardware & Embedded Device implementation











## References

- Github code for simple QRS detection: [PanTompkinsQRS](https://github.com/rafaelmmoreira/PanTompkinsQRS/blob/master/README.txt)




- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 32(10), 230–236. [IEEE Xplore](https://www.robots.ox.ac.uk/~gari/teaching/cdt/A3/readings/ECG/Pan+Tompkins.pdf)

"""
Acquisition Module for Neural Signal Acquisition

This module is responsible for acquiring neural signals, either from real hardware
or through simulation for testing and development purposes.

Features:
- Simulates various types of neural signals (spikes, LFPs, noise)
- Supports different neurological conditions (normal, Parkinson's, epilepsy, etc.)
- Provides realistic noise profiles and artifacts
- Real-time signal processing and feature extraction
"""

import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt
from enum import Enum
import random


class NeuralState(Enum):
    """Enumeration of possible neural states/conditions"""
    NORMAL = 0
    DAMAGED = 1
    RECOVERY = 2
    PARKINSONS = 3
    EPILEPSY = 4
    ALZHEIMERS = 5


class SignalType(Enum):
    """Enumeration of possible neural signal types"""
    SPIKE = 0
    LFP = 1
    EEG = 2
    EMG = 3


class AcquisitionModule:
    """Module for neural signal acquisition and simulation"""
    
    def __init__(self, sampling_rate=1000, n_channels=4, buffer_size=10000):
        """
        Initialize the acquisition module
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz
        n_channels : int
            Number of recording channels
        buffer_size : int
            Size of the signal buffer
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, n_channels))
        self.buffer_idx = 0
        self.is_recording = False
        self.current_state = NeuralState.NORMAL
        
        # Noise parameters
        self.base_noise_level = 0.1
        self.line_noise_freq = 60  # Hz (power line noise)
        self.line_noise_amp = 0.05
        
        # Initialize time vector
        self.time = np.arange(0, buffer_size / sampling_rate, 1 / sampling_rate)
        
        # State-specific parameters dictionary
        self.state_params = {
            NeuralState.NORMAL: {
                'spike_rate': 20,  # Hz
                'lfp_freq_bands': [(4, 8, 0.5), (8, 13, 0.8), (13, 30, 0.4)],  # (start_hz, end_hz, amplitude)
                'noise_level': 0.1,
                'spike_amplitude': 1.0,
                'burst_probability': 0.1,
            },
            NeuralState.DAMAGED: {
                'spike_rate': 5,  # Lower activity
                'lfp_freq_bands': [(4, 8, 0.2), (8, 13, 0.3), (13, 30, 0.2)],  # Reduced power
                'noise_level': 0.2,  # Higher noise
                'spike_amplitude': 0.6,  # Lower amplitude
                'burst_probability': 0.05,
            },
            NeuralState.RECOVERY: {
                'spike_rate': 15,  # Recovering activity
                'lfp_freq_bands': [(4, 8, 0.4), (8, 13, 0.6), (13, 30, 0.3)],
                'noise_level': 0.15,
                'spike_amplitude': 0.8,
                'burst_probability': 0.08,
            },
            NeuralState.PARKINSONS: {
                'spike_rate': 30,  # Hyperactivity
                'lfp_freq_bands': [(13, 30, 1.5)],  # Strong beta band (parkinsonian)
                'noise_level': 0.12,
                'spike_amplitude': 1.2,
                'burst_probability': 0.3,  # More bursting
                'tremor_freq': 5,  # 5 Hz tremor
                'tremor_amp': 0.8,
            },
            NeuralState.EPILEPSY: {
                'spike_rate': 50,  # Very high during seizure
                'lfp_freq_bands': [(0.5, 4, 2.0), (4, 8, 1.5)],  # Strong delta/theta bands
                'noise_level': 0.15,
                'spike_amplitude': 1.5,
                'burst_probability': 0.8,  # Mostly bursting
                'seizure_probability': 0.01,  # Probability of seizure event per second
            },
            NeuralState.ALZHEIMERS: {
                'spike_rate': 10,  # Reduced activity
                'lfp_freq_bands': [(0.5, 4, 1.0), (4, 8, 0.5), (8, 13, 0.2)],  # Shift to lower frequencies
                'noise_level': 0.18,
                'spike_amplitude': 0.7,
                'burst_probability': 0.1,
            }
        }
        
        # Channel-specific parameters (slight variations for realism)
        self.channel_params = []
        for i in range(n_channels):
            # Create slight variations in each channel
            self.channel_params.append({
                'gain': 0.9 + 0.2 * np.random.random(),  # 0.9-1.1
                'noise_mod': 0.8 + 0.4 * np.random.random(),  # 0.8-1.2
                'phase_shift': 2 * np.pi * np.random.random(),  # 0-2π
                'electrode_quality': 0.7 + 0.3 * np.random.random()  # 0.7-1.0
            })
            
        print(f"Acquisition module initialized with {n_channels} channels at {sampling_rate} Hz")

    def start_recording(self):
        """Start the recording process"""
        self.is_recording = True
        self.recording_start_time = time.time()
        print("Recording started")
        
    def stop_recording(self):
        """Stop the recording process"""
        self.is_recording = False
        print("Recording stopped")
        
    def clear_buffer(self):
        """Clear the signal buffer"""
        self.buffer = np.zeros((self.buffer_size, self.n_channels))
        self.buffer_idx = 0
        print("Buffer cleared")
        
    def set_neural_state(self, state):
        """
        Set the neural state/condition to simulate
        
        Parameters:
        -----------
        state : NeuralState
            The neural state to simulate
        """
        if not isinstance(state, NeuralState):
            try:
                state = NeuralState[state]
            except (KeyError, TypeError):
                raise ValueError(f"Invalid neural state: {state}")
                
        self.current_state = state
        print(f"Neural state set to: {state.name}")
        
    def get_buffer_data(self):
        """
        Get the current data in the buffer
        
        Returns:
        --------
        buffer : ndarray
            The signal buffer
        """
        return self.buffer.copy()
    
    def get_recent_data(self, n_samples=1000):
        """
        Get the most recent data from the buffer
        
        Parameters:
        -----------
        n_samples : int
            Number of recent samples to return
            
        Returns:
        --------
        recent_data : ndarray
            The most recent data
        """
        if n_samples > self.buffer_size:
            n_samples = self.buffer_size
            
        if self.buffer_idx < n_samples:
            # Buffer hasn't wrapped around yet
            return self.buffer[:self.buffer_idx, :]
        else:
            # Get the most recent n_samples
            return np.concatenate([
                self.buffer[self.buffer_idx - n_samples:self.buffer_idx, :],
                self.buffer[:max(0, n_samples - (self.buffer_size - self.buffer_idx)), :]
            ])
    
    def generate_spike(self, amplitude=1.0):
        """
        Generate a realistic spike waveform
        
        Parameters:
        -----------
        amplitude : float
            Amplitude of the spike
            
        Returns:
        --------
        spike : ndarray
            The spike waveform
        """
        # Characteristic time points of the spike in ms
        t = np.linspace(-2, 5, int(0.007 * self.sampling_rate))
        
        # Generate a realistic spike shape with depolarization and repolarization
        spike = -amplitude * np.exp(-((t+0.5)/0.5)**2) + 0.1 * amplitude * np.exp(-((t-2)/1.5)**2)
        
        # Add some random variations
        spike += 0.05 * amplitude * np.random.normal(0, 1, len(spike))
        
        return spike
    
    def generate_lfp_components(self, duration, state_params):
        """
        Generate LFP components with specific frequency bands
        
        Parameters:
        -----------
        duration : float
            Duration of the signal in seconds
        state_params : dict
            Parameters specific to the current neural state
            
        Returns:
        --------
        lfp : ndarray
            The LFP signal
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)
        lfp = np.zeros(n_samples)
        
        # Generate each frequency band
        for freq_start, freq_end, amplitude in state_params['lfp_freq_bands']:
            # Create a band-limited noise in the specified frequency range
            band_width = freq_end - freq_start
            center_freq = (freq_start + freq_end) / 2
            
            # Generate a sinusoidal component with random phase shifts
            for _ in range(5):  # Multiple components per band
                f = freq_start + band_width * np.random.random()
                phase = 2 * np.pi * np.random.random()
                # Amplitude variation over time (envelope)
                env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t + np.random.random() * np.pi)
                lfp += amplitude * env * np.sin(2 * np.pi * f * t + phase)
        
        return lfp
    
    def generate_noise(self, n_samples, noise_level=0.1):
        """
        Generate realistic neural recording noise
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        noise_level : float
            Base level of the noise
            
        Returns:
        --------
        noise : ndarray
            The noise signal
        """
        # Generate pink (1/f) noise
        noise = np.random.normal(0, noise_level, n_samples)
        
        # Add 60 Hz line noise
        t = np.arange(n_samples) / self.sampling_rate
        line_noise = self.line_noise_amp * np.sin(2 * np.pi * self.line_noise_freq * t)
        
        # Add occasional artifacts
        artifacts = np.zeros(n_samples)
        if n_samples > 100 and np.random.random() < 0.1:  # 10% chance of artifact
            artifact_pos = np.random.randint(0, n_samples - 100)
            artifact_len = np.random.randint(10, 100)
            artifacts[artifact_pos:artifact_pos+artifact_len] = 3 * noise_level * np.random.randn(artifact_len)
        
        return noise + line_noise + artifacts
    
    def simulate_parkinsonian_tremor(self, n_samples, params):
        """
        Simulate parkinsonian tremor (4-6 Hz oscillation)
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        params : dict
            Parameters for the tremor
            
        Returns:
        --------
        tremor : ndarray
            The tremor signal
        """
        t = np.arange(n_samples) / self.sampling_rate
        tremor_freq = params.get('tremor_freq', 5)
        tremor_amp = params.get('tremor_amp', 1.0)
        
        # Add some frequency variation
        freq_mod = 0.2 * np.sin(2 * np.pi * 0.1 * t)  # Slow modulation of frequency
        
        # Generate tremor with amplitude modulation
        env = 0.8 + 0.2 * np.sin(2 * np.pi * 0.3 * t)  # Envelope
        tremor = tremor_amp * env * np.sin(2 * np.pi * (tremor_freq + freq_mod) * t)
        
        return tremor
    
    def simulate_epileptic_seizure(self, n_samples, params):
        """
        Simulate an epileptic seizure event
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        params : dict
            Parameters for the seizure
            
        Returns:
        --------
        seizure : ndarray
            The seizure signal
        """
        seizure = np.zeros(n_samples)
        
        # Decide if a seizure occurs in this time window
        if np.random.random() < params.get('seizure_probability', 0.01) * (n_samples / self.sampling_rate):
            # Seizure start and end points
            start = np.random.randint(0, n_samples // 2)
            duration = np.random.randint(n_samples // 5, n_samples // 2)
            end = min(start + duration, n_samples)
            
            # Generate high-amplitude oscillations with frequency changes
            t = np.arange(start, end) / self.sampling_rate
            
            # Evolving frequency: starts fast, gets faster, then slows
            seizure_len = end - start
            freq = 3 + 20 * np.sin(np.pi * np.arange(seizure_len) / seizure_len)**2
            
            # Build up and then taper down
            amp_env = np.sin(np.pi * np.arange(seizure_len) / seizure_len)
            
            # Generate the seizure pattern
            seizure[start:end] = 3.0 * amp_env * np.sin(2 * np.pi * np.cumsum(freq) / self.sampling_rate)
            
            # Add harmonics
            seizure[start:end] += 1.0 * amp_env * np.sin(4 * np.pi * np.cumsum(freq) / self.sampling_rate)
            
            print("Seizure event simulated")
        
        return seizure
    
    def simulate_bursting(self, n_samples, state_params, channel_idx):
        """
        Simulate bursting activity (multiple spikes in short succession)
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        state_params : dict
            Parameters specific to the current neural state
        channel_idx : int
            Index of the channel being simulated
            
        Returns:
        --------
        burst_signal : ndarray
            The bursting signal
        """
        burst_signal = np.zeros(n_samples)
        
        # Extract parameters
        spike_rate = state_params.get('spike_rate', 20)
        burst_prob = state_params.get('burst_probability', 0.1)
        spike_amp = state_params.get('spike_amplitude', 1.0) * self.channel_params[channel_idx]['gain']
        
        # Determine if bursting occurs (more likely in certain states)
        if np.random.random() < burst_prob:
            # Burst start point
            start = np.random.randint(0, n_samples - 500)  # Ensure room for the burst
            
            # Determine burst parameters
            burst_size = np.random.randint(3, 10)  # Number of spikes in burst
            isi = np.random.randint(10, 30)  # Inter-spike interval (in samples)
            
            # Generate each spike in the burst
            for i in range(burst_size):
                spike_pos = start + i * isi
                if spike_pos + 100 < n_samples:  # Ensure spike fits in the window
                    spike = self.generate_spike(amplitude=spike_amp)
                    burst_signal[spike_pos:spike_pos+len(spike)] += spike
            
            # Add some noise to the burst
            burst_length = burst_size * isi
            if start + burst_length < n_samples:
                burst_signal[start:start+burst_length] += 0.1 * np.random.normal(0, spike_amp, burst_length)
        
        return burst_signal
    
    def simulate_data(self, duration=1.0, neural_state=None):
        """
        Simulate neural signal data
        
        Parameters:
        -----------
        duration : float
            Duration of data to simulate in seconds
        neural_state : NeuralState
            The neural state to simulate (uses current state if None)
            
        Returns:
        --------
        data : ndarray
            Simulated neural data of shape (n_samples, n_channels)
        """
        # Use the specified state or the current state
        if neural_state is not None:
            self.set_neural_state(neural_state)
        
        # Get the parameters for the current state
        state_params = self.state_params[self.current_state]
        
        # Calculate the number of samples to generate
        n_samples = int(duration * self.sampling_rate)
        
        # Initialize the data array
        data = np.zeros((n_samples, self.n_channels))
        
        # Generate data for each channel
        for ch in range(self.n_channels):
            # Get the channel-specific parameters
            ch_params = self.channel_params[ch]
            
            # 1. Generate LFP components
            lfp = self.generate_lfp_components(duration, state_params)
            
            # 2. Generate spiking activity based on the state's spike rate
            spiking = np.zeros(n_samples)
            
            # Adjusted spike rate based on state and channel
            adjusted_rate = state_params['spike_rate'] * ch_params['electrode_quality']
            
            # Expected number of spikes in this time period
            n_spikes = int(adjusted_rate * duration)
            
            # Generate random spike times
            spike_times = np.random.randint(0, n_samples - 100, n_spikes)
            
            # Add a spike at each spike time
            for t in spike_times:
                spike = self.generate_spike(amplitude=state_params['spike_amplitude'] * ch_params['gain'])
                if t + len(spike) <= n_samples:
                    spiking[t:t+len(spike)] += spike
            
            # 3. Generate bursting activity
            bursting = self.simulate_bursting(n_samples, state_params, ch)
            
            # 4. Add condition-specific effects
            condition_effects = np.zeros(n_samples)
            
            if self.current_state == NeuralState.PARKINSONS:
                # Add parkinsonian tremor
                condition_effects += self.simulate_parkinsonian_tremor(n_samples, state_params)
                
            elif self.current_state == NeuralState.EPILEPSY:
                # Add potential epileptic seizure
                condition_effects += self.simulate_epileptic_seizure(n_samples, state_params)
            
            # 5. Generate noise
            noise_level = state_params['noise_level'] * ch_params['noise_mod']
            noise = self.generate_noise(n_samples, noise_level)
            
            # 6. Combine all components with channel-specific phase shift
            t = np.arange(n_samples) / self.sampling_rate
            phase_shift = ch_params['phase_shift']
            
            # Apply phase shift to LFP component
            lfp_shifted = np.interp(
                t, 
                t, 
                lfp, 
                period=duration
            )
            
            # Combine all components
            data[:, ch] = lfp_shifted + spiking + bursting + condition_effects + noise
        
        # If recording, add to buffer
        if self.is_recording:
            # Calculate how many samples will fit in the buffer
            remaining_space = self.buffer_size - self.buffer_idx
            if n_samples <= remaining_space:
                # All data fits in the current buffer position
                self.buffer[self.buffer_idx:self.buffer_idx+n_samples] = data
                self.buffer_idx += n_samples
            else:
                # Data wraps around the buffer
                # First fill the remaining space
                self.buffer[self.buffer_idx:] = data[:remaining_space]
                # Then wrap around to the beginning
                samples_to_wrap = n_samples - remaining_space
                self.buffer[:samples_to_wrap] = data[remaining_space:]
                self.buffer_idx = samples_to_wrap
        
        return data
    
    def plot_simulated_data(self, duration=5.0, neural_state=None):
        """
        Simulate and plot neural data
        
        Parameters:
        -----------
        duration : float
            Duration of data to simulate in seconds
        neural_state : NeuralState
            The neural state to simulate
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure containing the plots
        """
        # Simulate the data
        data = self.simulate_data(duration, neural_state)
        
        # Create time vector
        t = np.arange(data.shape[0]) / self.sampling_rate
        
        # Create the figure
        fig, axes = plt.subplots(self.n_channels, 1, figsize=(12, 3*self.n_channels), sharex=True)
        
        # Make sure axes is always a list
        if self.n_channels == 1:
            axes = [axes]
        
        # Plot each channel
        for ch in range(self.n_channels):
            axes[ch].plot(t, data[:, ch], 'k-', lw=1)
            axes[ch].set_ylabel(f'Channel {ch+1}')
            axes[ch].set_title(f'Channel {ch+1} - {self.current_state.name}')
            
            # Add a scale bar (1 second, 1 unit amplitude)
            axes[ch].plot([t[-1]-1, t[-1]], [-2, -2], 'k-', lw=2)
            axes[ch].plot([t[-1], t[-1]], [-2, -1], 'k-', lw=2)
            axes[ch].text(t[-1]-0.5, -2.2, '1 s', ha='center')
            axes[ch].text(t[-1]+0.2, -1.5, '1 unit', va='center')
            
            # Set y-limits with some margin
            y_max = max(2, np.max(data[:, ch]) * 1.2)
            y_min = min(-2, np.min(data[:, ch]) * 1.2)
            axes[ch].set_ylim(y_min, y_max)
        
        # Set x-label for the bottom plot only
        axes[-1].set_xlabel('Time (s)')
        
        # Add an overall title
        plt.suptitle(f'Simulated Neural Signals - {self.current_state.name} State', fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def analyze_frequency_content(self, data=None, channel=0, window_size=1000):
        """
        Analyze the frequency content of the signal
        
        Parameters:
        -----------
        data : ndarray
            The data to analyze. If None, get from buffer
        channel : int
            The channel to analyze
        window_size : int
            The number of samples to analyze
            
        Returns:
        --------
        freqs : ndarray
            The frequency vector
        psd : ndarray
            The power spectral density
        """
        if data is None:
            data = self.get_recent_data(window_size)
            
        if data.shape[0] < window_size:
            window_size = data.shape[0]
            
        # Extract the specified channel
        if data.ndim > 1:
            signal = data[:window_size, channel]
        else:
            signal = data[:window_size]
            
        # Calculate the PSD using Welch's method
        freqs, psd = signal.welch(signal, self.sampling_rate, nperseg=min(512, window_size // 2))
        
        return freqs, psd
    
    def extract_features(self, data=None, window_size=1000):
        """
        Extract features from the signal
        
        Parameters:
        -----------
        data : ndarray
            The data to analyze. If None, get from buffer
        window_size : int
            The number of samples to analyze
            
        Returns:
        --------
        features : dict
            Dictionary of extracted features
        """
        if data is None:
            data = self.get_recent_data(window_size)
            
        if data.shape[0] < window_size:
            window_size = data.shape[0]
            
        # Initialize feature dictionary
        features = {}
        
        # Calculate features for each channel
        for ch in range(min(self.n_channels, data.shape[1])):
            signal = data[:window_size, ch]
            
            # Time domain features
            features[f'mean_ch{ch}'] = np.mean(signal)
            features[f'std_ch{ch}'] = np.std(signal)
            features[f'max_ch{ch}'] = np.max(signal)
            features[f'min_ch{ch}'] = np.min(signal)
            features[f'range_ch{ch}'] = np.max(signal) - np.min(signal)
            features[f'rms_ch{ch}'] = np.sqrt(np.mean(signal**2))
            
            # Estimate spike rate
            threshold = features[f'std_ch{ch}'] * 3  # 3 sigma threshold
            crossings = np.where(np.diff((signal > threshold).astype(int)) > 0)[0]
            features[f'spike_rate_ch{ch}'] = len(crossings) / (window_size / self.sampling_rate)
            
            # Frequency domain features
            freqs, psd = self.analyze_frequency_content(data, ch, window_size)
            
            # Frequency bands power
            freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
            
            for band_name, (low_freq, high_freq) in freq_bands.items():
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(mask):
                    features[f'{band_name}_power_ch{ch}'] = np.sum(psd[mask])
            
            # Calculate beta/theta ratio (useful for Parkinson's)
            if ('beta_power_ch{ch}' in features and 
                'theta_power_ch{ch}' in features and 
                features[f'theta_power_ch{ch}'] > 0):
                features[f'beta_theta_ratio_ch{ch}'] = features[f'beta_power_ch{ch}'] / features[f'theta_power_ch{ch}']
            
        return features


# Example usage
if __name__ == "__main__":
    # Create an acquisition module
    acq = AcquisitionModule(sampling_rate=1000, n_channels=4)
    
    # Simulate and plot normal state
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    data_normal = acq.simulate_data(duration=2.0, neural_state=NeuralState.NORMAL)
    plt.plot(np.arange(len(data_normal))/acq.sampling_rate, data_normal[:, 0])
    plt.title('Normal State')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    data_parkinsons = acq.simulate_data(duration=2.0, neural_state=NeuralState.PARKINSONS)
    plt.plot(np.arange(len(data_parkinsons))/acq.sampling_rate, data_parkinsons[:, 0])
    plt.title('Parkinsonian State (Note tremor oscillations)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    data_epilepsy = acq.simulate_data(duration=2.0, neural_state=NeuralState.EPILEPSY)
    plt.plot(np.arange(len(data_epilepsy))/acq.sampling_rate, data_epilepsy[:, 0])
    plt.title('Epileptic State')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('../images/simulated_neural_states.png', dpi=300)
    plt.show()

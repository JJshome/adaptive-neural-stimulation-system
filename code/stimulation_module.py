# adaptive-neural-stimulation-system/code/stimulation_module.py
"""
Stimulation Module for Neural Electrical Stimulation

This module handles the electrical stimulation parameters and interfaces
with stimulation hardware (or provides a simulation when hardware is unavailable).

Features:
- Parameter validation and safety checks
- Stimulation profile generation
- Hardware interface (or simulation)
- Stimulation logging
- Real-time waveform visualization
- Support for various stimulation patterns tailored for neural regeneration
"""

import numpy as np
import time
import threading
import json
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # Not directly used with current plt.ion() approach, but kept for context
from datetime import datetime
from enum import Enum
import logging
import pickle # Not used in this version, but kept from original import
import scipy.signal as signal # Not used in this version, but kept from original import

# Import configuration and logging setup
from config import SAFETY_LIMITS, PATTERN_TEMPLATES
from logger_config import setup_logging

# Setup logger for this module
logger = setup_logging()

class StimulationMode(Enum):
    """Stimulation mode enumeration"""
    BIPHASIC = "BIPHASIC"
    MONOPHASIC_CATHODIC = "MONOPHASIC_CATHODIC"
    MONOPHASIC_ANODIC = "MONOPHASIC_ANODIC"
    BURST = "BURST"
    HIGH_FREQUENCY = "HIGH_FREQUENCY"
    THETA_BURST = "THETA_BURST"
    LONG_DURATION_LOW_FREQUENCY = "LONG_DURATION_LOW_FREQUENCY"
    PAIRED_ASSOCIATIVE = "PAIRED_ASSOCIATIVE"
    RANDOMIZED = "RANDOMIZED"

class NeuralRegenerationPattern(Enum):
    """Neural regeneration stimulation patterns based on research"""
    EARLY_STAGE = "EARLY_STAGE"
    MID_STAGE = "MID_STAGE"
    LATE_STAGE = "LATE_STAGE"
    ACUTE_INJURY = "ACUTE_INJURY"
    CHRONIC_INJURY = "CHRONIC_INJURY"
    MOTOR_NERVE = "MOTOR_NERVE"
    SENSORY_NERVE = "SENSORY_NERVE"
    BDNF_ENHANCING = "BDNF_ENHANCING"
    GDNF_ENHANCING = "GDNF_ENHANCING"

    @property
    def default_params(self) -> dict:
        """Returns the default parameters for this pattern from config."""
        return PATTERN_TEMPLATES.get(self.value, {})

class StimulationChannel:
    """Represents a stimulation channel with its parameters and state."""
    
    def __init__(self, channel_id: int, safety_limits: dict):
        """
        Initialize a stimulation channel.
        
        Parameters:
        -----------
        channel_id : int
            Channel identifier
        safety_limits : dict
            Dictionary of safety limits for parameters.
        """
        self.channel_id = channel_id
        self._safety_limits = safety_limits
        self.logger = logging.getLogger(f"StimulationSystem.Channel{channel_id}")
        
        # Default parameters (using consistent naming)
        self.amplitude_mA = 1.0       # mA
        self.pulse_width_us = 200     # μs
        self.frequency_hz = 50        # Hz
        self.duty_cycle_percent = 100 # %
        self.mode = StimulationMode.BIPHASIC
        self.burst_count = 3
        self.burst_frequency_hz = 5   # Hz (for burst mode)
        self.interphase_gap_us = 100  # μs (for biphasic pulses)
        
        # Advanced parameters for neural regeneration
        self.ramp_up_time_s = 0.5    # Seconds for amplitude to reach target
        self.ramp_down_time_s = 0.5  # Seconds for amplitude to drop to zero
        self.phase_asymmetry_ratio = 1.0 # Ratio between positive and negative phases
        self.pulse_train_interval_s = 1.0  # Seconds between pulse trains
        self.random_frequency_range_hz = (0, 0)  # For randomized stimulation
        self.random_amplitude_range_mA = (0, 0)  # For randomized stimulation
        
        # Status
        self._is_stimulating = False
        self._start_time = None
        self._stop_time = None
        self._current_amplitude_factor = 0.0 # For ramp up/down
        
        # Waveform generation for visualization
        self._current_waveform = np.array([])
        self._waveform_time = np.array([])

        # Thread safety lock for parameters and state
        self._lock = threading.Lock()
    
    @property
    def is_stimulating(self) -> bool:
        """Returns True if the channel is currently stimulating."""
        with self._lock:
            return self._is_stimulating

    @property
    def current_waveform(self) -> np.ndarray:
        """Returns the last generated waveform for visualization."""
        with self._lock:
            return self._current_waveform.copy()

    @property
    def waveform_time(self) -> np.ndarray:
        """Returns the time vector for the last generated waveform."""
        with self._lock:
            return self._waveform_time.copy()

    def _validate_parameters(self, params: dict) -> bool:
        """
        Validate stimulation parameters for safety.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameter values to validate.
            
        Returns:
        --------
        valid : bool
            Whether the parameters are valid.
        """
        for param, value in params.items():
            if param in self._safety_limits:
                min_val, max_val = self._safety_limits[param]
                if not (min_val <= value <= max_val):
                    self.logger.warning(
                        f"Channel {self.channel_id}: Parameter '{param}' value {value} "
                        f"is outside safe range [{min_val}, {max_val}]."
                    )
                    return False
        
        # Advanced validation: Charge density safety check
        # Ensure 'amplitude_mA' and 'pulse_width_us' are present for this check
        amplitude = params.get('amplitude_mA', self.amplitude_mA)
        pulse_width = params.get('pulse_width_us', self.pulse_width_us)
        
        if 'charge_mC_per_phase' in self._safety_limits:
            # Calculate charge per phase (mA * us -> mC)
            # 1 mA * 1 us = 1 nC = 0.000001 mC
            charge_per_phase = (amplitude * pulse_width) * 1e-6 
            
            min_charge, max_charge = self._safety_limits['charge_mC_per_phase']
            
            if not (min_charge <= charge_per_phase <= max_charge):
                self.logger.error(
                    f"Channel {self.channel_id}: Calculated charge density ({charge_per_phase:.6f} mC) "
                    f"is outside safe range [{min_charge}, {max_charge}] mC. "
                    "This is a critical safety violation. Parameters not applied."
                )
                return False # CRITICAL: Return False for safety violation

        # Check for required parameters for a basic stimulation
        # This check is more relevant when setting a full protocol, less so for individual param updates
        # For simplicity, we assume basic parameters are always present or default.
        
        return True

    def set_parameters(self, params: dict) -> bool:
        """
        Set stimulation parameters for this channel after validation.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameter values.
            
        Returns:
        --------
        success : bool
            Whether the parameters were set successfully.
        """
        with self._lock:
            # Create a temporary dict to validate current + new params
            temp_params = self.get_parameters() # Get current params
            
            # Convert Enum values to their string representation for validation if needed
            for key, value in params.items():
                if isinstance(value, Enum):
                    temp_params[key] = value.value
                else:
                    temp_params[key] = value

            if not self._validate_parameters(temp_params):
                self.logger.error(f"Channel {self.channel_id}: Parameter validation failed. Parameters not applied.")
                return False
            
            # Apply parameters if valid
            for key, value in params.items():
                if hasattr(self, key):
                    if isinstance(getattr(self, key), Enum) and isinstance(value, str):
                        # Convert string back to Enum if target is Enum
                        setattr(self, key, type(getattr(self, key))[value])
                    else:
                        setattr(self, key, value)
                else:
                    self.logger.warning(f"Channel {self.channel_id}: Unknown parameter '{key}'. Ignoring.")

            self.logger.info(f"Channel {self.channel_id}: Parameters updated.")
            return True
    
    def get_parameters(self) -> dict:
        """
        Get current stimulation parameters.
        
        Returns:
        --------
        params : dict
            Dictionary of parameter values.
        """
        with self._lock:
            # Dynamically get all relevant parameters
            params = {
                'amplitude_mA': self.amplitude_mA,
                'pulse_width_us': self.pulse_width_us,
                'frequency_hz': self.frequency_hz,
                'duty_cycle_percent': self.duty_cycle_percent,
                'mode': self.mode.value, # Return Enum value as string
                'burst_count': self.burst_count,
                'burst_frequency_hz': self.burst_frequency_hz,
                'interphase_gap_us': self.interphase_gap_us,
                'ramp_up_time_s': self.ramp_up_time_s,
                'ramp_down_time_s': self.ramp_down_time_s,
                'phase_asymmetry_ratio': self.phase_asymmetry_ratio,
                'pulse_train_interval_s': self.pulse_train_interval_s,
                'random_frequency_range_hz': self.random_frequency_range_hz,
                'random_amplitude_range_mA': self.random_amplitude_range_mA
            }
            return params
    
    def start_stimulation(self) -> bool:
        """Start stimulation on this channel."""
        with self._lock:
            if not self._is_stimulating:
                self._is_stimulating = True
                self._start_time = time.time()
                self._stop_time = None
                self._current_amplitude_factor = 0.0 # Start ramp-up
                self.logger.info(f"Channel {self.channel_id}: Stimulation started.")
                return True
            self.logger.info(f"Channel {self.channel_id}: Stimulation already running.")
            return False
    
    def stop_stimulation(self) -> bool:
        """Stop stimulation on this channel."""
        with self._lock:
            if self._is_stimulating:
                self._is_stimulating = False
                self._stop_time = time.time()
                self.logger.info(f"Channel {self.channel_id}: Stimulation stopped.")
                return True
            self.logger.info(f"Channel {self.channel_id}: Stimulation already stopped.")
            return False

    def _generate_waveform_segment(self, duration_ms: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate stimulation waveform segment based on channel parameters.
        This function is called periodically to get the current waveform.
        
        Parameters:
        -----------
        duration_ms : float
            Duration of the waveform segment to generate in milliseconds.
        
        Returns:
        --------
        waveform : ndarray
            Generated current waveform (mA).
        time_vector : ndarray
            Time vector for the waveform (s).
        """
        with self._lock: # Lock to ensure consistent parameter reading
            amplitude = self.amplitude_mA
            pulse_width = self.pulse_width_us
            frequency = self.frequency_hz
            duty_cycle = self.duty_cycle_percent
            mode = self.mode
            burst_count = self.burst_count
            burst_frequency = self.burst_frequency_hz
            interphase_gap = self.interphase_gap_us
            phase_asymmetry = self.phase_asymmetry_ratio
            
            # Handle randomized parameters
            if mode == StimulationMode.RANDOMIZED:
                if self.random_frequency_range_hz[1] > 0:
                    frequency = np.random.uniform(
                        self.random_frequency_range_hz[0], 
                        self.random_frequency_range_hz[1]
                    )
                if self.random_amplitude_range_mA[1] > 0:
                    amplitude = np.random.uniform(
                        self.random_amplitude_range_mA[0], 
                        self.random_amplitude_range_mA[1]
                    )

            # Calculate timing parameters
            sampling_rate = 100000  # Hz (100 kHz for accurate pulse shapes)
            
            # Duration for the waveform segment
            segment_duration_s = duration_ms / 1000.0
            n_samples = int(segment_duration_s * sampling_rate)
            time_vector = np.linspace(0, segment_duration_s, n_samples, endpoint=False)
            
            waveform = np.zeros(n_samples)

            # Generate waveform based on mode
            if mode == StimulationMode.BIPHASIC:
                waveform = self._generate_biphasic_waveform_segment_impl(
                    time_vector, frequency, amplitude,
                    pulse_width, interphase_gap, phase_asymmetry
                )
            elif mode == StimulationMode.MONOPHASIC_CATHODIC:
                waveform = self._generate_monophasic_waveform_segment_impl(
                    time_vector, frequency, -amplitude, pulse_width
                )
            elif mode == StimulationMode.MONOPHASIC_ANODIC:
                waveform = self._generate_monophasic_waveform_segment_impl(
                    time_vector, frequency, amplitude, pulse_width
                )
            elif mode == StimulationMode.BURST:
                waveform = self._generate_burst_waveform_segment_impl(
                    time_vector, frequency, amplitude,
                    pulse_width, interphase_gap, burst_count, burst_frequency
                )
            elif mode == StimulationMode.HIGH_FREQUENCY:
                # High frequency is essentially biphasic with high frequency
                waveform = self._generate_biphasic_waveform_segment_impl(
                    time_vector, frequency, amplitude,
                    pulse_width, interphase_gap, phase_asymmetry
                )
            elif mode == StimulationMode.THETA_BURST:
                waveform = self._generate_theta_burst_waveform_segment_impl(
                    time_vector, amplitude, pulse_width, interphase_gap, burst_count
                )
            elif mode == StimulationMode.LONG_DURATION_LOW_FREQUENCY:
                # This mode might involve long silent periods between pulses/trains
                # For a short segment, it might just be a single pulse or silence
                waveform = self._generate_biphasic_waveform_segment_impl(
                    time_vector, frequency, amplitude,
                    pulse_width, interphase_gap, phase_asymmetry
                )
                # Further logic needed if pulse_train_interval_s is larger than segment_duration_s
            elif mode == StimulationMode.PAIRED_ASSOCIATIVE:
                # Placeholder for complex paired associative stimulation
                waveform = self._generate_biphasic_waveform_segment_impl(
                    time_vector, frequency, amplitude,
                    pulse_width, interphase_gap, phase_asymmetry
                )
            elif mode == StimulationMode.RANDOMIZED:
                 # Already handled randomized amplitude/frequency above
                waveform = self._generate_biphasic_waveform_segment_impl(
                    time_vector, frequency, amplitude,
                    pulse_width, interphase_gap, phase_asymmetry
                )
            else:
                self.logger.warning(f"Channel {self.channel_id}: Unknown stimulation mode '{mode.value}'. Generating zero waveform.")
                waveform = np.zeros(n_samples)
            
            # Apply duty cycle if less than 100%
            if duty_cycle < 100:
                active_samples = int(n_samples * duty_cycle / 100)
                waveform[active_samples:] = 0 # Zero out samples beyond duty cycle
            
            # Apply ramp factor (already managed in _stimulation_loop)
            waveform *= self._current_amplitude_factor
            
            # Store for visualization (thread-safe access via properties)
            self._current_waveform = waveform
            self._waveform_time = time_vector
            
            return waveform, time_vector
    
    def _generate_biphasic_waveform_segment_impl(self, time_vector: np.ndarray, frequency_hz: float, amplitude_mA: float,
                                            pulse_width_us: float, interphase_gap_us: float, phase_asymmetry_ratio: float) -> np.ndarray:
        """Generate a segment of biphasic stimulation waveform."""
        waveform = np.zeros_like(time_vector)
        
        if frequency_hz <= 0: return waveform
        
        pulse_period_s = 1.0 / frequency_hz
        pulse_width_s = pulse_width_us * 1e-6
        interphase_gap_s = interphase_gap_us * 1e-6
        
        # Calculate phase durations based on asymmetry
        phase1_duration_s = pulse_width_s / (1 + phase_asymmetry_ratio)
        phase2_duration_s = pulse_width_s - phase1_duration_s
        
        # Calculate phase amplitudes based on asymmetry to maintain charge balance (if desired)
        # For current-controlled, amplitude is usually fixed, and duration varies.
        # Here, we vary amplitude for simplicity based on ratio.
        amplitude1 = -amplitude_mA
        amplitude2 = amplitude_mA / phase_asymmetry_ratio

        for i, t in enumerate(time_vector):
            time_in_period = t % pulse_period_s
            
            if 0 <= time_in_period < phase1_duration_s:
                waveform[i] = amplitude1
            elif phase1_duration_s <= time_in_period < (phase1_duration_s + interphase_gap_s):
                waveform[i] = 0 # Interphase gap
            elif (phase1_duration_s + interphase_gap_s) <= time_in_period < (phase1_duration_s + interphase_gap_s + phase2_duration_s):
                waveform[i] = amplitude2
        
        return waveform
    
    def _generate_monophasic_waveform_segment_impl(self, time_vector: np.ndarray, frequency_hz: float, amplitude_mA: float, pulse_width_us: float) -> np.ndarray:
        """Generate a segment of monophasic stimulation waveform."""
        waveform = np.zeros_like(time_vector)
        
        if frequency_hz <= 0: return waveform
        
        pulse_period_s = 1.0 / frequency_hz
        pulse_width_s = pulse_width_us * 1e-6
        
        for i, t in enumerate(time_vector):
            time_in_period = t % pulse_period_s
            if 0 <= time_in_period < pulse_width_s:
                waveform[i] = amplitude_mA
        
        return waveform
    
    def _generate_burst_waveform_segment_impl(self, time_vector: np.ndarray, frequency_hz: float, amplitude_mA: float,
                                         pulse_width_us: float, interphase_gap_us: float, burst_count: int, burst_frequency_hz: float) -> np.ndarray:
        """Generate a segment of burst stimulation waveform."""
        waveform = np.zeros_like(time_vector)
        
        if burst_frequency_hz <= 0: return waveform
        
        burst_period_s = 1.0 / burst_frequency_hz # Period between bursts
        pulse_period_within_burst_s = 1.0 / frequency_hz # Period between pulses within a burst
        
        pulse_width_s = pulse_width_us * 1e-6
        interphase_gap_s = interphase_gap_us * 1e-6
        
        # Total duration of one burst (burst_count pulses)
        total_burst_duration_s = burst_count * pulse_period_within_burst_s
        
        for i, t in enumerate(time_vector):
            time_in_burst_period = t % burst_period_s
            
            if time_in_burst_period < total_burst_duration_s:
                # Within the active burst window
                time_within_burst = time_in_burst_period
                
                # Determine which pulse within the burst
                pulse_idx = int(time_within_burst / pulse_period_within_burst_s)
                time_in_pulse_period = time_within_burst % pulse_period_within_burst_s
                
                if pulse_idx < burst_count:
                    # Generate a single biphasic pulse
                    if 0 <= time_in_pulse_period < pulse_width_s:
                        waveform[i] = -amplitude_mA
                    elif pulse_width_s <= time_in_pulse_period < (pulse_width_s + interphase_gap_s):
                        waveform[i] = 0
                    elif (pulse_width_s + interphase_gap_s) <= time_in_pulse_period < (2 * pulse_width_s + interphase_gap_s):
                        waveform[i] = amplitude_mA
        
        return waveform
    
    def _generate_theta_burst_waveform_segment_impl(self, time_vector: np.ndarray, amplitude_mA: float, pulse_width_us: float, interphase_gap_us: float, burst_count: int) -> np.ndarray:
        """Generate a segment of theta burst stimulation waveform (TBS)."""
        # TBS parameters: typically 3-5 pulses at 50Hz, repeated at 5Hz theta rhythm
        theta_freq_hz = 5  # Hz (burst repetition frequency)
        burst_pulse_freq_hz = 50  # Hz (frequency of pulses within a burst)
        
        waveform = np.zeros_like(time_vector)
        
        theta_period_s = 1.0 / theta_freq_hz
        pulse_period_within_burst_s = 1.0 / burst_pulse_freq_hz
        
        pulse_width_s = pulse_width_us * 1e-6
        interphase_gap_s = interphase_gap_us * 1e-6
        
        # Total duration of one burst (burst_count pulses)
        total_burst_duration_s = burst_count * pulse_period_within_burst_s
        
        for i, t in enumerate(time_vector):
            time_in_theta_period = t % theta_period_s
            
            if time_in_theta_period < total_burst_duration_s:
                # Within the active burst window
                time_within_burst = time_in_theta_period
                
                # Determine which pulse within the burst
                pulse_idx = int(time_within_burst / pulse_period_within_burst_s)
                time_in_pulse_period = time_within_burst % pulse_period_within_burst_s
                
                if pulse_idx < burst_count:
                    # Generate a single biphasic pulse
                    if 0 <= time_in_pulse_period < pulse_width_s:
                        waveform[i] = -amplitude_mA
                    elif pulse_width_s <= time_in_pulse_period < (pulse_width_s + interphase_gap_s):
                        waveform[i] = 0
                    elif (pulse_width_s + interphase_gap_s) <= time_in_pulse_period < (2 * pulse_width_s + interphase_gap_s):
                        waveform[i] = amplitude_mA
        
        return waveform

    def _initialize_hardware(self):
        """Placeholder for actual hardware initialization."""
        self.logger.info(f"Initializing hardware for device ID: {self.device_id}")
        # In a real system, this would involve connecting to the device,
        # loading firmware, performing self-tests, etc.
        time.sleep(0.5) # Simulate initialization time
        self.logger.info("Hardware initialization complete (simulated).")

    def _apply_parameters_to_hardware(self, channel_id: int, params: dict) -> bool:
        """
        Simulates applying parameters to the stimulation hardware.
        In a real system, this would send commands via a specific hardware SDK/API.
        """
        if not self.hardware_connected:
            self.logger.debug(f"Hardware not connected. Skipping actual hardware command for Channel {channel_id}.")
            return True # Simulate success if no hardware
        
        try:
            # Simulate hardware communication delay
            time.sleep(0.001) # Very short delay for real-time simulation
            self.logger.debug(f"Hardware: Channel {channel_id} received parameters: {params}")
            # Actual hardware command would go here
            return True
        except Exception as e:
            self.logger.error(f"Hardware communication error for Channel {channel_id}: {e}")
            return False
    
    def _stimulation_loop(self, duration: float | None):
        """
        Main stimulation loop. Runs in a separate thread.
        Generates waveforms and applies them to simulated hardware.
        """
        self.logger.info(f"Stimulation loop started" + (f" for {duration} seconds" if duration else " indefinitely."))
        start_time = time.time()
        
        try:
            while self._running:
                current_time = time.time()
                if duration is not None and (current_time - start_time) >= duration:
                    self.logger.info("Stimulation duration reached, stopping.")
                    self.stop_stimulation() # Call stop to clean up
                    break
                
                for channel in self.channels:
                    with channel._lock: # Acquire lock to safely read/modify channel state
                        if channel.is_stimulating:
                            # Update ramp factor
                            if channel.ramp_up_time_s > 0 and (current_time - channel._start_time) < channel.ramp_up_time_s:
                                channel._current_amplitude_factor = (current_time - channel._start_time) / channel.ramp_up_time_s
                            elif channel.ramp_down_time_s > 0 and channel._stop_time is not None and (current_time - channel._stop_time) < channel.ramp_down_time_s:
                                # This path is for when stop_stimulation was called and ramp-down is active
                                channel._current_amplitude_factor = 1.0 - ((current_time - channel._stop_time) / channel.ramp_down_time_s)
                                if channel._current_amplitude_factor <= 0: # Ensure it goes to zero
                                    channel._current_amplitude_factor = 0.0
                                    channel.stop_stimulation() # Fully stop if ramped down
                            else:
                                channel._current_amplitude_factor = 1.0 # Full amplitude
                            
                            # Generate and apply waveform (waveform generation also uses channel's lock)
                            waveform, waveform_time = channel._generate_waveform_segment(duration_ms=100) # Generate 100ms segment
                            
                            # Simulate applying parameters to hardware
                            params_to_apply = channel.get_parameters() # Get current parameters
                            params_to_apply['amplitude_mA'] *= channel._current_amplitude_factor # Apply ramp factor to amplitude
                            
                            hardware_success = self._apply_parameters_to_hardware(channel.channel_id, params_to_apply)
                            
                            if not hardware_success:
                                self.logger.error(f"Failed to apply stimulation to hardware for channel {channel.channel_id}. Attempting emergency stop.")
                                self._emergency_stop() # Critical error, stop everything
                                break # Exit loop
                
                time.sleep(0.05) # Simulate a 50ms stimulation cycle (20 Hz update rate)
        
        except Exception as e:
            self.logger.critical(f"Critical error in stimulation loop: {e}", exc_info=True)
            self._emergency_stop()
        finally:
            self.logger.info("Stimulation loop ended.")
            # Ensure all channels are marked as stopped
            for channel in self.channels:
                channel.stop_stimulation() # This will set _is_stimulating to False

    def _start_visualization(self):
        """Start real-time waveform visualization."""
        if not self.visualization_enabled:
            return
        
        if not self._visualization_active:
            self.fig, self.axes = plt.subplots(self.n_channels, 1, figsize=(10, 2 * self.n_channels), sharex=True)
            
            if self.n_channels == 1:
                self.axes = [self.axes]
            
            self.lines = []
            self.param_texts = []
            for i, ax in enumerate(self.axes):
                line, = ax.plot([], [], 'b-', lw=2)
                self.lines.append(line)
                ax.set_ylabel(f'Ch {i+1}\n(mA)')
                ax.set_xlim(0, 0.1) # 100ms window
                ax.set_ylim(-12, 12) # Max amplitude + buffer
                ax.grid(True)
                param_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                     fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
                self.param_texts.append(param_text)
            
            self.axes[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            
            self._visualization_active = True
            self._visualization_thread = threading.Thread(target=self._update_visualization_loop)
            self._visualization_thread.daemon = True
            self._visualization_thread.start()
            self.logger.info("Visualization thread started.")

    def _update_visualization_loop(self):
        """Updates the visualization plot in a separate thread."""
        self.logger.info("Visualization update loop started.")
        plt.ion() # Turn on interactive mode
        self.fig.show() # Show the figure

        while self._visualization_active:
            try:
                for i, channel in enumerate(self.channels):
                    if channel.is_stimulating:
                        # Access waveform data safely
                        waveform = channel.current_waveform
                        time_vector = channel.waveform_time
                        
                        self.lines[i].set_data(time_vector, waveform)
                        
                        # Update parameters text on plot
                        params = channel.get_parameters()
                        param_text_str = (
                            f"Amp: {params.get('amplitude_mA',0):.1f}mA, PW: {params.get('pulse_width_us',0)}us\n"
                            f"Freq: {params.get('frequency_hz',0)}Hz, Mode: {params.get('mode','N/A')}"
                        )
                        self.param_texts[i].set_text(param_text_str)
                    else:
                        self.lines[i].set_data([], []) # Clear plot if not stimulating
                        self.param_texts[i].set_text("Stopped")

                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                time.sleep(0.1) # Update every 100ms
            except Exception as e:
                self.logger.error(f"Visualization error: {e}", exc_info=True)
                self._visualization_active = False # Stop visualization on error
                break
        
        self.logger.info("Visualization update loop ended. Closing plot.")
        plt.close(self.fig) # Close the plot window when loop ends
        self.fig, self.axes, self.lines, self.param_texts = None, None, [], [] # Clear references

    def _emergency_stop(self):
        """Emergency stop of all stimulation channels."""
        self.logger.critical("EMERGENCY STOP: Halting all stimulation due to critical error!")
        
        # Signal main loop to stop
        self._running = False
        
        # Stop all channels
        for channel in self.channels:
            channel.stop_stimulation()
        
        # In a real system, would send immediate stop command to hardware
        if self.hardware_connected:
            # Placeholder for hardware emergency halt command
            self.logger.critical("Sending emergency halt command to hardware (simulated).")
            pass
        
        # Log the emergency stop
        self._log_event("emergency_stop", {
            "reason": "Critical error in stimulation loop or hardware communication"
        })
    
    def _log_event(self, event_type: str, data: dict):
        """
        Log stimulation events using the centralized logging system.
        
        Parameters:
        -----------
        event_type : str
            Type of event.
        data : dict
            Event data.
        """
        self.logger.info(f"EVENT - {event_type}: {data}")
    
    def get_status(self, channel_id: int | None = None) -> dict | list | None:
        """
        Get stimulation status.
        
        Parameters:
        -----------
        channel_id : int or None
            Channel ID (None for all).
            
        Returns:
        --------
        status : dict or list
            Stimulation status.
        """
        if channel_id is not None:
            if not (0 <= channel_id < self.n_channels):
                self.logger.warning(f"Invalid channel ID: {channel_id}")
                return None
            
            channel = self.channels[channel_id]
            current_time = time.time()
            duration = (current_time - channel._start_time) if channel._start_time else 0.0
            
            return {
                "channel_id": channel_id,
                "is_stimulating": channel.is_stimulating,
                "parameters": channel.get_parameters(),
                "start_time": channel._start_time,
                "duration_s": duration
            }
        else:
            # Return status for all channels
            status = []
            for i, channel in enumerate(self.channels):
                current_time = time.time()
                duration = (current_time - channel._start_time) if channel._start_time else 0.0
                status.append({
                    "channel_id": i,
                    "is_stimulating": channel.is_stimulating,
                    "parameters": channel.get_parameters(),
                    "start_time": channel._start_time,
                    "duration_s": duration
                })
            return status
    
    def save_protocol(self, filepath: str, channel_id: int | None = None) -> bool:
        """
        Save stimulation protocol to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the protocol.
        channel_id : int or None
            Channel ID to save (None for all).
            
        Returns:
        --------
        success : bool
            Whether the protocol was saved successfully.
        """
        try:
            protocol = {
                "timestamp": datetime.now().isoformat(),
                "device_id": self.device_id,
                "channels": []
            }
            
            channels_to_save = []
            if channel_id is not None:
                if not (0 <= channel_id < self.n_channels):
                    self.logger.warning(f"Invalid channel ID: {channel_id}")
                    return False
                channels_to_save.append(self.channels[channel_id])
            else:
                channels_to_save = self.channels
            
            for channel in channels_to_save:
                protocol["channels"].append({
                    "channel_id": channel.channel_id,
                    "parameters": channel.get_parameters() # get_parameters returns Enum values as strings
                })
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                    json.dump(protocol, f, indent=2)
            
            self.logger.info(f"Protocol saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving protocol to {filepath}: {e}", exc_info=True)
            return False
    
    def load_protocol(self, filepath: str, channel_id: int | None = None) -> bool:
        """
        Load stimulation protocol from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the protocol from.
        channel_id : int or None
            Channel ID to load (None for all).
            
        Returns:
        --------
        success : bool
            Whether the protocol was loaded successfully.
        """
        try:
            with open(filepath, 'r') as f:
                protocol = json.load(f)
            
            for ch_data in protocol.get("channels", []):
                ch_id = ch_data["channel_id"]
                
                if channel_id is not None and ch_id != channel_id:
                    continue
                
                if not (0 <= ch_id < self.n_channels):
                    self.logger.warning(f"Protocol contains invalid channel ID: {ch_id}. Skipping.")
                    continue

                params_to_load = ch_data["parameters"]
                if "mode" in params_to_load and isinstance(params_to_load["mode"], str):
                    try:
                        params_to_load["mode"] = StimulationMode[params_to_load["mode"]] # 문자열을 Enum으로 변환
                    except KeyError:
                        self.logger.warning(f"Unknown stimulation mode '{params_to_load['mode']}' in protocol for channel {ch_id}. Skipping mode setting.")
                        del params_to_load["mode"] # Remove invalid mode

                if not self.set_parameters(ch_id, params_to_load):
                    self.logger.error(f"Failed to apply parameters from protocol to channel {ch_id}.")
                    return False # Indicate failure if any channel fails to load
            
            self.logger.info(f"Protocol loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            self.logger.error(f"Protocol file not found: {filepath}")
            return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from protocol file {filepath}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading protocol from {filepath}: {e}", exc_info=True)
            return False
    
    def start_stimulation(self, channel_id: int | list[int] | None = None, duration: float | None = None) -> bool:
        """
        Start stimulation on specified channels.
        
        Parameters:
        -----------
        channel_id : int or list or None
            Channel ID(s) to start stimulation on (None for all).
        duration : float or None
            Duration in seconds (None for indefinite).
            
        Returns:
        --------
        success : bool
            Whether stimulation was started successfully.
        """
        channels_to_start = []
        if channel_id is None:
            channels_to_start = list(range(self.n_channels))
        elif isinstance(channel_id, (list, tuple)):
            channels_to_start = channel_id
        else:
            channels_to_start = [channel_id]
        
        valid_channels_started = []
        for ch_id in channels_to_start:
            if not (0 <= ch_id < self.n_channels):
                self.logger.warning(f"Invalid channel ID: {ch_id}. Skipping start.")
                continue
            if self.channels[ch_id].start_stimulation(): # Attempt to start individual channel
                valid_channels_started.append(ch_id)
        
        if not valid_channels_started:
            self.logger.warning("No valid channels to start stimulation or all already running.")
            return False
        
        # Start the main stimulation thread if not already running
        if not self._running:
            self._running = True
            self._stimulation_thread = threading.Thread(target=self._stimulation_loop, args=(duration,))
            self._stimulation_thread.daemon = True # Allow main program to exit even if thread is running
            self._stimulation_thread.start()
            self.logger.info(f"Main stimulation thread started for channels: {valid_channels_started}")
            
            # Start visualization if enabled and not already running
            if self.visualization_enabled and not self._visualization_active:
                self._start_visualization()
        else:
            self.logger.info(f"Main stimulation thread already running. Channels {valid_channels_started} activated.")
        
        self._log_event("start_stimulation", {
            "channels": valid_channels_started,
            "duration_s": duration
        })
        
        return True
    
    def stop_stimulation(self, channel_id: int | list[int] | None = None) -> bool:
        """
        Stop stimulation on specified channels.
        
        Parameters:
        -----------
        channel_id : int or list or None
            Channel ID(s) to stop stimulation on (None for all).
            
        Returns:
        --------
        success : bool
            Whether stimulation was stopped successfully.
        """
        channels_to_stop = []
        if channel_id is None:
            channels_to_stop = list(range(self.n_channels))
        elif isinstance(channel_id, (list, tuple)):
            channels_to_stop = channel_id
        else:
            channels_to_stop = [channel_id]
        
        stopped_any = False
        for ch_id in channels_to_stop:
            if not (0 <= ch_id < self.n_channels):
                self.logger.warning(f"Invalid channel ID: {ch_id}. Skipping stop.")
                continue
            if self.channels[ch_id].stop_stimulation(): # Attempt to stop individual channel
                stopped_any = True
        
        if not stopped_any:
            self.logger.info("No channels were actively stimulating or valid channels provided to stop.")
            return False

        # Check if any channels are still stimulating
        any_channel_still_stimulating = any(channel.is_stimulating for channel in self.channels)
        
        # If no channels are stimulating, stop the main stimulation thread
        if not any_channel_still_stimulating:
            self.logger.info("All channels stopped. Signalling main stimulation thread to stop.")
            self._running = False # Signal the _stimulation_loop to terminate
            if self._stimulation_thread and self._stimulation_thread.is_alive():
                self._stimulation_thread.join(timeout=2.0) # Wait for the thread to finish
                if self._stimulation_thread.is_alive():
                    self.logger.warning("Stimulation thread did not terminate gracefully within timeout.")
            self._stimulation_thread = None # Clear thread reference

            # Stop visualization
            if self._visualization_active:
                self._visualization_active = False # Signal visualization thread to terminate
                if self._visualization_thread and self._visualization_thread.is_alive():
                    self._visualization_thread.join(timeout=1.0)
                    if self._visualization_thread.is_alive():
                        self.logger.warning("Visualization thread did not terminate gracefully within timeout.")
                self._visualization_thread = None
        
        self._log_event("stop_stimulation", {
            "channels": list(channels_to_stop)
        })
        
        return True

    def close(self):
        """Clean up resources and close the module."""
        self.logger.info("Closing Stimulation Module...")
        # Stop all stimulation
        self.stop_stimulation(channel_id=None) # Stop all channels and main thread

        # Wait for threads to fully terminate
        if self._stimulation_thread and self._stimulation_thread.is_alive():
            self._stimulation_thread.join(timeout=2.0)
        if self._visualization_thread and self._visualization_thread.is_alive():
            self._visualization_thread.join(timeout=1.0)
        
        # Disconnect from hardware
        if self.hardware_connected:
            # Placeholder for hardware disconnect
            self.logger.info("Disconnecting from hardware (simulated).")
            pass
        
        self.logger.info("Stimulation module closed.")


# Example usage (for direct testing of this module)
if __name__ == "__main__":
    # Create stimulation module
    stim = StimulationModule(n_channels=2, hardware_connected=False, visualization_enabled=True)
    
    # Create a directory for protocols if it doesn't exist
    os.makedirs("stimulation_protocols", exist_ok=True)

    # --- Demo 1: Apply a neural regeneration pattern and start stimulation ---
    logger.info("\n--- Demo 1: Apply Regeneration Pattern (Channel 0) & Start ---")
    if stim.apply_regeneration_pattern(0, NeuralRegenerationPattern.EARLY_STAGE):
        logger.info("Early Stage pattern applied to Channel 0.")
    else:
        logger.error("Failed to apply pattern to Channel 0.")
        
    print("\nStarting stimulation on Channel 0 for 10 seconds...")
    stim.start_stimulation(channel_id=0, duration=10)
    time.sleep(3) # Let it run for a bit

    # --- Demo 2: Apply a different pattern to another channel and start ---
    logger.info("\n--- Demo 2: Apply Different Pattern (Channel 1) & Start ---")
    if stim.apply_regeneration_pattern(1, NeuralRegenerationPattern.MOTOR_NERVE):
        logger.info("Motor Nerve pattern applied to Channel 1.")
    else:
        logger.error("Failed to apply pattern to Channel 1.")

    print("\nStarting stimulation on Channel 1 indefinitely...")
    stim.start_stimulation(channel_id=1)
    time.sleep(3) # Let it run for a bit

    # --- Demo 3: Change parameters mid-stimulation (Channel 1) ---
    logger.info("\n--- Demo 3: Change Parameters Mid-Stimulation (Channel 1) ---")
    print("\nChanging Channel 1 frequency to 150Hz and amplitude to 3.0mA...")
    if stim.set_parameters(1, {"frequency_hz": 150, "amplitude_mA": 3.0}):
        logger.info("Channel 1 parameters updated successfully.")
    else:
        logger.error("Failed to update Channel 1 parameters.")
    time.sleep(3)

    # --- Demo 4: Safety Limit Violation Test ---
    logger.info("\n--- Demo 4: Safety Limit Violation Test (Channel 0) ---")
    print("\nAttempting to set unsafe amplitude for Channel 0 (should fail)...")
    if not stim.set_parameters(0, {"amplitude_mA": 15.0}): # Max is 10.0 mA
        logger.info("Successfully blocked unsafe amplitude setting for Channel 0.")
    else:
        logger.error("UNSAFE: Unsafe amplitude was set for Channel 0!")
    time.sleep(1)

    print("\nAttempting to set unsafe charge density for Channel 0 (should fail)...")
    # This combination (10mA * 6000us) results in 0.06 mC, which is within the default 0.5mC limit.
    # To make it unsafe, let's try a value that exceeds 0.5mC, e.g., 100mA * 6000us = 0.6 mC
    if not stim.set_parameters(0, {"amplitude_mA": 100.0, "pulse_width_us": 6000.0}):
        logger.info("Successfully blocked unsafe charge density for Channel 0.")
    else:
        logger.error("UNSAFE: Unsafe charge density was set for Channel 0!")
    time.sleep(1)

    # --- Demo 5: Stop specific channel ---
    logger.info("\n--- Demo 5: Stop Specific Channel (Channel 1) ---")
    print("\nStopping stimulation on Channel 1...")
    stim.stop_stimulation(channel_id=1)
    time.sleep(2)

    # --- Demo 6: Save and Load Protocol ---
    logger.info("\n--- Demo 6: Save and Load Protocol ---")
    protocol_filepath = "stimulation_protocols/my_test_protocol.json"
    print(f"\nSaving current protocol to {protocol_filepath}...")
    if stim.save_protocol(protocol_filepath):
        logger.info("Protocol saved.")
    else:
        logger.error("Failed to save protocol.")
    
    # Change Channel 0 parameters to something else for testing load
    stim.set_parameters(0, {"amplitude_mA": 0.1, "frequency_hz": 1})
    print("\nChannel 0 parameters temporarily changed for load test.")
    time.sleep(1)

    print(f"\nLoading protocol from {protocol_filepath} to Channel 0...")
    if stim.load_protocol(filepath=protocol_filepath, channel_id=0):
        logger.info("Protocol loaded to Channel 0.")
        loaded_params = stim.get_parameters(0)
        logger.info(f"Channel 0 parameters after load: {loaded_params}")
    else:
        logger.error("Failed to load protocol to Channel 0.")
    time.sleep(1)

    # --- Demo 7: Get Status ---
    logger.info("\n--- Demo 7: Get Status ---")
    status_all = stim.get_status()
    print("\nCurrent system status:")
    for ch_status in status_all:
        print(f"  Channel {ch_status['channel_id']}: Stimulating={ch_status['is_stimulating']}, "
              f"Amp={ch_status['parameters']['amplitude_mA']:.1f}mA, Freq={ch_status['parameters']['frequency_hz']}Hz")
    time.sleep(1)

    # --- Final Cleanup ---
    logger.info("\n--- Final Cleanup ---")
    print("\nClosing stimulation module and stopping all remaining stimulation...")
    stim.close()
    print("Demonstration complete.")

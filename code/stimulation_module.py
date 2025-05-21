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
from matplotlib.animation import FuncAnimation
from datetime import datetime
from enum import Enum
import logging
import pickle
import scipy.signal as signal


class StimulationMode(Enum):
    """Stimulation mode enumeration"""
    BIPHASIC = 0
    MONOPHASIC_CATHODIC = 1
    MONOPHASIC_ANODIC = 2
    BURST = 3
    HIGH_FREQUENCY = 4
    THETA_BURST = 5
    LONG_DURATION_LOW_FREQUENCY = 6
    PAIRED_ASSOCIATIVE = 7
    RANDOMIZED = 8


class NeuralRegenerationPattern(Enum):
    """Neural regeneration stimulation patterns based on research"""
    EARLY_STAGE = 0    # For early post-injury (first week): promote neuroprotection
    MID_STAGE = 1      # For weeks 1-4: promote axon sprouting and growth
    LATE_STAGE = 2     # For weeks 4+: promote synaptic refinement
    ACUTE_INJURY = 3   # For immediate post-injury application
    CHRONIC_INJURY = 4 # For long-term injuries (months to years)
    MOTOR_NERVE = 5    # Patterns optimized for motor nerve regeneration
    SENSORY_NERVE = 6  # Patterns optimized for sensory nerve regeneration
    BDNF_ENHANCING = 7 # Patterns shown to enhance BDNF expression
    GDNF_ENHANCING = 8 # Patterns shown to enhance GDNF expression


class StimulationChannel:
    """Represents a stimulation channel with its parameters"""
    
    def __init__(self, channel_id, active=True):
        """
        Initialize a stimulation channel
        
        Parameters:
        -----------
        channel_id : int
            Channel identifier
        active : bool
            Whether the channel is active
        """
        self.channel_id = channel_id
        self.active = active
        
        # Default parameters
        self.amplitude = 1.0       # mA
        self.pulse_width = 200     # μs
        self.frequency = 50        # Hz
        self.duty_cycle = 100      # %
        self.mode = StimulationMode.BIPHASIC
        self.burst_count = 3
        self.burst_frequency = 5   # Hz (for burst mode)
        self.interphase_gap = 100  # μs (for biphasic pulses)
        
        # Advanced parameters for neural regeneration
        self.ramp_up_time = 0.5    # Seconds for amplitude to reach target
        self.ramp_down_time = 0.5  # Seconds for amplitude to drop to zero
        self.phase_asymmetry = 1.0 # Ratio between positive and negative phases
        self.pulse_train_interval = 1.0  # Seconds between pulse trains
        self.random_frequency_range = (0, 0)  # For randomized stimulation
        self.random_amplitude_range = (0, 0)  # For randomized stimulation
        
        # Status
        self.is_stimulating = False
        self.start_time = None
        self.stop_time = None
        
        # Waveform generation
        self.current_waveform = None
        self.waveform_time = None
    
    def set_parameters(self, params):
        """
        Set stimulation parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameter values
        """
        # Basic parameters
        if 'amplitude' in params:
            self.amplitude = params['amplitude']
        if 'pulse_width' in params:
            self.pulse_width = params['pulse_width']
        if 'frequency' in params:
            self.frequency = params['frequency']
        if 'duty_cycle' in params:
            self.duty_cycle = params['duty_cycle']
        if 'mode' in params:
            self.mode = params['mode']
        if 'burst_count' in params:
            self.burst_count = params['burst_count']
        if 'burst_frequency' in params:
            self.burst_frequency = params['burst_frequency']
        if 'interphase_gap' in params:
            self.interphase_gap = params['interphase_gap']
            
        # Advanced parameters
        if 'ramp_up_time' in params:
            self.ramp_up_time = params['ramp_up_time']
        if 'ramp_down_time' in params:
            self.ramp_down_time = params['ramp_down_time']
        if 'phase_asymmetry' in params:
            self.phase_asymmetry = params['phase_asymmetry']
        if 'pulse_train_interval' in params:
            self.pulse_train_interval = params['pulse_train_interval']
        if 'random_frequency_range' in params:
            self.random_frequency_range = params['random_frequency_range']
        if 'random_amplitude_range' in params:
            self.random_amplitude_range = params['random_amplitude_range']
    
    def get_parameters(self):
        """
        Get current stimulation parameters
        
        Returns:
        --------
        params : dict
            Dictionary of parameter values
        """
        return {
            # Basic parameters
            'amplitude': self.amplitude,
            'pulse_width': self.pulse_width,
            'frequency': self.frequency,
            'duty_cycle': self.duty_cycle,
            'mode': self.mode,
            'burst_count': self.burst_count,
            'burst_frequency': self.burst_frequency,
            'interphase_gap': self.interphase_gap,
            
            # Advanced parameters
            'ramp_up_time': self.ramp_up_time,
            'ramp_down_time': self.ramp_down_time,
            'phase_asymmetry': self.phase_asymmetry,
            'pulse_train_interval': self.pulse_train_interval,
            'random_frequency_range': self.random_frequency_range,
            'random_amplitude_range': self.random_amplitude_range
        }
    
    def start_stimulation(self):
        """Start stimulation on this channel"""
        if not self.active:
            return False
        
        self.is_stimulating = True
        self.start_time = time.time()
        self.stop_time = None
        return True
    
    def stop_stimulation(self):
        """Stop stimulation on this channel"""
        if not self.is_stimulating:
            return
        
        self.is_stimulating = False
        self.stop_time = time.time()


class StimulationModule:
    """Module for controlling neural electrical stimulation"""
    
    def __init__(self, n_channels=4, hardware_connected=False, device_id=None, visualization=False):
        """
        Initialize the stimulation module
        
        Parameters:
        -----------
        n_channels : int
            Number of stimulation channels
        hardware_connected : bool
            Whether hardware is connected
        device_id : str
            Identifier for the stimulation device
        visualization : bool
            Enable real-time waveform visualization
        """
        self.n_channels = n_channels
        self.hardware_connected = hardware_connected
        self.device_id = device_id
        self.visualization = visualization
        
        # Create channels
        self.channels = [StimulationChannel(i) for i in range(n_channels)]
        
        # Initialize the stimulation thread
        self.stimulation_thread = None
        self.running = False
        
        # For visualization
        self.visualization_thread = None
        self.fig = None
        self.axes = None
        
        # Parameter constraints for safety
        self.safety_limits = {
            'amplitude': (0.1, 10.0),     # mA
            'pulse_width': (50, 1000),    # μs
            'frequency': (1, 500),        # Hz
            'duty_cycle': (1, 100),       # %
            'interphase_gap': (0, 1000),  # μs
            'phase_asymmetry': (0.1, 10.0)  # Ratio
        }
        
        # Set up logging
        self.setup_logging()
        
        # Neural regeneration pattern templates based on research
        self.pattern_templates = self._initialize_regeneration_patterns()
        
        print(f"Stimulation module initialized with {n_channels} channels")
        if hardware_connected:
            print(f"Hardware connected: Device ID {device_id}")
            self._initialize_hardware()
        else:
            print("No hardware connected, using simulation mode")
    
    def _stimulation_loop(self, duration=None):
        """
        Main stimulation loop
        
        Parameters:
        -----------
        duration : float or None
            Stimulation duration in seconds (None for indefinite)
        """
        start_time = time.time()
        end_time = start_time + duration if duration is not None else None
        
        self.logger.info(f"Stimulation started" + (f" for {duration} seconds" if duration else ""))
        
        try:
            while self.running:
                # Check duration limit
                if end_time and time.time() >= end_time:
                    self.logger.info("Stimulation duration reached, stopping")
                    self.stop_stimulation()
                    break
                
                # Generate stimulation waveform for each active channel
                for channel in self.channels:
                    if not channel.is_stimulating:
                        continue
                    
                    # Generate and apply waveform
                    waveform, waveform_time = self._generate_waveform(channel)
                    
                    # Store for visualization
                    channel.current_waveform = waveform
                    channel.waveform_time = waveform_time
                
                # Sleep a short time to avoid CPU overload
                time.sleep(0.01)
        
        except Exception as e:
            self.logger.error(f"Error in stimulation loop: {e}")
            self.running = False
            
            # Emergency stop in case of error
            if self.hardware_connected:
                self._emergency_stop()
        
        self.logger.info("Stimulation loop ended")
    
    def _generate_waveform(self, channel):
        """
        Generate stimulation waveform based on channel parameters
        
        Parameters:
        -----------
        channel : StimulationChannel
            The channel to generate waveform for
        
        Returns:
        --------
        waveform : ndarray
            Generated current waveform (mA)
        time_vector : ndarray
            Time vector for the waveform (s)
        """
        # Calculate timing parameters
        sampling_rate = 100000  # Hz (100 kHz for accurate pulse shapes)
        
        # Duration for one pulse cycle
        if channel.mode == StimulationMode.BURST:
            # For burst mode, generate a burst of pulses
            cycle_duration = 1.0 / channel.burst_frequency  # seconds
        else:
            # For other modes, base on main frequency
            cycle_duration = 1.0 / channel.frequency  # seconds
        
        # Create time vector
        n_samples = int(cycle_duration * sampling_rate)
        time_vector = np.linspace(0, cycle_duration, n_samples)
        
        # Initialize waveform
        waveform = np.zeros(n_samples)
        
        # Generate waveform based on mode
        if channel.mode == StimulationMode.BIPHASIC:
            waveform = self._generate_biphasic_waveform(
                time_vector, channel.frequency, channel.amplitude,
                channel.pulse_width, channel.interphase_gap, channel.phase_asymmetry
            )
        
        elif channel.mode == StimulationMode.MONOPHASIC_CATHODIC:
            waveform = self._generate_monophasic_waveform(
                time_vector, channel.frequency, -channel.amplitude,
                channel.pulse_width
            )
        
        elif channel.mode == StimulationMode.MONOPHASIC_ANODIC:
            waveform = self._generate_monophasic_waveform(
                time_vector, channel.frequency, channel.amplitude,
                channel.pulse_width
            )
        
        elif channel.mode == StimulationMode.BURST:
            waveform = self._generate_burst_waveform(
                time_vector, channel.frequency, channel.amplitude,
                channel.pulse_width, channel.interphase_gap, 
                channel.burst_count
            )
        
        elif channel.mode == StimulationMode.HIGH_FREQUENCY:
            waveform = self._generate_biphasic_waveform(
                time_vector, channel.frequency, channel.amplitude,
                channel.pulse_width, channel.interphase_gap, channel.phase_asymmetry
            )
        
        elif channel.mode == StimulationMode.THETA_BURST:
            waveform = self._generate_theta_burst_waveform(
                time_vector, channel.frequency, channel.amplitude,
                channel.pulse_width, channel.interphase_gap, 
                channel.burst_count
            )
        
        elif channel.mode == StimulationMode.RANDOMIZED:
            # Randomized parameters within specified ranges
            if channel.random_frequency_range[1] > 0:
                rand_freq = np.random.uniform(
                    channel.random_frequency_range[0], 
                    channel.random_frequency_range[1]
                )
            else:
                rand_freq = channel.frequency
                
            if channel.random_amplitude_range[1] > 0:
                rand_amp = np.random.uniform(
                    channel.random_amplitude_range[0], 
                    channel.random_amplitude_range[1]
                )
            else:
                rand_amp = channel.amplitude
                
            waveform = self._generate_biphasic_waveform(
                time_vector, rand_freq, rand_amp,
                channel.pulse_width, channel.interphase_gap, channel.phase_asymmetry
            )
        
        # Apply duty cycle if less than 100%
        if channel.duty_cycle < 100:
            # Create duty cycle mask
            duty_cycle_period = 1.0  # 1 second
            n_duty_samples = int(duty_cycle_period * sampling_rate)
            
            # If duty cycle period is longer than the current waveform, extend waveform
            if n_duty_samples > n_samples:
                repetitions = int(np.ceil(n_duty_samples / n_samples))
                waveform = np.tile(waveform, repetitions)
                time_vector = np.linspace(0, cycle_duration * repetitions, len(waveform))
                
                # Trim to exact size
                waveform = waveform[:n_duty_samples]
                time_vector = time_vector[:n_duty_samples]
            
            # Calculate active samples
            active_samples = int(n_duty_samples * channel.duty_cycle / 100)
            
            # Create mask
            mask = np.zeros(len(waveform))
            mask[:active_samples] = 1.0
            
            # Apply mask
            waveform = waveform * mask
        
        # Apply ramp up/down if needed (if stimulation just started or is about to stop)
        if channel.ramp_up_time > 0 and channel.start_time:
            elapsed = time.time() - channel.start_time
            if elapsed < channel.ramp_up_time:
                # Still in ramp-up phase
                ramp_factor = elapsed / channel.ramp_up_time
                waveform = waveform * ramp_factor
        
        if channel.ramp_down_time > 0 and channel.stop_time:
            remaining = time.time() - channel.stop_time
            if remaining < channel.ramp_down_time:
                # In ramp-down phase
                ramp_factor = 1.0 - (remaining / channel.ramp_down_time)
                waveform = waveform * ramp_factor
        
        # In a real implementation, this would be sent to the hardware
        # For simulation, we return the waveform
        
        return waveform, time_vector
    
    def _generate_biphasic_waveform(self, time_vector, frequency, amplitude, pulse_width, interphase_gap, phase_asymmetry=1.0):
        """Generate biphasic stimulation waveform"""
        period = 1.0 / frequency
        samples_per_period = int(len(time_vector) * period / (time_vector[-1] - time_vector[0]))
        
        # Convert microseconds to seconds
        pulse_width_sec = pulse_width * 1e-6
        interphase_gap_sec = interphase_gap * 1e-6
        
        # Number of samples for each phase
        samples_per_pulse = int(pulse_width_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        samples_per_gap = int(interphase_gap_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        
        # Initialize waveform with zeros
        waveform = np.zeros(len(time_vector))
        
        # Generate one period of the waveform
        for i in range(int(len(time_vector) / samples_per_period)):
            start_idx = i * samples_per_period
            
            # First (cathodic) phase
            if start_idx + samples_per_pulse < len(waveform):
                waveform[start_idx:start_idx + samples_per_pulse] = -amplitude
            
            # Interphase gap
            gap_start = start_idx + samples_per_pulse
            if gap_start + samples_per_gap < len(waveform):
                # Gap remains at zero
                pass
            
            # Second (anodic) phase
            second_phase_start = gap_start + samples_per_gap
            second_phase_amplitude = amplitude / phase_asymmetry  # Adjust amplitude based on asymmetry
            second_phase_samples = int(samples_per_pulse * phase_asymmetry)  # Adjust duration based on asymmetry
            
            if second_phase_start + second_phase_samples < len(waveform):
                waveform[second_phase_start:second_phase_start + second_phase_samples] = second_phase_amplitude
        
        return waveform
    
    def _generate_monophasic_waveform(self, time_vector, frequency, amplitude, pulse_width):
        """Generate monophasic stimulation waveform"""
        period = 1.0 / frequency
        samples_per_period = int(len(time_vector) * period / (time_vector[-1] - time_vector[0]))
        
        # Convert microseconds to seconds
        pulse_width_sec = pulse_width * 1e-6
        
        # Number of samples for pulse
        samples_per_pulse = int(pulse_width_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        
        # Initialize waveform with zeros
        waveform = np.zeros(len(time_vector))
        
        # Generate one period of the waveform
        for i in range(int(len(time_vector) / samples_per_period)):
            start_idx = i * samples_per_period
            if start_idx + samples_per_pulse < len(waveform):
                waveform[start_idx:start_idx + samples_per_pulse] = amplitude
        
        return waveform
    
    def _generate_burst_waveform(self, time_vector, frequency, amplitude, pulse_width, interphase_gap, burst_count):
        """Generate burst stimulation waveform"""
        # Calculate timing parameters
        burst_period = 1.0 / frequency  # Period between pulses within a burst
        
        # Convert microseconds to seconds
        pulse_width_sec = pulse_width * 1e-6
        interphase_gap_sec = interphase_gap * 1e-6
        
        # Calculate samples
        samples_per_pulse_period = int(burst_period * len(time_vector) / (time_vector[-1] - time_vector[0]))
        samples_per_pulse = int(pulse_width_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        samples_per_gap = int(interphase_gap_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        
        # Initialize waveform with zeros
        waveform = np.zeros(len(time_vector))
        
        # Generate biphasic pulses within the burst
        for i in range(burst_count):
            start_idx = i * samples_per_pulse_period
            
            # Ensure we don't exceed the waveform length
            if start_idx + samples_per_pulse >= len(waveform):
                break
            
            # First (cathodic) phase
            waveform[start_idx:start_idx + samples_per_pulse] = -amplitude
            
            # Interphase gap
            gap_start = start_idx + samples_per_pulse
            
            # Second (anodic) phase
            second_phase_start = gap_start + samples_per_gap
            if second_phase_start + samples_per_pulse < len(waveform):
                waveform[second_phase_start:second_phase_start + samples_per_pulse] = amplitude
        
        return waveform
    
    def _generate_theta_burst_waveform(self, time_vector, frequency, amplitude, pulse_width, interphase_gap, burst_count):
        """Generate theta burst stimulation waveform (TBS)"""
        # TBS parameters: 3 pulses at 50Hz, repeated at 5Hz theta rhythm
        theta_freq = 5  # Hz
        burst_freq = 50  # Hz
        
        # Use burst_count from parameters
        
        # Calculate timing
        theta_period = 1.0 / theta_freq  # 200ms between burst starts
        burst_period = 1.0 / burst_freq  # 20ms between pulses within a burst
        
        # Convert microseconds to seconds
        pulse_width_sec = pulse_width * 1e-6
        interphase_gap_sec = interphase_gap * 1e-6
        
        # Calculate samples
        samples_per_theta_period = int(theta_period * len(time_vector) / (time_vector[-1] - time_vector[0]))
        samples_per_burst_period = int(burst_period * len(time_vector) / (time_vector[-1] - time_vector[0]))
        samples_per_pulse = int(pulse_width_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        samples_per_gap = int(interphase_gap_sec * len(time_vector) / (time_vector[-1] - time_vector[0]))
        
        # Initialize waveform with zeros
        waveform = np.zeros(len(time_vector))
        
        # Create TBS pattern: one burst within the theta period
        for i in range(int(len(time_vector) / samples_per_theta_period)):
            theta_start = i * samples_per_theta_period
            
            # Generate pulses within the burst
            for j in range(burst_count):
                pulse_start = theta_start + j * samples_per_burst_period
                
                # Ensure we don't exceed the waveform length
                if pulse_start + samples_per_pulse >= len(waveform):
                    break
                
                # First (cathodic) phase
                waveform[pulse_start:pulse_start + samples_per_pulse] = -amplitude
                
                # Interphase gap
                gap_start = pulse_start + samples_per_pulse
                
                # Second (anodic) phase
                second_phase_start = gap_start + samples_per_gap
                if second_phase_start + samples_per_pulse < len(waveform):
                    waveform[second_phase_start:second_phase_start + samples_per_pulse] = amplitude
        
        return waveform
    
    def _start_visualization(self):
        """Start real-time waveform visualization"""
        if not self.visualization:
            return
        
        # Create figure and axes for visualization
        self.fig, self.axes = plt.subplots(self.n_channels, 1, figsize=(10, 2 * self.n_channels), sharex=True)
        
        # Ensure axes is a list even for single channel
        if self.n_channels == 1:
            self.axes = [self.axes]
        
        # Set up each axis
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(f'Channel {i+1}\n(mA)')
            ax.set_xlim(0, 0.1)  # 100ms window
            ax.set_ylim(-5, 5)   # -5 to 5 mA
            ax.grid(True)
        
        self.axes[-1].set_xlabel('Time (s)')
        
        # Create empty lines
        self.lines = []
        for ax in self.axes:
            line, = ax.plot([], [], 'b-', lw=2)
            self.lines.append(line)
        
        # Initialize animation
        def init():
            for line in self.lines:
                line.set_data([], [])
            return self.lines
        
        # Update function for animation
        def update(frame):
            for i, (line, channel) in enumerate(zip(self.lines, self.channels)):
                if channel.is_stimulating and channel.current_waveform is not None:
                    line.set_data(channel.waveform_time, channel.current_waveform)
                    self.axes[i].set_title(f'Channel {i+1}: {channel.mode.name}, {channel.frequency} Hz, {channel.amplitude} mA')
                else:
                    line.set_data([], [])
                    self.axes[i].set_title(f'Channel {i+1}: Inactive')
            return self.lines
        
        # Create animation
        self.anim = FuncAnimation(self.fig, update, init_func=init, interval=100, blit=True)
        
        # Display the plot
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
    
    def _emergency_stop(self):
        """Emergency stop of all stimulation channels"""
        self.logger.warning("EMERGENCY STOP: Halting all stimulation")
        
        # Stop all channels
        for channel in self.channels:
            channel.stop_stimulation()
        
        # In a real system, would send immediate stop command to hardware
        if self.hardware_connected:
            # Emergency halt command (placeholder)
            pass
        
        # Log the emergency stop
        self._log_event("emergency_stop", {
            "reason": "Error in stimulation loop"
        })
    
    def _log_event(self, event_type, data):
        """
        Log stimulation events
        
        Parameters:
        -----------
        event_type : str
            Type of event
        data : dict
            Event data
        """
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        self.log_data.append(log_entry)
        
        # Periodically write to log file
        if len(self.log_data) >= 10:
            self._write_log()
    
    def _write_log(self):
        """Write log data to file"""
        if not self.log_data:
            return
        
        try:
            with open(self.log_file, 'a') as f:
                for entry in self.log_data:
                    f.write(json.dumps(entry) + '\n')
            
            # Clear the log data
            self.log_data = []
        except Exception as e:
            self.logger.error(f"Error writing to log file: {e}")
    
    def get_status(self, channel_id=None):
        """
        Get stimulation status
        
        Parameters:
        -----------
        channel_id : int or None
            Channel ID (None for all)
            
        Returns:
        --------
        status : dict or list
            Stimulation status
        """
        if channel_id is not None:
            if channel_id < 0 or channel_id >= self.n_channels:
                self.logger.warning(f"Invalid channel ID: {channel_id}")
                return None
            
            channel = self.channels[channel_id]
            return {
                "channel_id": channel_id,
                "active": channel.active,
                "is_stimulating": channel.is_stimulating,
                "parameters": channel.get_parameters(),
                "start_time": channel.start_time,
                "duration": time.time() - channel.start_time if channel.start_time else None
            }
        else:
            # Return status for all channels
            status = []
            for i, channel in enumerate(self.channels):
                status.append({
                    "channel_id": i,
                    "active": channel.active,
                    "is_stimulating": channel.is_stimulating,
                    "parameters": channel.get_parameters(),
                    "start_time": channel.start_time,
                    "duration": time.time() - channel.start_time if channel.start_time else None
                })
            return status
    
    def save_protocol(self, filepath, channel_id=None):
        """
        Save stimulation protocol to file
        
        Parameters:
        -----------
        filepath : str
            Path to save the protocol
        channel_id : int or None
            Channel ID to save (None for all)
            
        Returns:
        --------
        success : bool
            Whether the protocol was saved successfully
        """
        try:
            # Create protocol dictionary
            protocol = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "device_id": self.device_id,
                "channels": []
            }
            
            # Add channel data
            if channel_id is not None:
                if channel_id < 0 or channel_id >= self.n_channels:
                    self.logger.warning(f"Invalid channel ID: {channel_id}")
                    return False
                
                protocol["channels"].append({
                    "channel_id": channel_id,
                    "parameters": self.channels[channel_id].get_parameters()
                })
            else:
                # Add all channels
                for i, channel in enumerate(self.channels):
                    protocol["channels"].append({
                        "channel_id": i,
                        "parameters": channel.get_parameters()
                    })
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save protocol to file
            with open(filepath, 'w') as f:
                json.dump(protocol, f, indent=2)
            
            self.logger.info(f"Protocol saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving protocol: {e}")
            return False
    
    def load_protocol(self, filepath, channel_id=None):
        """
        Load stimulation protocol from file
        
        Parameters:
        -----------
        filepath : str
            Path to load the protocol from
        channel_id : int or None
            Channel ID to load (None for all)
            
        Returns:
        --------
        success : bool
            Whether the protocol was loaded successfully
        """
        try:
            # Load protocol from file
            with open(filepath, 'r') as f:
                protocol = json.load(f)
            
            # Apply protocol
            for ch_data in protocol["channels"]:
                ch_id = ch_data["channel_id"]
                
                # Check if we should apply to this channel
                if channel_id is not None and ch_id != channel_id:
                    continue
                
                # Apply parameters
                if ch_id < self.n_channels:
                    self.set_parameters(ch_id, ch_data["parameters"])
            
            self.logger.info(f"Protocol loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading protocol: {e}")
            return False
    
    def close(self):
        """Clean up resources and close the module"""
        # Stop all stimulation
        self.stop_stimulation()
        
        # Wait for stimulation thread to end
        if self.stimulation_thread and self.stimulation_thread.is_alive():
            self.stimulation_thread.join(timeout=1.0)
        
        # Disconnect from hardware
        if self.hardware_connected:
            # Placeholder for hardware disconnect
            pass
        
        # Write remaining log data
        self._write_log()
        
        self.logger.info("Stimulation module closed")


# Example usage
if __name__ == "__main__":
    # Create stimulation module
    stim = StimulationModule(n_channels=2, hardware_connected=False, visualization=True)
    
    # Apply a neural regeneration pattern
    stim.apply_regeneration_pattern(0, NeuralRegenerationPattern.EARLY_STAGE)
    stim.apply_regeneration_pattern(1, NeuralRegenerationPattern.MID_STAGE)
    
    # Start stimulation
    print("\nStarting stimulation on all channels for 10 seconds...")
    stim.start_stimulation(duration=10)
    
    # Wait for completion
    time.sleep(12)
    
    # Check status
    status = stim.get_status()
    print("\nStimulation status:")
    for channel_status in status:
        print(f"Channel {channel_status['channel_id']}: " + 
              f"{'Active' if channel_status['active'] else 'Inactive'}, " + 
              f"{'Stimulating' if channel_status['is_stimulating'] else 'Not stimulating'}")
    
    # Save protocol
    stim.save_protocol("models/stimulation_protocols/regeneration_protocol.json")
    
    # Close the module
    stim.close()

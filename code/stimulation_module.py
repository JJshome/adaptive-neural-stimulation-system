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
    
    def setup_logging(self):
        """Set up logging configuration"""
        self.log_path = "logs/stimulation"
        os.makedirs(self.log_path, exist_ok=True)
        
        # File for structured logs
        self.log_file = os.path.join(self.log_path, f"stim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.log_data = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_path, f"stim_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('stimulation_module')
    
    def _initialize_regeneration_patterns(self):
        """Initialize predefined neural regeneration stimulation patterns based on research"""
        patterns = {
            # Early stage post-injury: neuroprotection
            NeuralRegenerationPattern.EARLY_STAGE: {
                'frequency': 20,          # Hz - lower frequency for early stages
                'amplitude': 0.8,         # mA - moderate amplitude
                'pulse_width': 300,       # μs - wider pulses
                'duty_cycle': 50,         # % - intermittent stimulation
                'mode': StimulationMode.BIPHASIC,
                'interphase_gap': 100,    # μs
                'burst_count': 5,
                'burst_frequency': 5,     # Hz
                'ramp_up_time': 1.0,      # sec - gradual onset
                'ramp_down_time': 1.0,    # sec - gradual offset
                'phase_asymmetry': 1.0    # symmetric
            },
            
            # Mid-stage regeneration: axon sprouting and growth
            NeuralRegenerationPattern.MID_STAGE: {
                'frequency': 50,          # Hz - medium frequency
                'amplitude': 1.5,         # mA - stronger amplitude
                'pulse_width': 200,       # μs
                'duty_cycle': 70,         # % - more stimulation time
                'mode': StimulationMode.BURST,
                'interphase_gap': 80,     # μs
                'burst_count': 8,
                'burst_frequency': 8,     # Hz
                'ramp_up_time': 0.5,      # sec
                'ramp_down_time': 0.5,    # sec
                'phase_asymmetry': 1.2    # slight asymmetry
            },
            
            # Late-stage regeneration: synaptic refinement
            NeuralRegenerationPattern.LATE_STAGE: {
                'frequency': 100,         # Hz - higher frequency
                'amplitude': 1.2,         # mA
                'pulse_width': 150,       # μs - narrower pulses
                'duty_cycle': 80,         # % - more continuous
                'mode': StimulationMode.HIGH_FREQUENCY,
                'interphase_gap': 50,     # μs
                'burst_count': 3,
                'burst_frequency': 10,    # Hz
                'ramp_up_time': 0.3,      # sec
                'ramp_down_time': 0.3,    # sec
                'phase_asymmetry': 0.8    # asymmetric
            },
            
            # Acute injury pattern
            NeuralRegenerationPattern.ACUTE_INJURY: {
                'frequency': 10,          # Hz - very low frequency
                'amplitude': 0.5,         # mA - low amplitude to avoid excitotoxicity
                'pulse_width': 400,       # μs - wider pulses for better recruitment
                'duty_cycle': 30,         # % - minimal stimulation
                'mode': StimulationMode.BIPHASIC,
                'interphase_gap': 200,    # μs - longer gap
                'burst_count': 2,
                'burst_frequency': 2,     # Hz - sparse bursts
                'ramp_up_time': 2.0,      # sec - very gradual onset
                'ramp_down_time': 2.0,    # sec - very gradual offset
                'phase_asymmetry': 1.0    # symmetric
            },
            
            # Chronic injury pattern
            NeuralRegenerationPattern.CHRONIC_INJURY: {
                'frequency': 80,          # Hz - higher frequency
                'amplitude': 2.0,         # mA - higher amplitude
                'pulse_width': 250,       # μs
                'duty_cycle': 90,         # % - nearly continuous
                'mode': StimulationMode.THETA_BURST,
                'interphase_gap': 80,     # μs
                'burst_count': 10,
                'burst_frequency': 5,     # Hz
                'ramp_up_time': 0.2,      # sec - quick onset
                'ramp_down_time': 0.5,    # sec
                'phase_asymmetry': 1.5    # asymmetric
            },
            
            # Motor nerve pattern
            NeuralRegenerationPattern.MOTOR_NERVE: {
                'frequency': 60,          # Hz - good for motor recruitment
                'amplitude': 1.8,         # mA
                'pulse_width': 200,       # μs
                'duty_cycle': 60,         # %
                'mode': StimulationMode.BURST,
                'interphase_gap': 100,    # μs
                'burst_count': 6,
                'burst_frequency': 10,    # Hz
                'ramp_up_time': 0.3,      # sec
                'ramp_down_time': 0.3,    # sec
                'phase_asymmetry': 1.0    # symmetric
            },
            
            # Sensory nerve pattern
            NeuralRegenerationPattern.SENSORY_NERVE: {
                'frequency': 40,          # Hz - optimal for sensory fibers
                'amplitude': 1.0,         # mA - lower amplitude for sensory
                'pulse_width': 300,       # μs - wider for better recruitment
                'duty_cycle': 50,         # %
                'mode': StimulationMode.RANDOMIZED,
                'interphase_gap': 150,    # μs
                'burst_count': 4,
                'burst_frequency': 5,     # Hz
                'ramp_up_time': 0.5,      # sec
                'ramp_down_time': 0.5,    # sec
                'phase_asymmetry': 1.2,   # slight asymmetry
                'random_frequency_range': (30, 50),  # Hz - variation for better sensory response
                'random_amplitude_range': (0.8, 1.2)  # mA
            },
            
            # BDNF-enhancing pattern
            NeuralRegenerationPattern.BDNF_ENHANCING: {
                'frequency': 20,          # Hz - lower frequency shown to increase BDNF expression
                'amplitude': 1.5,         # mA
                'pulse_width': 250,       # μs
                'duty_cycle': 75,         # %
                'mode': StimulationMode.LONG_DURATION_LOW_FREQUENCY,
                'interphase_gap': 100,    # μs
                'burst_count': 0,         # No bursting
                'burst_frequency': 0,     # Hz
                'ramp_up_time': 1.0,      # sec
                'ramp_down_time': 1.0,    # sec
                'phase_asymmetry': 1.0,   # symmetric
                'pulse_train_interval': 5.0  # sec - long intervals between trains
            },
            
            # GDNF-enhancing pattern
            NeuralRegenerationPattern.GDNF_ENHANCING: {
                'frequency': 100,         # Hz - high frequency for GDNF
                'amplitude': 1.2,         # mA
                'pulse_width': 200,       # μs
                'duty_cycle': 40,         # % - intermittent
                'mode': StimulationMode.PAIRED_ASSOCIATIVE,
                'interphase_gap': 80,     # μs
                'burst_count': 10,
                'burst_frequency': 4,     # Hz - slow burst frequency
                'ramp_up_time': 0.3,      # sec
                'ramp_down_time': 0.3,    # sec
                'phase_asymmetry': 0.8,   # asymmetric
                'pulse_train_interval': 10.0  # sec - very long intervals
            }
        }
        
        return patterns
    
    def _initialize_hardware(self):
        """Initialize the stimulation hardware (if available)"""
        # In practice, this would initialize communication with the hardware
        # This is a placeholder for demonstration
        try:
            # Simulate hardware initialization
            time.sleep(0.5)
            
            # Try to detect the actual hardware
            # Code for hardware detection would go here
            
            self.logger.info("Hardware initialization successful")
            return True
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            self.hardware_connected = False
            return False
    
    def set_parameters(self, channel_id, params):
        """
        Set stimulation parameters for a specific channel
        
        Parameters:
        -----------
        channel_id : int
            Channel ID
        params : dict
            Parameter values
            
        Returns:
        --------
        success : bool
            Whether the parameters were set successfully
        """
        if channel_id < 0 or channel_id >= self.n_channels:
            self.logger.warning(f"Invalid channel ID: {channel_id}")
            return False
        
        # Validate parameters
        valid_params = self._validate_parameters(params)
        if not valid_params:
            self.logger.warning("Invalid parameters")
            return False
        
        # Set the parameters
        self.channels[channel_id].set_parameters(params)
        
        # Apply to hardware if connected
        if self.hardware_connected:
            self._apply_parameters_to_hardware(channel_id, params)
        
        # Log parameter change
        self._log_event("set_parameters", {
            "channel_id": channel_id,
            "parameters": params
        })
        
        return True
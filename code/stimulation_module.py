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

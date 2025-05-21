"""
Stimulation Module for Neural Electrical Stimulation

This module handles the electrical stimulation parameters and interfaces
with stimulation hardware (or provides a simulation when hardware is unavailable).

Features:
- Parameter validation and safety checks
- Stimulation profile generation
- Hardware interface (or simulation)
- Stimulation logging
"""

import numpy as np
import time
import threading
import json
import os
from datetime import datetime
from enum import Enum


class StimulationMode(Enum):
    """Stimulation mode enumeration"""
    BIPHASIC = 0
    MONOPHASIC_CATHODIC = 1
    MONOPHASIC_ANODIC = 2
    BURST = 3
    HIGH_FREQUENCY = 4


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
        
        # Status
        self.is_stimulating = False
        self.start_time = None
        self.stop_time = None
    
    def set_parameters(self, params):
        """
        Set stimulation parameters
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameter values
        """
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
    
    def get_parameters(self):
        """
        Get current stimulation parameters
        
        Returns:
        --------
        params : dict
            Dictionary of parameter values
        """
        return {
            'amplitude': self.amplitude,
            'pulse_width': self.pulse_width,
            'frequency': self.frequency,
            'duty_cycle': self.duty_cycle,
            'mode': self.mode,
            'burst_count': self.burst_count,
            'burst_frequency': self.burst_frequency,
            'interphase_gap': self.interphase_gap
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
    
    def __init__(self, n_channels=4, hardware_connected=False, device_id=None):
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
        """
        self.n_channels = n_channels
        self.hardware_connected = hardware_connected
        self.device_id = device_id
        
        # Create channels
        self.channels = [StimulationChannel(i) for i in range(n_channels)]
        
        # Initialize the stimulation thread
        self.stimulation_thread = None
        self.running = False
        
        # Parameter constraints for safety
        self.safety_limits = {
            'amplitude': (0.1, 10.0),     # mA
            'pulse_width': (50, 1000),    # μs
            'frequency': (1, 500),        # Hz
            'duty_cycle': (1, 100),       # %
            'interphase_gap': (0, 1000)   # μs
        }
        
        # Logging
        self.log_path = "logs/stimulation"
        os.makedirs(self.log_path, exist_ok=True)
        self.log_file = os.path.join(self.log_path, f"stim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.log_data = []
        
        print(f"Stimulation module initialized with {n_channels} channels")
        if hardware_connected:
            print(f"Hardware connected: Device ID {device_id}")
            self._initialize_hardware()
        else:
            print("No hardware connected, using simulation mode")
    
    def _initialize_hardware(self):
        """Initialize the stimulation hardware (if available)"""
        # In practice, this would initialize communication with the hardware
        # This is a placeholder for demonstration
        try:
            # Simulate hardware initialization
            time.sleep(0.5)
            print("Hardware initialization successful")
            return True
        except Exception as e:
            print(f"Hardware initialization failed: {e}")
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
            print(f"Invalid channel ID: {channel_id}")
            return False
        
        # Validate parameters
        valid_params = self._validate_parameters(params)
        if not valid_params:
            print("Invalid parameters")
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
    
    def get_parameters(self, channel_id):
        """
        Get current stimulation parameters for a specific channel
        
        Parameters:
        -----------
        channel_id : int
            Channel ID
            
        Returns:
        --------
        params : dict
            Parameter values or None if invalid channel
        """
        if channel_id < 0 or channel_id >= self.n_channels:
            print(f"Invalid channel ID: {channel_id}")
            return None
        
        return self.channels[channel_id].get_parameters()
    
    def _validate_parameters(self, params):
        """
        Validate stimulation parameters for safety
        
        Parameters:
        -----------
        params : dict
            Parameter values
            
        Returns:
        --------
        valid : bool
            Whether the parameters are valid
        """
        for param, value in params.items():
            if param in self.safety_limits:
                min_val, max_val = self.safety_limits[param]
                if value < min_val or value > max_val:
                    print(f"Parameter {param} value {value} is outside safe range [{min_val}, {max_val}]")
                    return False
        
        # Advanced validation (parameter combinations)
        if 'amplitude' in params and 'pulse_width' in params:
            # Charge density safety check
            charge = params['amplitude'] * params['pulse_width'] * 1e-6  # mA * μs -> mC
            if charge > 0.5:  # More than 0.5 mC per phase
                print(f"Warning: High charge density ({charge:.2f} mC)")
                # Could return False here for stricter safety
        
        return True
    
    def _apply_parameters_to_hardware(self, channel_id, params):
        """
        Apply parameters to the stimulation hardware
        
        Parameters:
        -----------
        channel_id : int
            Channel ID
        params : dict
            Parameter values
        """
        # In practice, this would send commands to the actual hardware
        # This is a placeholder for demonstration
        try:
            # Simulate hardware communication
            time.sleep(0.1)
            print(f"Applied parameters to hardware: Channel {channel_id}, {params}")
            return True
        except Exception as e:
            print(f"Hardware communication error: {e}")
            return False
    
    def start_stimulation(self, channel_id=None, duration=None):
        """
        Start stimulation on specified channels
        
        Parameters:
        -----------
        channel_id : int or list or None
            Channel ID(s) to start stimulation on (None for all)
        duration : float or None
            Duration in seconds (None for indefinite)
            
        Returns:
        --------
        success : bool
            Whether stimulation was started successfully
        """
        if channel_id is None:
            # Start on all active channels
            channels_to_start = range(self.n_channels)
        elif isinstance(channel_id, (list, tuple)):
            # Start on multiple specific channels
            channels_to_start = channel_id
        else:
            # Start on a single channel
            channels_to_start = [channel_id]
        
        # Validate channels
        valid_channels = []
        for ch_id in channels_to_start:
            if ch_id < 0 or ch_id >= self.n_channels:
                print(f"Invalid channel ID: {ch_id}")
                continue
            valid_channels.append(ch_id)
        
        if not valid_channels:
            print("No valid channels to start stimulation")
            return False
        
        # Start stimulation on each channel
        for ch_id in valid_channels:
            success = self.channels[ch_id].start_stimulation()
            if not success:
                print(f"Failed to start stimulation on channel {ch_id}")
        
        # Start the stimulation thread if not already running
        if not self.running:
            self.running = True
            self.stimulation_thread = threading.Thread(target=self._stimulation_loop, args=(duration,))
            self.stimulation_thread.daemon = True
            self.stimulation_thread.start()
        
        # Log event
        self._log_event("start_stimulation", {
            "channels": valid_channels,
            "duration": duration
        })
        
        return True
    
    def stop_stimulation(self, channel_id=None):
        """
        Stop stimulation on specified channels
        
        Parameters:
        -----------
        channel_id : int or list or None
            Channel ID(s) to stop stimulation on (None for all)
            
        Returns:
        --------
        success : bool
            Whether stimulation was stopped successfully
        """
        if channel_id is None:
            # Stop on all channels
            channels_to_stop = range(self.n_channels)
        elif isinstance(channel_id, (list, tuple)):
            # Stop on multiple specific channels
            channels_to_stop = channel_id
        else:
            # Stop on a single channel
            channels_to_stop = [channel_id]
        
        # Stop stimulation on each channel
        for ch_id in channels_to_stop:
            if ch_id < 0 or ch_id >= self.n_channels:
                print(f"Invalid channel ID: {ch_id}")
                continue
            
            self.channels[ch_id].stop_stimulation()
        
        # Check if any channels are still stimulating
        any_stimulating = False
        for channel in self.channels:
            if channel.is_stimulating:
                any_stimulating = True
                break
        
        # Stop the stimulation thread if no channels are stimulating
        if not any_stimulating:
            self.running = False
            if self.stimulation_thread:
                self.stimulation_thread.join(timeout=1.0)
        
        # Log event
        self._log_event("stop_stimulation", {
            "channels": list(channels_to_stop)
        })
        
        return True
    
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
                print(f"Invalid channel ID: {channel_id}")
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
        
        print(f"Stimulation started" + (f" for {duration} seconds" if duration else ""))
        
        try:
            while self.running:
                # Check duration limit
                if end_time and time.time() >= end_time:
                    print("Stimulation duration reached, stopping")
                    self.stop_stimulation()
                    break
                
                # Generate stimulation waveform for each active channel
                for channel in self.channels:
                    if not channel.is_stimulating:
                        continue
                    
                    # In a real system, this would apply the waveform to the hardware
                    # For simulation, we just generate it
                    self._generate_waveform(channel)
                
                # Sleep a short time to avoid CPU overload
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Error in stimulation loop: {e}")
            self.running = False
            
            # Emergency stop in case of error
            if self.hardware_connected:
                self._emergency_stop()
        
        print("Stimulation loop ended")
    
    def _generate_waveform(self, channel):
        """
        Generate stimulation waveform for a channel
        
        Parameters:
        -----------
        channel : StimulationChannel
            The channel to generate waveform for
            
        Returns:
        --------
        waveform : ndarray
            Generated waveform (or None if just simulating)
        """
        # This is a simplified simulation
        # In a real system, this would generate actual waveform data
        
        # Calculate duty cycle timing
        if channel.duty_cycle < 100:
            cycle_duration = 1.0  # 1 second cycle
            on_time = cycle_duration * (channel.duty_cycle / 100.0)
            off_time = cycle_duration - on_time
            
            # Check if we're in the ON or OFF period of the duty cycle
            elapsed = (time.time() - channel.start_time) % cycle_duration
            if elapsed > on_time:
                # We're in the OFF period
                return None
        
        # Calculate timing parameters
        period = 1.0 / channel.frequency  # seconds
        
        # Different waveform shapes based on mode
        if channel.mode == StimulationMode.BIPHASIC:
            # Biphasic pulse: positive phase, gap, negative phase
            pulse_duration = 2 * channel.pulse_width + channel.interphase_gap
            
        elif channel.mode == StimulationMode.MONOPHASIC_CATHODIC or channel.mode == StimulationMode.MONOPHASIC_ANODIC:
            # Monophasic pulse: single phase
            pulse_duration = channel.pulse_width
            
        elif channel.mode == StimulationMode.BURST:
            # Burst mode: multiple pulses in quick succession
            pulse_duration = channel.burst_count * (2 * channel.pulse_width + channel.interphase_gap)
            period = 1.0 / channel.burst_frequency
            
        elif channel.mode == StimulationMode.HIGH_FREQUENCY:
            # High frequency: rapid biphasic pulses
            pulse_duration = 2 * channel.pulse_width
            # Period already set by frequency
        
        # In reality, would generate an actual waveform here
        # For simulation, just log the activity
        if self.hardware_connected:
            # Send to hardware (placeholder)
            pass
        else:
            # Simulate the waveform
            pass
        
        return None
    
    def _emergency_stop(self):
        """Emergency stop of all stimulation channels"""
        print("EMERGENCY STOP: Halting all stimulation")
        
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
            print(f"Error writing to log file: {e}")
    
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
        
        print("Stimulation module closed")


# Example usage
if __name__ == "__main__":
    # Create stimulation module
    stim = StimulationModule(n_channels=2, hardware_connected=False)
    
    # Set parameters for both channels
    stim.set_parameters(0, {
        'amplitude': 2.0,
        'pulse_width': 200,
        'frequency': 50,
        'duty_cycle': 50
    })
    
    stim.set_parameters(1, {
        'amplitude': 1.5,
        'pulse_width': 300,
        'frequency': 30,
        'duty_cycle': 75,
        'mode': StimulationMode.BURST,
        'burst_count': 5,
        'burst_frequency': 3
    })
    
    # Start stimulation
    print("\nStarting stimulation on all channels for 5 seconds...")
    stim.start_stimulation(duration=5)
    
    # Wait for completion
    time.sleep(6)
    
    # Check status
    status = stim.get_status()
    print("\nStimulation status:")
    for channel_status in status:
        print(f"Channel {channel_status['channel_id']}: " + 
              f"{'Active' if channel_status['active'] else 'Inactive'}, " + 
              f"{'Stimulating' if channel_status['is_stimulating'] else 'Not stimulating'}")
    
    # Close the module
    stim.close()

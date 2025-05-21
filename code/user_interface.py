"""
User Interface for Adaptive Neural Stimulation System

This module provides a graphical user interface (GUI) for controlling 
and monitoring the adaptive neural stimulation system.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import time
import threading
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from datetime import datetime

# Import system modules (adjust imports based on your project structure)
try:
    from code.acquisition_module import AcquisitionModule, NeuralState
    from code.stimulation_module import StimulationModule, StimulationMode
    from code.control_algorithm import PIDController
    from code.rl_controllers import QLearningController, ActorCriticController
except ImportError:
    print("Warning: Could not import all system modules. Running in demo mode.")
    # Define fallback classes for demo mode
    class AcquisitionModule:
        def __init__(self, **kwargs):
            pass
        def simulate_data(self, duration=1.0, neural_state=None):
            # Generate random data for demonstration
            n_samples = int(duration * 1000)
            return np.random.randn(n_samples, 4) * 0.5 + np.sin(np.linspace(0, 10, n_samples))[:, np.newaxis]
    
    class NeuralState:
        NORMAL = "normal"
        DAMAGED = "damaged"
        RECOVERY = "recovery"
    
    class StimulationModule:
        def __init__(self, **kwargs):
            pass
        def set_parameters(self, channel_id, params):
            return True
        def start_stimulation(self, channel_id=None, duration=None):
            return True
        def stop_stimulation(self, channel_id=None):
            return True
        def get_status(self, channel_id=None):
            return {"is_stimulating": False}
    
    class StimulationMode:
        BIPHASIC = "biphasic"
    
    class PIDController:
        def __init__(self, param_ranges, **kwargs):
            self.current_params = {param: (min_val + max_val) / 2 for param, (min_val, max_val, _) in param_ranges.items()}
        def update(self, neural_state, target_state, features):
            return self.current_params


class AdaptiveStimulationUI:
    """Main UI class for adaptive neural stimulation system"""
    
    def __init__(self, root):
        """Initialize the UI"""
        self.root = root
        self.root.title("Adaptive Neural Stimulation System")
        self.root.geometry("1200x800")
        
        # Initialize system modules
        self.init_system_modules()
        
        # Create the main UI frame
        self.create_ui()
        
        # Initialize data for plotting
        self.signal_data = np.zeros((1000, 4))
        self.time_vector = np.arange(1000) / 1000
        
        # Start data acquisition thread
        self.acquisition_running = False
        self.acquisition_thread = None
        self.start_acquisition()
    
    def init_system_modules(self):
        """Initialize system modules"""
        # Parameter ranges
        self.param_ranges = {
            'frequency': (10, 200, 5),      # Hz (min, max, step)
            'amplitude': (0.5, 5.0, 0.1),   # mA
            'pulse_width': (50, 500, 10),   # µs
            'duty_cycle': (10, 100, 5)      # %
        }
        
        # Acquisition module
        try:
            self.acq_module = AcquisitionModule(sampling_rate=1000, n_channels=4)
        except Exception as e:
            print(f"Error initializing acquisition module: {e}")
            self.acq_module = None
        
        # Stimulation module
        try:
            self.stim_module = StimulationModule(n_channels=4, hardware_connected=False)
        except Exception as e:
            print(f"Error initializing stimulation module: {e}")
            self.stim_module = None
        
        # Controller
        try:
            self.controller = PIDController(self.param_ranges, control_interval=1.0)
        except Exception as e:
            print(f"Error initializing controller: {e}")
            self.controller = None
        
        # Neural state
        self.current_neural_state = NeuralState.NORMAL
        self.target_neural_state = NeuralState.NORMAL
        
        # Stimulation status
        self.is_stimulating = False
    
    def create_ui(self):
        """Create the user interface components"""
        # Create main frame with notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.monitoring_tab = ttk.Frame(self.notebook)
        self.parameters_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.monitoring_tab, text="Monitoring")
        self.notebook.add(self.parameters_tab, text="Parameters")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Create content for each tab
        self.create_monitoring_tab()
        self.create_parameters_tab()
        self.create_settings_tab()
        
        # Status bar at the bottom
        self.status_bar = ttk.Label(self.root, text="System ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_monitoring_tab(self):
        """Create monitoring tab content"""
        # Top control panel
        control_frame = ttk.Frame(self.monitoring_tab)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Neural state selection
        ttk.Label(control_frame, text="Current State:").pack(side=tk.LEFT, padx=5)
        self.state_var = tk.StringVar(value="Normal")
        state_combo = ttk.Combobox(control_frame, textvariable=self.state_var, 
                                  values=["Normal", "Damaged", "Recovery", "Parkinsons"])
        state_combo.pack(side=tk.LEFT, padx=5)
        state_combo.bind("<<ComboboxSelected>>", self.change_neural_state)
        
        # Target state selection
        ttk.Label(control_frame, text="Target State:").pack(side=tk.LEFT, padx=5)
        self.target_var = tk.StringVar(value="Normal")
        target_combo = ttk.Combobox(control_frame, textvariable=self.target_var,
                                   values=["Normal", "Damaged", "Recovery"])
        target_combo.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        self.start_button = ttk.Button(control_frame, text="Start Stimulation", 
                                      command=self.start_stimulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Stimulation", 
                                     command=self.stop_stimulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Signal display
        signal_frame = ttk.LabelFrame(self.monitoring_tab, text="Neural Signals")
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure for the plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Initialize the plot
        self.lines = []
        for i in range(4):  # 4 channels
            line, = self.ax.plot(self.time_vector, self.signal_data[:, i], 
                               label=f"Channel {i+1}")
            self.lines.append(line)
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Real-time Neural Signals")
        self.ax.legend()
        self.ax.grid(True)
        
        # Embedding the plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=signal_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=False)
    
    def create_parameters_tab(self):
        """Create parameters tab content"""
        # Control method selection
        control_frame = ttk.Frame(self.parameters_tab)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Control Method:").pack(side=tk.LEFT, padx=5)
        self.control_method_var = tk.StringVar(value="PID")
        control_method_combo = ttk.Combobox(control_frame, textvariable=self.control_method_var,
                                           values=["PID", "Q-Learning", "Actor-Critic", "Manual"])
        control_method_combo.pack(side=tk.LEFT, padx=5)
        control_method_combo.bind("<<ComboboxSelected>>", self.change_control_method)
        
        # Parameters sliders
        params_frame = ttk.LabelFrame(self.parameters_tab, text="Stimulation Parameters")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create sliders for each parameter
        self.parameter_sliders = {}
        
        row = 0
        for param, (min_val, max_val, step) in self.param_ranges.items():
            # Label
            ttk.Label(params_frame, text=f"{param.capitalize()}:").grid(
                row=row, column=0, padx=5, pady=5, sticky=tk.W)
            
            # Slider
            slider_var = tk.DoubleVar(value=(min_val + max_val) / 2)
            slider = ttk.Scale(params_frame, from_=min_val, to=max_val, 
                             variable=slider_var, orient=tk.HORIZONTAL,
                             length=300, command=lambda v, p=param: self.update_parameter(p))
            slider.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
            
            # Value display
            value_label = ttk.Label(params_frame, text=f"{slider_var.get():.1f}")
            value_label.grid(row=row, column=2, padx=5, pady=5, sticky=tk.W)
            
            # Units
            units = {"frequency": "Hz", "amplitude": "mA", 
                    "pulse_width": "µs", "duty_cycle": "%"}
            ttk.Label(params_frame, text=units.get(param, "")).grid(
                row=row, column=3, padx=5, pady=5, sticky=tk.W)
            
            # Store references
            self.parameter_sliders[param] = {
                "var": slider_var,
                "slider": slider,
                "label": value_label
            }
            
            row += 1
        
        # Apply button
        ttk.Button(params_frame, text="Apply Parameters", 
                  command=self.apply_parameters).grid(
            row=row, column=0, columnspan=4, pady=10)
    
    def create_settings_tab(self):
        """Create settings tab content"""
        settings_frame = ttk.LabelFrame(self.settings_tab, text="System Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Acquisition settings
        acq_frame = ttk.LabelFrame(settings_frame, text="Acquisition Settings")
        acq_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(acq_frame, text="Sampling Rate (Hz):").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sampling_rate_var = tk.IntVar(value=1000)
        ttk.Entry(acq_frame, textvariable=self.sampling_rate_var, width=10).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Stimulation settings
        stim_frame = ttk.LabelFrame(settings_frame, text="Stimulation Settings")
        stim_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(stim_frame, text="Stimulation Mode:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.stim_mode_var = tk.StringVar(value="Biphasic")
        ttk.Combobox(stim_frame, textvariable=self.stim_mode_var,
                    values=["Biphasic", "Monophasic Cathodic", "Monophasic Anodic", "Burst"]).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Control settings
        control_frame = ttk.LabelFrame(settings_frame, text="Control Settings")
        control_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # PID Settings
        ttk.Label(control_frame, text="PID Gains:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(control_frame, text="Kp:").grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.kp_var = tk.DoubleVar(value=1.0)
        ttk.Entry(control_frame, textvariable=self.kp_var, width=5).grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(control_frame, text="Ki:").grid(
            row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.ki_var = tk.DoubleVar(value=0.1)
        ttk.Entry(control_frame, textvariable=self.ki_var, width=5).grid(
            row=0, column=4, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(control_frame, text="Kd:").grid(
            row=0, column=5, padx=5, pady=5, sticky=tk.W)
        self.kd_var = tk.DoubleVar(value=0.2)
        ttk.Entry(control_frame, textvariable=self.kd_var, width=5).grid(
            row=0, column=6, padx=5, pady=5, sticky=tk.W)
        
        # Save/Load settings
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save Settings", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings", 
                  command=self.load_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Settings", 
                  command=self.apply_settings).pack(side=tk.LEFT, padx=5)
    
    def start_acquisition(self):
        """Start data acquisition thread"""
        if self.acquisition_running:
            return
        
        self.acquisition_running = True
        self.acquisition_thread = threading.Thread(target=self.acquisition_loop)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
    
    def acquisition_loop(self):
        """Main data acquisition loop"""
        while self.acquisition_running:
            try:
                # Simulate or acquire data
                if self.acq_module:
                    new_data = self.acq_module.simulate_data(
                        duration=0.1, neural_state=self.current_neural_state)
                else:
                    # Fallback to dummy data
                    new_data = np.random.randn(100, 4) * 0.5
                
                # Update signal data buffer (rolling window)
                self.signal_data = np.vstack([self.signal_data[new_data.shape[0]:], new_data])
                
                time.sleep(0.05)  # Sleep to avoid CPU overload
            except Exception as e:
                print(f"Error in acquisition loop: {e}")
                self.acquisition_running = False
                break
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        # Update each line with new data
        for i, line in enumerate(self.lines):
            line.set_ydata(self.signal_data[:, i])
        
        # Adjust y limits dynamically
        self.ax.relim()
        self.ax.autoscale_view()
        
        return self.lines
    
    def update_parameter(self, parameter):
        """Update parameter display when slider is moved"""
        slider_data = self.parameter_sliders[parameter]
        value = slider_data["var"].get()
        slider_data["label"].config(text=f"{value:.1f}")
    
    def apply_parameters(self):
        """Apply parameter changes to the stimulation system"""
        # Get parameters from sliders
        params = {}
        for param, slider_data in self.parameter_sliders.items():
            params[param] = slider_data["var"].get()
        
        # Apply to all stimulation channels
        if self.stim_module:
            for channel in range(4):  # Assuming 4 channels
                self.stim_module.set_parameters(channel, params)
        
        # Update status
        self.status_bar.config(text=f"Parameters applied at {datetime.now().strftime('%H:%M:%S')}")
    
    def start_stimulation(self):
        """Start the stimulation"""
        if self.is_stimulating:
            return
        
        # Apply parameters first
        self.apply_parameters()
        
        # Start stimulation
        if self.stim_module:
            self.stim_module.start_stimulation()
        
        # Update UI
        self.is_stimulating = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_bar.config(text="Stimulation started")
    
    def stop_stimulation(self):
        """Stop the stimulation"""
        if not self.is_stimulating:
            return
        
        # Stop stimulation
        if self.stim_module:
            self.stim_module.stop_stimulation()
        
        # Update UI
        self.is_stimulating = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_bar.config(text="Stimulation stopped")
    
    def change_neural_state(self, event):
        """Change the neural state"""
        state_name = self.state_var.get().upper()
        
        # Map state name to enum
        state_map = {
            "NORMAL": NeuralState.NORMAL,
            "DAMAGED": NeuralState.DAMAGED,
            "RECOVERY": NeuralState.RECOVERY
        }
        
        if state_name in state_map:
            self.current_neural_state = state_map[state_name]
            if self.acq_module:
                self.acq_module.set_neural_state(self.current_neural_state)
            
            self.status_bar.config(text=f"Neural state changed to {state_name}")
    
    def change_control_method(self, event):
        """Change the control method"""
        method = self.control_method_var.get()
        
        # Create appropriate controller based on method
        if method == "PID":
            self.controller = PIDController(self.param_ranges)
        elif method == "Q-Learning":
            self.controller = QLearningController(self.param_ranges)
        elif method == "Actor-Critic":
            self.controller = ActorCriticController(self.param_ranges)
        else:  # Manual
            self.controller = None
        
        self.status_bar.config(text=f"Control method changed to {method}")
    
    def save_settings(self):
        """Save current settings to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Save Settings"
        )
        
        if not file_path:
            return
        
        # Collect settings
        settings = {
            "sampling_rate": self.sampling_rate_var.get(),
            "stimulation_mode": self.stim_mode_var.get(),
            "pid_gains": {
                "kp": self.kp_var.get(),
                "ki": self.ki_var.get(),
                "kd": self.kd_var.get()
            },
            "parameters": {param: slider_data["var"].get() 
                         for param, slider_data in self.parameter_sliders.items()}
        }
        
        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.status_bar.config(text=f"Settings saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from a file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Load Settings"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)
            
            # Apply loaded settings
            if "sampling_rate" in settings:
                self.sampling_rate_var.set(settings["sampling_rate"])
            
            if "stimulation_mode" in settings:
                self.stim_mode_var.set(settings["stimulation_mode"])
            
            if "pid_gains" in settings:
                self.kp_var.set(settings["pid_gains"].get("kp", 1.0))
                self.ki_var.set(settings["pid_gains"].get("ki", 0.1))
                self.kd_var.set(settings["pid_gains"].get("kd", 0.2))
            
            if "parameters" in settings:
                for param, value in settings["parameters"].items():
                    if param in self.parameter_sliders:
                        self.parameter_sliders[param]["var"].set(value)
                        self.parameter_sliders[param]["label"].config(text=f"{value:.1f}")
            
            self.status_bar.config(text=f"Settings loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading settings: {e}")
    
    def apply_settings(self):
        """Apply the current settings"""
        # Apply to acquisition module
        if self.acq_module:
            pass  # In a real implementation, update acquisition settings
        
        # Apply to control module
        if self.controller and isinstance(self.controller, PIDController):
            self.controller.kp = self.kp_var.get()
            self.controller.ki = self.ki_var.get()
            self.controller.kd = self.kd_var.get()
        
        self.status_bar.config(text="Settings applied")


# Main function
def main():
    root = tk.Tk()
    app = AdaptiveStimulationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

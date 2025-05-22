\"\"\"
Web Interface for Adaptive Neural Stimulation System

This module provides a web-based interface for monitoring and controlling 
the adaptive neural stimulation system using Flask.
\"\"\"

from flask import Flask, render_template, jsonify, request, send_from_directory
import threading
import time
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import system modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Import system modules (adjust imports based on your project structure)
try:
    from code.acquisition_module import AcquisitionModule, NeuralState
    from code.stimulation_module import StimulationModule, StimulationMode
    from code.control_algorithm import PIDController
    from code.rl_controllers import QLearningController, ActorCriticController
except ImportError:
    print(\"Warning: Could not import all system modules. Running in demo mode.\")
    # Define fallback classes for demo mode
    class AcquisitionModule:
        def __init__(self, **kwargs):
            pass
        def simulate_data(self, duration=1.0, neural_state=None):
            # Generate random data for demonstration
            n_samples = int(duration * 1000)
            return np.random.randn(n_samples, 4) * 0.5 + np.sin(np.linspace(0, 10, n_samples))[:, np.newaxis]
    
    class NeuralState:
        NORMAL = \"normal\"
        DAMAGED = \"damaged\"
        RECOVERY = \"recovery\"
    
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
            return {\"is_stimulating\": False}
    
    class StimulationMode:
        BIPHASIC = \"biphasic\"
    
    class PIDController:
        def __init__(self, param_ranges, **kwargs):
            self.current_params = {param: (min_val + max_val) / 2 for param, (min_val, max_val, _) in param_ranges.items()}
        def update(self, neural_state, target_state, features):
            return self.current_params

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System state and data
class SystemState:
    def __init__(self):
        # Parameter ranges
        self.param_ranges = {
            'frequency': (10, 200, 5),      # Hz (min, max, step)
            'amplitude': (0.5, 5.0, 0.1),   # mA
            'pulse_width': (50, 500, 10),   # µs
            'duty_cycle': (10, 100, 5)      # %
        }
        
        # Initialize modules
        self.init_modules()
        
        # Data buffer
        self.signal_data = np.zeros((1000, 4))
        self.signal_data_lock = threading.Lock()
        
        # Neural state
        self.current_neural_state = NeuralState.NORMAL
        self.target_neural_state = NeuralState.NORMAL
        
        # Stimulation status
        self.is_stimulating = False
        
        # Current parameters
        self.current_params = {
            'frequency': 50,      # Hz
            'amplitude': 2.0,     # mA
            'pulse_width': 200,   # µs
            'duty_cycle': 50      # %
        }
        
        # Control method
        self.control_method = "PID"
        
        # Start acquisition thread
        self.acquisition_running = False
        self.acquisition_thread = None
        self.start_acquisition()
        
        # History data for trends
        self.history = {
            'timestamps': [],
            'neural_state': [],
            'parameters': []
        }
        
        # Recovery metrics
        self.recovery_metrics = {
            'axon_density': 30,    # percentage
            'conduction_velocity': 25,  # percentage of normal
            'functional_score': 15      # percentage recovery
        }
        
    def init_modules(self):
        """Initialize system modules"""
        try:
            self.acq_module = AcquisitionModule(sampling_rate=1000, n_channels=4)
            logger.info("Acquisition module initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing acquisition module: {e}")
            self.acq_module = None
        
        try:
            self.stim_module = StimulationModule(n_channels=4, hardware_connected=False)
            logger.info("Stimulation module initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing stimulation module: {e}")
            self.stim_module = None
        
        try:
            self.controller = PIDController(self.param_ranges, control_interval=1.0)
            logger.info("Controller initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing controller: {e}")
            self.controller = None
    
    def start_acquisition(self):
        """Start data acquisition thread"""
        if self.acquisition_running:
            return
        
        self.acquisition_running = True
        self.acquisition_thread = threading.Thread(target=self.acquisition_loop)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
        logger.info("Acquisition thread started")
    
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
                with self.signal_data_lock:
                    self.signal_data = np.vstack([self.signal_data[new_data.shape[0]:], new_data])
                
                # Every 10 seconds, save a snapshot to history
                if len(self.history['timestamps']) == 0 or \
                   (datetime.now() - self.history['timestamps'][-1]).total_seconds() >= 10:
                    self.history['timestamps'].append(datetime.now())
                    self.history['neural_state'].append(self.current_neural_state)
                    self.history['parameters'].append(self.current_params.copy())
                    
                    # Update recovery metrics with small changes to simulate progress
                    for key in self.recovery_metrics:
                        if self.is_stimulating and self.current_neural_state != NeuralState.NORMAL:
                            # Simulate improvement during stimulation
                            self.recovery_metrics[key] += np.random.uniform(0.1, 0.3)
                        elif not self.is_stimulating and self.current_neural_state == NeuralState.DAMAGED:
                            # Simulate slight degradation if damaged and not stimulating
                            self.recovery_metrics[key] -= np.random.uniform(0, 0.1)
                        
                        # Keep within bounds
                        self.recovery_metrics[key] = max(0, min(100, self.recovery_metrics[key]))
                
                time.sleep(0.05)  # Sleep to avoid CPU overload
            except Exception as e:
                logger.error(f"Error in acquisition loop: {e}")
                self.acquisition_running = False
                break
    
    def get_signal_data(self):
        """Get current signal data safely"""
        with self.signal_data_lock:
            return self.signal_data.copy()
    
    def set_neural_state(self, state_name):
        """Set the current neural state"""
        state_map = {
            "NORMAL": NeuralState.NORMAL,
            "DAMAGED": NeuralState.DAMAGED,
            "RECOVERY": NeuralState.RECOVERY
        }
        
        if state_name.upper() in state_map:
            self.current_neural_state = state_map[state_name.upper()]
            if self.acq_module:
                self.acq_module.set_neural_state(self.current_neural_state)
            logger.info(f"Neural state changed to {state_name}")
            return True
        return False
    
    def set_target_state(self, state_name):
        """Set the target neural state"""
        state_map = {
            "NORMAL": NeuralState.NORMAL,
            "DAMAGED": NeuralState.DAMAGED,
            "RECOVERY": NeuralState.RECOVERY
        }
        
        if state_name.upper() in state_map:
            self.target_neural_state = state_map[state_name.upper()]
            logger.info(f"Target state changed to {state_name}")
            return True
        return False
    
    def set_control_method(self, method):
        """Change the control method"""
        if method in ["PID", "Q-Learning", "Actor-Critic", "Manual"]:
            self.control_method = method
            
            # Create appropriate controller based on method
            if method == "PID":
                self.controller = PIDController(self.param_ranges)
            elif method == "Q-Learning":
                self.controller = QLearningController(self.param_ranges)
            elif method == "Actor-Critic":
                self.controller = ActorCriticController(self.param_ranges)
            else:  # Manual
                self.controller = None
            
            logger.info(f"Control method changed to {method}")
            return True
        return False
    
    def set_parameters(self, params):
        """Set stimulation parameters"""
        for param, value in params.items():
            if param in self.current_params:
                min_val, max_val, _ = self.param_ranges.get(param, (0, 100, 1))
                # Ensure value is within range
                self.current_params[param] = max(min_val, min(max_val, float(value)))
        
        # Apply to stimulation module
        if self.stim_module:
            for channel in range(4):  # Assuming 4 channels
                self.stim_module.set_parameters(channel, self.current_params)
        
        logger.info(f"Parameters updated: {self.current_params}")
        return True
    
    def start_stimulation(self):
        """Start stimulation"""
        if self.is_stimulating:
            return False
        
        if self.stim_module:
            self.stim_module.start_stimulation()
        
        self.is_stimulating = True
        logger.info("Stimulation started")
        return True
    
    def stop_stimulation(self):
        """Stop stimulation"""
        if not self.is_stimulating:
            return False
        
        if self.stim_module:
            self.stim_module.stop_stimulation()
        
        self.is_stimulating = False
        logger.info("Stimulation stopped")
        return True

# Initialize system state
system_state = SystemState()

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/monitoring')
def monitoring():
    """Render the monitoring page"""
    return render_template('monitoring.html')

@app.route('/parameters')
def parameters():
    """Render the parameters configuration page"""
    return render_template('parameters.html')

@app.route('/analytics')
def analytics():
    """Render the analytics and results page"""
    return render_template('analytics.html')

@app.route('/settings')
def settings():
    """Render the system settings page"""
    return render_template('settings.html')

@app.route('/api/system-state')
def get_system_state():
    """Return the current system state as JSON"""
    return jsonify({
        'neural_state': system_state.current_neural_state,
        'target_state': system_state.target_neural_state,
        'is_stimulating': system_state.is_stimulating,
        'control_method': system_state.control_method,
        'parameters': system_state.current_params,
        'recovery_metrics': system_state.recovery_metrics
    })

@app.route('/api/signal-data')
def get_signal_data():
    """Return the latest signal data for plotting"""
    data = system_state.get_signal_data()
    # Convert to list of lists for JSON serialization
    return jsonify({
        'data': data.tolist(),
        'timestamps': list(range(len(data)))
    })

@app.route('/api/history-data')
def get_history_data():
    """Return historical data for trend analysis"""
    # Convert datetime objects to strings
    timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in system_state.history['timestamps']]
    
    return jsonify({
        'timestamps': timestamps,
        'neural_state': system_state.history['neural_state'],
        'parameters': system_state.history['parameters']
    })

@app.route('/api/recovery-metrics')
def get_recovery_metrics():
    """Return recovery metrics data"""
    return jsonify(system_state.recovery_metrics)

@app.route('/api/set-neural-state', methods=['POST'])
def set_neural_state():
    """Set the current neural state"""
    data = request.get_json()
    if 'state' in data:
        success = system_state.set_neural_state(data['state'])
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'State parameter missing'})

@app.route('/api/set-target-state', methods=['POST'])
def set_target_state():
    """Set the target neural state"""
    data = request.get_json()
    if 'state' in data:
        success = system_state.set_target_state(data['state'])
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'State parameter missing'})

@app.route('/api/set-parameters', methods=['POST'])
def set_parameters():
    """Set stimulation parameters"""
    data = request.get_json()
    if 'parameters' in data and isinstance(data['parameters'], dict):
        success = system_state.set_parameters(data['parameters'])
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'Parameters missing or invalid'})

@app.route('/api/set-control-method', methods=['POST'])
def set_control_method():
    """Set the control method"""
    data = request.get_json()
    if 'method' in data:
        success = system_state.set_control_method(data['method'])
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': 'Method parameter missing'})

@app.route('/api/start-stimulation', methods=['POST'])
def start_stimulation():
    """Start stimulation"""
    success = system_state.start_stimulation()
    return jsonify({'success': success})

@app.route('/api/stop-stimulation', methods=['POST'])
def stop_stimulation():
    """Stop stimulation"""
    success = system_state.stop_stimulation()
    return jsonify({'success': success})

@app.route('/api/save-settings', methods=['POST'])
def save_settings():
    """Save system settings to a file"""
    data = request.get_json()
    if 'filename' in data:
        filename = data['filename']
        if not filename.endswith('.json'):
            filename += '.json'
        
        settings = {
            'parameters': system_state.current_params,
            'control_method': system_state.control_method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # Save to settings directory
            settings_dir = os.path.join(parent_dir, 'settings')
            os.makedirs(settings_dir, exist_ok=True)
            
            file_path = os.path.join(settings_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            
            return jsonify({'success': True, 'path': file_path})
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Filename parameter missing'})

@app.route('/api/load-settings', methods=['POST'])
def load_settings():
    """Load system settings from a file"""
    data = request.get_json()
    if 'filename' in data:
        filename = data['filename']
        if not filename.endswith('.json'):
            filename += '.json'
        
        try:
            # Load from settings directory
            settings_dir = os.path.join(parent_dir, 'settings')
            file_path = os.path.join(settings_dir, filename)
            
            with open(file_path, 'r') as f:
                settings = json.load(f)
            
            # Apply settings
            if 'parameters' in settings:
                system_state.set_parameters(settings['parameters'])
            
            if 'control_method' in settings:
                system_state.set_control_method(settings['control_method'])
            
            return jsonify({'success': True, 'settings': settings})
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Filename parameter missing'})

@app.route('/api/list-settings')
def list_settings():
    """List available settings files"""
    settings_dir = os.path.join(parent_dir, 'settings')
    os.makedirs(settings_dir, exist_ok=True)
    
    files = [f for f in os.listdir(settings_dir) if f.endswith('.json')]
    return jsonify({'files': files})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# adaptive-neural-stimulation-system/code/config.py

# Safety limits for stimulation parameters
# Format: {parameter_name: (min_value, max_value)}
# IMPORTANT: Parameter names here must match the attributes in StimulationChannel
SAFETY_LIMITS = {
    "amplitude_mA": (0.1, 10.0),     # mA
    "pulse_width_us": (50, 1000),    # μs
    "frequency_hz": (1, 500),        # Hz
    "duty_cycle_percent": (1, 100),  # %
    "interphase_gap_us": (0, 1000),  # μs
    "phase_asymmetry_ratio": (0.1, 10.0), # Ratio
    "burst_count": (1, 10),          # Number of pulses in a burst
    "burst_frequency_hz": (1, 50),   # Hz (frequency of bursts)
    "ramp_up_time_s": (0.0, 5.0),    # Seconds
    "ramp_down_time_s": (0.0, 5.0),  # Seconds
    "pulse_train_interval_s": (0.1, 60.0), # Seconds
    "charge_mC_per_phase": (0.000001, 0.5) # mC (charge density limit per phase)
}

# Predefined neural regeneration patterns
# These parameters are examples and should be based on actual research.
# IMPORTANT: Pattern names here must match the string values of NeuralRegenerationPattern Enum members
PATTERN_TEMPLATES = {
    "EARLY_STAGE": {
        "amplitude_mA": 0.8,
        "pulse_width_us": 150,
        "frequency_hz": 20,
        "mode": "BIPHASIC",
        "ramp_up_time_s": 0.2,
        "ramp_down_time_s": 0.2
    },
    "MID_STAGE": {
        "amplitude_mA": 1.2,
        "pulse_width_us": 200,
        "frequency_hz": 50,
        "mode": "BURST",
        "burst_count": 5,
        "burst_frequency_hz": 10
    },
    "LATE_STAGE": {
        "amplitude_mA": 0.6,
        "pulse_width_us": 100,
        "frequency_hz": 100,
        "mode": "BIPHASIC",
        "phase_asymmetry_ratio": 1.2
    },
    "ACUTE_INJURY": {
        "amplitude_mA": 0.5,
        "pulse_width_us": 100,
        "frequency_hz": 1,
        "mode": "MONOPHASIC_CATHODIC"
    },
    "CHRONIC_INJURY": {
        "amplitude_mA": 2.0,
        "pulse_width_us": 250,
        "frequency_hz": 130,
        "mode": "BIPHASIC"
    },
    "MOTOR_NERVE": {
        "amplitude_mA": 1.5,
        "pulse_width_us": 200,
        "frequency_hz": 30,
        "mode": "BIPHASIC",
        "duty_cycle_percent": 80
    },
    "SENSORY_NERVE": {
        "amplitude_mA": 0.7,
        "pulse_width_us": 120,
        "frequency_hz": 80,
        "mode": "BIPHASIC"
    },
    "BDNF_ENHANCING": {
        "amplitude_mA": 1.0,
        "pulse_width_us": 180,
        "frequency_hz": 20,
        "mode": "BURST",
        "burst_count": 3,
        "burst_frequency_hz": 5
    },
    "GDNF_ENHANCING": {
        "amplitude_mA": 1.1,
        "pulse_width_us": 220,
        "frequency_hz": 10,
        "mode": "LONG_DURATION_LOW_FREQUENCY",
        "pulse_train_interval_s": 5.0
    }
}

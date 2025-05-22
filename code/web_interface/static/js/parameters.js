/**
 * Parameters JavaScript for Adaptive Neural Stimulation System
 * 
 * This script handles the parameters page's functionality including:
 * - Parameter control via sliders
 * - Protocol templates
 * - Advanced stimulation settings
 * - Settings save/load functionality
 */

// Global variables
let currentParameters = {
    frequency: 50,
    amplitude: 2.0,
    pulse_width: 200,
    duty_cycle: 50
};

let advancedSettings = {
    waveform: 'biphasic',
    stimDuration: 30,
    interStimInterval: 0,
    rampUp: 5,
    rampDown: 5,
    randomize: false
};

let isAdaptive = true;
let isStimulating = false;
let saveProtocolModal = null;
let loadSettingsModal = null;
let selectedProtocol = null;

// Parameter protocols templates
const protocolTemplates = {
    bdnf: {
        name: 'BDNF 유도 프로토콜',
        description: 'BDNF 발현을 최대화하기 위한 20Hz 기반 자극 프로토콜',
        parameters: {
            frequency: 20,
            amplitude: 2.5,
            pulse_width: 300,
            duty_cycle: 50
        }
    },
    gdnf: {
        name: '슈반세포 증식 프로토콜',
        description: '슈반세포 증식과 GDNF 발현을 촉진하는 저주파 프로토콜',
        parameters: {
            frequency: 5,
            amplitude: 3.0,
            pulse_width: 400,
            duty_cycle: 70
        }
    },
    gap43: {
        name: '축삭 성장 가속화 프로토콜',
        description: 'cAMP 생성과 GAP-43 발현을 촉진하는 고주파 프로토콜',
        parameters: {
            frequency: 50,
            amplitude: 2.0,
            pulse_width: 250,
            duty_cycle: 60
        }
    },
    pain: {
        name: '통증 억제 프로토콜',
        description: 'GABA 분비를 촉진하고 통증을 억제하는 고주파 프로토콜',
        parameters: {
            frequency: 100,
            amplitude: 1.5,
            pulse_width: 150,
            duty_cycle: 40
        }
    },
    custom: {
        name: '사용자 정의 프로토콜',
        description: '저장된 사용자 정의 프로토콜',
        parameters: {
            frequency: 35,
            amplitude: 2.3,
            pulse_width: 275,
            duty_cycle: 55
        }
    }
};

// Initialize the parameters page when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI components
    initializeSliders();
    initializeEventListeners();
    
    // Initialize modals
    saveProtocolModal = new bootstrap.Modal(document.getElementById('saveProtocolModal'));
    loadSettingsModal = new bootstrap.Modal(document.getElementById('loadSettingsModal'));
    
    // Load initial system state
    loadSystemState();
});

/**
 * Initialize parameter sliders and their event listeners
 */
function initializeSliders() {
    // Frequency slider
    const frequencySlider = document.getElementById('frequencySlider');
    const frequencyValue = document.getElementById('frequencyValue');
    
    frequencySlider.addEventListener('input', function() {
        const value = parseInt(this.value);
        frequencyValue.textContent = value;
        currentParameters.frequency = value;
    });
    
    // Amplitude slider
    const amplitudeSlider = document.getElementById('amplitudeSlider');
    const amplitudeValue = document.getElementById('amplitudeValue');
    
    amplitudeSlider.addEventListener('input', function() {
        const value = parseFloat(this.value).toFixed(1);
        amplitudeValue.textContent = value;
        currentParameters.amplitude = parseFloat(value);
    });
    
    // Pulse width slider
    const pulseWidthSlider = document.getElementById('pulseWidthSlider');
    const pulseWidthValue = document.getElementById('pulseWidthValue');
    
    pulseWidthSlider.addEventListener('input', function() {
        const value = parseInt(this.value);
        pulseWidthValue.textContent = value;
        currentParameters.pulse_width = value;
    });
    
    // Duty cycle slider
    const dutyCycleSlider = document.getElementById('dutyCycleSlider');
    const dutyCycleValue = document.getElementById('dutyCycleValue');
    
    dutyCycleSlider.addEventListener('input', function() {
        const value = parseInt(this.value);
        dutyCycleValue.textContent = value;
        currentParameters.duty_cycle = value;
    });
}

/**
 * Initialize event listeners for UI controls
 */
function initializeEventListeners() {
    // Control method selection
    document.getElementById('controlMethodSelect').addEventListener('change', function() {
        const method = this.value;
        fetch('/api/set-control-method', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ method: method })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update any UI elements if needed
            }
        })
        .catch(error => console.error('Error setting control method:', error));
    });

    // Target state selection
    document.getElementById('targetStateSelect').addEventListener('change', function() {
        const state = this.value;
        fetch('/api/set-target-state', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ state: state })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update any UI elements if needed
            }
        })
        .catch(error => console.error('Error setting target state:', error));
    });

    // Adaptive toggle
    document.getElementById('adaptiveToggle').addEventListener('change', function() {
        isAdaptive = this.checked;
        
        // Enable/disable sliders based on adaptive mode
        const sliders = document.querySelectorAll('.form-range');
        sliders.forEach(slider => {
            slider.disabled = isAdaptive;
        });
        
        // Update UI
        document.getElementById('applyParamsBtn').disabled = isAdaptive;
    });

    // Apply parameters button
    document.getElementById('applyParamsBtn').addEventListener('click', function() {
        applyParameters();
    });

    // Protocol cards
    const protocolCards = document.querySelectorAll('.protocol-card');
    protocolCards.forEach(card => {
        card.addEventListener('click', function() {
            const protocolId = this.getAttribute('data-protocol');
            
            if (protocolId === 'new') {
                // Show save protocol modal
                updateModalWithCurrentParameters();
                saveProtocolModal.show();
            } else {
                // Apply the selected protocol
                applyProtocol(protocolId);
                
                // Update selected state
                protocolCards.forEach(c => c.classList.remove('selected'));
                this.classList.add('selected');
                selectedProtocol = protocolId;
            }
        });
    });

    // Save protocol button
    document.getElementById('saveProtocolBtn').addEventListener('click', function() {
        const name = document.getElementById('protocolNameInput').value;
        const description = document.getElementById('protocolDescInput').value;
        
        if (!name) {
            alert('프로토콜 이름을 입력해주세요.');
            return;
        }
        
        // Save the protocol (in a real implementation, this would save to a database or file)
        // For this demo, we'll just show an alert
        alert(`프로토콜 "${name}"이(가) 저장되었습니다.`);
        saveProtocolModal.hide();
    });

    // Start stimulation button
    document.getElementById('startStimBtn').addEventListener('click', function() {
        fetch('/api/start-stimulation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI
                isStimulating = true;
                document.getElementById('startStimBtn').disabled = true;
                document.getElementById('stopStimBtn').disabled = false;
            }
        })
        .catch(error => console.error('Error starting stimulation:', error));
    });

    // Stop stimulation button
    document.getElementById('stopStimBtn').addEventListener('click', function() {
        fetch('/api/stop-stimulation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI
                isStimulating = false;
                document.getElementById('startStimBtn').disabled = false;
                document.getElementById('stopStimBtn').disabled = true;
            }
        })
        .catch(error => console.error('Error stopping stimulation:', error));
    });

    // Advanced settings inputs
    document.getElementById('waveformSelect').addEventListener('change', function() {
        advancedSettings.waveform = this.value;
    });
    
    document.getElementById('stimDurationInput').addEventListener('change', function() {
        advancedSettings.stimDuration = parseInt(this.value);
    });
    
    document.getElementById('interStimIntervalInput').addEventListener('change', function() {
        advancedSettings.interStimInterval = parseInt(this.value);
    });
    
    document.getElementById('rampUpInput').addEventListener('change', function() {
        advancedSettings.rampUp = parseInt(this.value);
    });
    
    document.getElementById('rampDownInput').addEventListener('change', function() {
        advancedSettings.rampDown = parseInt(this.value);
    });
    
    document.getElementById('randomizeToggle').addEventListener('change', function() {
        advancedSettings.randomize = this.checked;
    });
    
    // Apply advanced settings button
    document.getElementById('applyAdvancedBtn').addEventListener('click', function() {
        // In a real implementation, this would send advanced settings to the server
        // For this demo, we'll just show an alert
        alert('고급 설정이 적용되었습니다.');
    });

    // Save settings button
    document.getElementById('saveParamsBtn').addEventListener('click', function() {
        // In a real implementation, this would save the current parameters to a file
        // For this demo, we'll just show an alert with the current parameters
        const paramsString = JSON.stringify({
            parameters: currentParameters,
            advanced: advancedSettings,
            isAdaptive: isAdaptive,
            controlMethod: document.getElementById('controlMethodSelect').value,
            targetState: document.getElementById('targetStateSelect').value
        }, null, 2);
        
        alert(`다음 설정이 저장되었습니다:\n${paramsString}`);
    });

    // Load settings button
    document.getElementById('loadParamsBtn').addEventListener('click', function() {
        // In a real implementation, this would load the settings files from the server
        // For this demo, we'll just show the modal with dummy data
        loadSettingsModal.show();
    });

    // Settings file list items
    const settingsFiles = document.querySelectorAll('#settingsFileList a');
    settingsFiles.forEach(file => {
        file.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Toggle selection
            settingsFiles.forEach(f => f.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Confirm load button
    document.getElementById('confirmLoadBtn').addEventListener('click', function() {
        const selectedFile = document.querySelector('#settingsFileList a.active');
        
        if (!selectedFile) {
            alert('불러올 설정 파일을 선택해주세요.');
            return;
        }
        
        const fileName = selectedFile.querySelector('h5').textContent;
        
        // In a real implementation, this would load the settings from the server
        // For this demo, we'll just show an alert and load a dummy protocol
        alert(`${fileName} 파일에서 설정을 불러옵니다.`);
        
        // Apply a dummy protocol
        if (fileName.includes('BDNF')) {
            applyProtocol('bdnf');
        } else {
            applyProtocol('custom');
        }
        
        loadSettingsModal.hide();
    });
}

/**
 * Load initial system state from the server
 */
function loadSystemState() {
    fetch('/api/system-state')
        .then(response => response.json())
        .then(data => {
            // Update stimulation status
            isStimulating = data.is_stimulating;
            if (isStimulating) {
                document.getElementById('startStimBtn').disabled = true;
                document.getElementById('stopStimBtn').disabled = false;
            } else {
                document.getElementById('startStimBtn').disabled = false;
                document.getElementById('stopStimBtn').disabled = true;
            }

            // Update control method
            document.getElementById('controlMethodSelect').value = data.control_method;

            // Update parameters and sliders
            updateParametersUI(data.parameters);
        })
        .catch(error => console.error('Error loading system state:', error));
}

/**
 * Update UI with the current parameters
 */
function updateParametersUI(parameters) {
    // Update current parameters
    currentParameters = parameters;
    
    // Update sliders and values
    document.getElementById('frequencySlider').value = parameters.frequency;
    document.getElementById('frequencyValue').textContent = parameters.frequency;
    
    document.getElementById('amplitudeSlider').value = parameters.amplitude;
    document.getElementById('amplitudeValue').textContent = parameters.amplitude.toFixed(1);
    
    document.getElementById('pulseWidthSlider').value = parameters.pulse_width;
    document.getElementById('pulseWidthValue').textContent = parameters.pulse_width;
    
    document.getElementById('dutyCycleSlider').value = parameters.duty_cycle;
    document.getElementById('dutyCycleValue').textContent = parameters.duty_cycle;
}

/**
 * Apply the current parameters to the system
 */
function applyParameters() {
    fetch('/api/set-parameters', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ parameters: currentParameters })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            alert('파라미터가 성공적으로 적용되었습니다.');
        }
    })
    .catch(error => console.error('Error applying parameters:', error));
}

/**
 * Apply a protocol template
 */
function applyProtocol(protocolId) {
    if (protocolTemplates[protocolId]) {
        const protocol = protocolTemplates[protocolId];
        
        // Update current parameters
        currentParameters = { ...protocol.parameters };
        
        // Update UI
        updateParametersUI(currentParameters);
        
        // Apply parameters to the system if not in adaptive mode
        if (!isAdaptive) {
            applyParameters();
        }
    }
}

/**
 * Update the save protocol modal with current parameters
 */
function updateModalWithCurrentParameters() {
    document.getElementById('modalFrequencyValue').textContent = `${currentParameters.frequency} Hz`;
    document.getElementById('modalAmplitudeValue').textContent = `${currentParameters.amplitude.toFixed(1)} mA`;
    document.getElementById('modalPulseWidthValue').textContent = `${currentParameters.pulse_width} µs`;
    document.getElementById('modalDutyCycleValue').textContent = `${currentParameters.duty_cycle} %`;
}

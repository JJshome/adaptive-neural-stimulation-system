/**
 * Dashboard JavaScript for Adaptive Neural Stimulation System
 * 
 * This script handles the dashboard's functionality including:
 * - Real-time signal visualization
 * - System state updates
 * - Parameter control
 * - Stimulation control
 * - Data refresh
 */

// Global variables
let signalChart = null;
let parametersChart = null;
let stimulationStartTime = null;
let updateInterval = null;
let isStimulating = false;

// Chart colors
const chartColors = {
    channel1: 'rgba(54, 162, 235, 1)',
    channel2: 'rgba(255, 99, 132, 1)',
    channel3: 'rgba(75, 192, 192, 1)',
    channel4: 'rgba(255, 159, 64, 1)',
    frequency: 'rgba(54, 162, 235, 1)',
    amplitude: 'rgba(255, 99, 132, 1)',
    pulseWidth: 'rgba(75, 192, 192, 1)',
    dutyCycle: 'rgba(255, 159, 64, 1)'
};

// Initialize the dashboard when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initializeCharts();
    
    // Initialize UI event listeners
    initializeEventListeners();
    
    // Load initial system state
    loadSystemState();
    
    // Start periodic updates
    startPeriodicUpdates();
});

/**
 * Initialize the signal and parameters charts
 */
function initializeCharts() {
    // Signal Chart
    const signalCtx = document.getElementById('signalChart').getContext('2d');
    signalChart = new Chart(signalCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 100}, (_, i) => i),
            datasets: [
                {
                    label: 'Channel 1',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel1,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Channel 2',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel2,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Channel 3',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel3,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Channel 4',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel4,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (samples)'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amplitude'
                    }
                }
            },
            animation: {
                duration: 0
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: false,
                    text: 'Neural Signals'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });

    // Parameters Chart
    const parametersCtx = document.getElementById('parametersChart').getContext('2d');
    parametersChart = new Chart(parametersCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Frequency (Hz)',
                    data: [],
                    borderColor: chartColors.frequency,
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'Amplitude (mA)',
                    data: [],
                    borderColor: chartColors.amplitude,
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    fill: true,
                    yAxisID: 'y1'
                },
                {
                    label: 'Pulse Width (µs)',
                    data: [],
                    borderColor: chartColors.pulseWidth,
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    fill: true,
                    yAxisID: 'y2'
                },
                {
                    label: 'Duty Cycle (%)',
                    data: [],
                    borderColor: chartColors.dutyCycle,
                    backgroundColor: 'rgba(255, 159, 64, 0.1)',
                    fill: true,
                    yAxisID: 'y3'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Frequency (Hz)'
                    },
                    min: 10,
                    max: 200
                },
                y1: {
                    type: 'linear',
                    display: false,
                    position: 'right',
                    min: 0.5,
                    max: 5.0
                },
                y2: {
                    type: 'linear',
                    display: false,
                    position: 'right',
                    min: 50,
                    max: 500
                },
                y3: {
                    type: 'linear',
                    display: false,
                    position: 'right',
                    min: 10,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

/**
 * Initialize event listeners for UI controls
 */
function initializeEventListeners() {
    // Neural state selection
    document.getElementById('neuralStateSelect').addEventListener('change', function() {
        const state = this.value;
        fetch('/api/set-neural-state', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ state: state })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI
                document.getElementById('neuralStateDisplay').textContent = stateDisplayMap[state] || state;
                
                // Add class for styling
                const stateDisplay = document.getElementById('neuralStateDisplay');
                stateDisplay.className = '';
                stateDisplay.classList.add('neural-state-indicator', `neural-state-${state.toLowerCase()}`);
            }
        })
        .catch(error => console.error('Error setting neural state:', error));
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
                // Update UI
                document.getElementById('targetStateDisplay').textContent = stateDisplayMap[state] || state;
            }
        })
        .catch(error => console.error('Error setting target state:', error));
    });

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
                // Update UI
                document.getElementById('controlMethodDisplay').textContent = method;
            }
        })
        .catch(error => console.error('Error setting control method:', error));
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
                document.getElementById('stimStatusDisplay').innerHTML = 
                    '<span class="badge bg-success">활성</span>';
                stimulationStartTime = new Date();
                updateStimulationDuration();
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
                document.getElementById('stimStatusDisplay').innerHTML = 
                    '<span class="badge bg-secondary">비활성</span>';
                stimulationStartTime = null;
                document.getElementById('stimDurationDisplay').textContent = "지속시간: 00:00:00";
            }
        })
        .catch(error => console.error('Error stopping stimulation:', error));
    });
}

/**
 * Load initial system state from the server
 */
function loadSystemState() {
    fetch('/api/system-state')
        .then(response => response.json())
        .then(data => {
            // Update neural state
            const neuralStateMap = {
                'normal': 'NORMAL',
                'damaged': 'DAMAGED',
                'recovery': 'RECOVERY'
            };
            const stateDisplayMap = {
                'NORMAL': '정상',
                'DAMAGED': '손상',
                'RECOVERY': '회복 중'
            };

            // Set the neural state dropdown and display
            const neuralState = neuralStateMap[data.neural_state] || 'NORMAL';
            document.getElementById('neuralStateSelect').value = neuralState;
            document.getElementById('neuralStateDisplay').textContent = stateDisplayMap[neuralState] || neuralState;
            
            // Add class for styling
            const stateDisplay = document.getElementById('neuralStateDisplay');
            stateDisplay.className = '';
            stateDisplay.classList.add('neural-state-indicator', `neural-state-${neuralState.toLowerCase()}`);

            // Set the target state dropdown and display
            const targetState = neuralStateMap[data.target_state] || 'NORMAL';
            document.getElementById('targetStateSelect').value = targetState;
            document.getElementById('targetStateDisplay').textContent = stateDisplayMap[targetState] || targetState;

            // Set control method
            document.getElementById('controlMethodSelect').value = data.control_method;
            document.getElementById('controlMethodDisplay').textContent = data.control_method;

            // Set stimulation status
            isStimulating = data.is_stimulating;
            if (isStimulating) {
                document.getElementById('startStimBtn').disabled = true;
                document.getElementById('stopStimBtn').disabled = false;
                document.getElementById('stimStatusDisplay').innerHTML = 
                    '<span class="badge bg-success">활성</span>';
                stimulationStartTime = new Date();
                updateStimulationDuration();
            } else {
                document.getElementById('startStimBtn').disabled = false;
                document.getElementById('stopStimBtn').disabled = true;
                document.getElementById('stimStatusDisplay').innerHTML = 
                    '<span class="badge bg-secondary">비활성</span>';
                stimulationStartTime = null;
                document.getElementById('stimDurationDisplay').textContent = "지속시간: 00:00:00";
            }

            // Set parameters
            document.getElementById('frequencyDisplay').textContent = data.parameters.frequency;
            document.getElementById('amplitudeDisplay').textContent = data.parameters.amplitude.toFixed(1);
            document.getElementById('pulseWidthDisplay').textContent = data.parameters.pulse_width;
            document.getElementById('dutyCycleDisplay').textContent = data.parameters.duty_cycle;

            // Set recovery metrics
            updateRecoveryMetrics(data.recovery_metrics);
        })
        .catch(error => console.error('Error loading system state:', error));
}

/**
 * Update recovery metrics in the UI
 */
function updateRecoveryMetrics(metrics) {
    const axonDensityProgress = document.getElementById('axonDensityProgress');
    const conductionVelocityProgress = document.getElementById('conductionVelocityProgress');
    const functionalScoreProgress = document.getElementById('functionalScoreProgress');

    axonDensityProgress.style.width = `${metrics.axon_density}%`;
    axonDensityProgress.textContent = `${Math.round(metrics.axon_density)}%`;
    axonDensityProgress.setAttribute('aria-valuenow', metrics.axon_density);

    conductionVelocityProgress.style.width = `${metrics.conduction_velocity}%`;
    conductionVelocityProgress.textContent = `${Math.round(metrics.conduction_velocity)}%`;
    conductionVelocityProgress.setAttribute('aria-valuenow', metrics.conduction_velocity);

    functionalScoreProgress.style.width = `${metrics.functional_score}%`;
    functionalScoreProgress.textContent = `${Math.round(metrics.functional_score)}%`;
    functionalScoreProgress.setAttribute('aria-valuenow', metrics.functional_score);
}

/**
 * Update the stimulation duration display
 */
function updateStimulationDuration() {
    if (!stimulationStartTime || !isStimulating) return;
    
    const now = new Date();
    const diffMs = now - stimulationStartTime;
    
    // Convert to hours, minutes, seconds
    const hours = Math.floor(diffMs / 3600000).toString().padStart(2, '0');
    const minutes = Math.floor((diffMs % 3600000) / 60000).toString().padStart(2, '0');
    const seconds = Math.floor((diffMs % 60000) / 1000).toString().padStart(2, '0');
    
    document.getElementById('stimDurationDisplay').textContent = `지속시간: ${hours}:${minutes}:${seconds}`;
    
    // Continue updating
    if (isStimulating) {
        setTimeout(updateStimulationDuration, 1000);
    }
}

/**
 * Start periodic data updates
 */
function startPeriodicUpdates() {
    // Update signal data every 500ms
    setInterval(updateSignalData, 500);
    
    // Update history data every 5 seconds
    setInterval(updateHistoryData, 5000);
    
    // Update system state every 3 seconds
    setInterval(updateSystemState, 3000);
}

/**
 * Update signal data chart
 */
function updateSignalData() {
    fetch('/api/signal-data')
        .then(response => response.json())
        .then(data => {
            // Extract a subset of the data for display
            const signalData = data.data;
            const subsampleFactor = Math.floor(signalData.length / 100);
            
            // Update chart data
            for (let channel = 0; channel < 4; channel++) {
                const channelData = [];
                for (let i = 0; i < 100; i++) {
                    const idx = Math.min(i * subsampleFactor, signalData.length - 1);
                    channelData.push(signalData[idx][channel]);
                }
                signalChart.data.datasets[channel].data = channelData;
            }
            
            // Update chart
            signalChart.update();
        })
        .catch(error => console.error('Error updating signal data:', error));
}

/**
 * Update parameter history data
 */
function updateHistoryData() {
    fetch('/api/history-data')
        .then(response => response.json())
        .then(data => {
            // Update parameters chart
            parametersChart.data.labels = data.timestamps.map(ts => new Date(ts));
            
            // Extract parameter data
            const frequencyData = [];
            const amplitudeData = [];
            const pulseWidthData = [];
            const dutyCycleData = [];
            
            data.parameters.forEach(params => {
                frequencyData.push(params.frequency);
                amplitudeData.push(params.amplitude);
                pulseWidthData.push(params.pulse_width);
                dutyCycleData.push(params.duty_cycle);
            });
            
            parametersChart.data.datasets[0].data = frequencyData;
            parametersChart.data.datasets[1].data = amplitudeData;
            parametersChart.data.datasets[2].data = pulseWidthData;
            parametersChart.data.datasets[3].data = dutyCycleData;
            
            parametersChart.update();
        })
        .catch(error => console.error('Error updating history data:', error));
}

/**
 * Update system state
 */
function updateSystemState() {
    fetch('/api/system-state')
        .then(response => response.json())
        .then(data => {
            // Update stimulation status
            isStimulating = data.is_stimulating;
            if (isStimulating) {
                if (!stimulationStartTime) {
                    stimulationStartTime = new Date();
                    document.getElementById('startStimBtn').disabled = true;
                    document.getElementById('stopStimBtn').disabled = false;
                    document.getElementById('stimStatusDisplay').innerHTML = 
                        '<span class="badge bg-success">활성</span>';
                    updateStimulationDuration();
                }
            } else {
                if (stimulationStartTime) {
                    stimulationStartTime = null;
                    document.getElementById('startStimBtn').disabled = false;
                    document.getElementById('stopStimBtn').disabled = true;
                    document.getElementById('stimStatusDisplay').innerHTML = 
                        '<span class="badge bg-secondary">비활성</span>';
                    document.getElementById('stimDurationDisplay').textContent = "지속시간: 00:00:00";
                }
            }

            // Update parameters
            document.getElementById('frequencyDisplay').textContent = data.parameters.frequency;
            document.getElementById('amplitudeDisplay').textContent = data.parameters.amplitude.toFixed(1);
            document.getElementById('pulseWidthDisplay').textContent = data.parameters.pulse_width;
            document.getElementById('dutyCycleDisplay').textContent = data.parameters.duty_cycle;

            // Update recovery metrics
            updateRecoveryMetrics(data.recovery_metrics);
        })
        .catch(error => console.error('Error updating system state:', error));
}

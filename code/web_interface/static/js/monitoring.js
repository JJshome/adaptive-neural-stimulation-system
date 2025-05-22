/**
 * Monitoring JavaScript for Adaptive Neural Stimulation System
 * 
 * This script handles the monitoring page's functionality including:
 * - Advanced signal visualization
 * - Signal filtering and analysis
 * - Spectral analysis
 * - Event management
 * - Channel control
 */

// Global variables
let signalChartLarge = null;
let spectrumChart = null;
let spectrogramChart = null;
let updateInterval = null;
let isRecording = false;
let recordingStartTime = null;
let recordedData = [];
let eventModal = null;
let selectedChannels = [true, true, true, true]; // All channels initially selected

// Chart colors
const chartColors = {
    channel1: 'rgba(54, 162, 235, 1)',
    channel2: 'rgba(255, 99, 132, 1)',
    channel3: 'rgba(75, 192, 192, 1)',
    channel4: 'rgba(255, 159, 64, 1)'
};

// Filter settings
let filterSettings = {
    type: 'lowpass',
    frequency: 50
};

// Display settings
let displaySettings = {
    timeWindow: 10, // seconds
    amplitudeScale: 1.0,
    autoScale: true
};

// Initialize the monitoring page when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initializeCharts();
    
    // Initialize UI event listeners
    initializeEventListeners();
    
    // Initialize event modal
    eventModal = new bootstrap.Modal(document.getElementById('eventModal'));
    
    // Start periodic updates
    startPeriodicUpdates();
});

/**
 * Initialize all charts for the monitoring page
 */
function initializeCharts() {
    // Main signal chart
    const signalCtx = document.getElementById('signalChartLarge').getContext('2d');
    signalChartLarge = new Chart(signalCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 1000}, (_, i) => i),
            datasets: [
                {
                    label: 'Channel 1',
                    data: Array(1000).fill(0),
                    borderColor: chartColors.channel1,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Channel 2',
                    data: Array(1000).fill(0),
                    borderColor: chartColors.channel2,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Channel 3',
                    data: Array(1000).fill(0),
                    borderColor: chartColors.channel3,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Channel 4',
                    data: Array(1000).fill(0),
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
                        text: 'Time (ms)'
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        callback: function(value, index, values) {
                            return value * (displaySettings.timeWindow * 1000 / 1000);
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amplitude (µV)'
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
                    intersect: false,
                    callbacks: {
                        title: function(tooltipItems) {
                            const idx = tooltipItems[0].dataIndex;
                            const timeMs = idx * (displaySettings.timeWindow * 1000 / 1000);
                            return `Time: ${timeMs.toFixed(0)} ms`;
                        }
                    }
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            mode: 'vertical',
                            scaleID: 'x',
                            value: 0,
                            borderColor: 'rgba(255, 0, 0, 0.5)',
                            borderWidth: 2,
                            label: {
                                content: 'Stimulation Start',
                                enabled: true,
                                position: 'top'
                            }
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });

    // Spectrum analysis chart
    const spectrumCtx = document.getElementById('spectrumChart').getContext('2d');
    spectrumChart = new Chart(spectrumCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 100}, (_, i) => i),
            datasets: [
                {
                    label: 'Channel 1',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel1,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 1.5,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Channel 2',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel2,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 1.5,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Channel 3',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel3,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 1.5,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Channel 4',
                    data: Array(100).fill(0),
                    borderColor: chartColors.channel4,
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    borderWidth: 1.5,
                    fill: true,
                    pointRadius: 0
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
                        text: 'Frequency (Hz)'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Power'
                    },
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: false,
                    text: 'Frequency Spectrum'
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
    
    // Create a simple spectrogram visualization using canvas
    initializeSpectrogram();
}

/**
 * Initialize spectrogram visualization
 */
function initializeSpectrogram() {
    const canvas = document.getElementById('spectrogramChart');
    const ctx = canvas.getContext('2d');
    
    // Clear the canvas
    ctx.fillStyle = 'rgb(0, 0, 0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw placeholder text
    ctx.font = '14px Arial';
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.fillText('Spectrogram visualization will appear here', canvas.width / 2, canvas.height / 2);
    
    // In a real implementation, this would use WebGL or a more sophisticated rendering approach
}

/**
 * Initialize event listeners for UI controls
 */
function initializeEventListeners() {
    // Time window selection
    document.getElementById('timeWindowSelect').addEventListener('change', function() {
        displaySettings.timeWindow = parseInt(this.value);
        updateSignalChart();
    });
    
    // Amplitude scale selection
    document.getElementById('amplitudeScaleSelect').addEventListener('change', function() {
        displaySettings.amplitudeScale = parseFloat(this.value);
        updateSignalChart();
    });
    
    // Filter type selection
    document.getElementById('filterTypeSelect').addEventListener('change', function() {
        filterSettings.type = this.value;
        updateSignalChart();
    });
    
    // Filter frequency input
    document.getElementById('filterFreqInput').addEventListener('change', function() {
        filterSettings.frequency = parseFloat(this.value);
        updateSignalChart();
    });
    
    // Auto scale toggle
    document.getElementById('autoScaleToggle').addEventListener('change', function() {
        displaySettings.autoScale = this.checked;
        updateSignalChart();
    });
    
    // Channel visibility checkboxes
    for (let i = 1; i <= 4; i++) {
        document.getElementById(`channel${i}Check`).addEventListener('change', function() {
            selectedChannels[i-1] = this.checked;
            signalChartLarge.data.datasets[i-1].hidden = !this.checked;
            spectrumChart.data.datasets[i-1].hidden = !this.checked;
            signalChartLarge.update();
            spectrumChart.update();
        });
    }
    
    // Start stimulation button
    document.getElementById('startStimBtn').addEventListener('click', function() {
        fetch('/api/start-stimulation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI
                document.getElementById('startStimBtn').disabled = true;
                document.getElementById('stopStimBtn').disabled = false;
                
                // Add event to the table
                addEventToTable('stimulation_start', '자극 시작');
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
                document.getElementById('startStimBtn').disabled = false;
                document.getElementById('stopStimBtn').disabled = true;
                
                // Add event to the table
                addEventToTable('stimulation_stop', '자극 중지');
            }
        })
        .catch(error => console.error('Error stopping stimulation:', error));
    });
    
    // Recording controls
    document.getElementById('recordBtn').addEventListener('click', function() {
        if (!isRecording) {
            // Start recording
            isRecording = true;
            recordingStartTime = new Date();
            recordedData = [];
            this.textContent = '기록 중지';
            this.classList.remove('btn-outline-primary');
            this.classList.add('btn-danger');
            document.getElementById('saveRecordingBtn').disabled = true;
            
            // Add recording start event
            addEventToTable('recording_start', '신호 기록 시작');
        } else {
            // Stop recording
            isRecording = false;
            this.textContent = '기록 시작';
            this.classList.remove('btn-danger');
            this.classList.add('btn-outline-primary');
            document.getElementById('saveRecordingBtn').disabled = false;
            
            // Add recording stop event
            addEventToTable('recording_stop', '신호 기록 중지');
        }
    });
    
    // Save recording button
    document.getElementById('saveRecordingBtn').addEventListener('click', function() {
        if (recordedData.length === 0) {
            alert('저장할 기록이 없습니다.');
            return;
        }
        
        // In a real implementation, this would save the data to a file
        // For this demo, we'll just show an alert
        alert(`${recordedData.length}개의 데이터 포인트가 저장되었습니다.`);
        this.disabled = true;
    });
    
    // Add event button
    document.getElementById('addEventBtn').addEventListener('click', function() {
        eventModal.show();
    });
    
    // Save event button
    document.getElementById('saveEventBtn').addEventListener('click', function() {
        const eventType = document.getElementById('eventTypeSelect').value;
        const description = document.getElementById('eventDescription').value;
        
        if (description.trim() === '') {
            alert('이벤트 설명을 입력해주세요.');
            return;
        }
        
        addEventToTable(eventType, description);
        eventModal.hide();
        
        // Clear the form
        document.getElementById('eventDescription').value = '';
    });
}

/**
 * Add an event to the events table
 */
function addEventToTable(eventType, description) {
    const eventsTable = document.getElementById('eventsTable');
    const now = new Date();
    const timeString = now.toTimeString().split(' ')[0];
    
    // Map event type to display name and badge color
    const eventTypeMap = {
        'stimulation_start': { name: '자극 시작', color: 'info' },
        'stimulation_stop': { name: '자극 중지', color: 'secondary' },
        'parameter_change': { name: '파라미터 변경', color: 'warning' },
        'observation': { name: '관찰 사항', color: 'primary' },
        'artifact': { name: '아티팩트', color: 'danger' },
        'recording_start': { name: '기록 시작', color: 'success' },
        'recording_stop': { name: '기록 중지', color: 'dark' },
        'custom': { name: '기타', color: 'light' }
    };
    
    const eventInfo = eventTypeMap[eventType] || { name: eventType, color: 'secondary' };
    
    // Create a new row
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${timeString}</td>
        <td><span class="badge bg-${eventInfo.color}">${eventInfo.name}</span></td>
        <td>${description}</td>
        <td>
            <button class="btn btn-sm btn-outline-secondary view-event-btn">보기</button>
            <button class="btn btn-sm btn-outline-danger delete-event-btn">삭제</button>
        </td>
    `;
    
    // Add event listeners for the buttons
    row.querySelector('.view-event-btn').addEventListener('click', function() {
        alert(`Event Details:\nTime: ${timeString}\nType: ${eventInfo.name}\nDescription: ${description}`);
    });
    
    row.querySelector('.delete-event-btn').addEventListener('click', function() {
        if (confirm('이 이벤트를 삭제하시겠습니까?')) {
            row.remove();
        }
    });
    
    // Add to the table
    eventsTable.prepend(row);
}

/**
 * Start periodic data updates
 */
function startPeriodicUpdates() {
    // Update signal data every 200ms
    setInterval(updateSignalData, 200);
    
    // Update spectrum data every 1000ms
    setInterval(updateSpectrumData, 1000);
    
    // Update signal statistics every 2000ms
    setInterval(updateSignalStatistics, 2000);
}

/**
 * Update the signal chart with new data
 */
function updateSignalData() {
    fetch('/api/signal-data')
        .then(response => response.json())
        .then(data => {
            const signalData = data.data;
            
            // If recording, store the data
            if (isRecording) {
                recordedData.push({
                    timestamp: new Date(),
                    data: signalData
                });
            }
            
            // Apply any filtering if needed
            const filteredData = applyFilter(signalData, filterSettings);
            
            // Update chart data for each channel
            for (let channel = 0; channel < 4; channel++) {
                // Extract channel data
                const channelData = filteredData.map(sample => sample[channel]);
                
                // Apply amplitude scaling
                const scaledData = channelData.map(value => value * displaySettings.amplitudeScale);
                
                // Update chart
                signalChartLarge.data.datasets[channel].data = scaledData;
            }
            
            // Update chart scales if auto-scaling is enabled
            if (displaySettings.autoScale) {
                // Find min and max values across all channels
                let min = Infinity;
                let max = -Infinity;
                
                for (let channel = 0; channel < 4; channel++) {
                    if (selectedChannels[channel]) {
                        const data = signalChartLarge.data.datasets[channel].data;
                        const channelMin = Math.min(...data);
                        const channelMax = Math.max(...data);
                        
                        if (channelMin < min) min = channelMin;
                        if (channelMax > max) max = channelMax;
                    }
                }
                
                // Add some padding
                const padding = (max - min) * 0.1;
                signalChartLarge.options.scales.y.min = min - padding;
                signalChartLarge.options.scales.y.max = max + padding;
            }
            
            // Update chart
            signalChartLarge.update();
            
            // Update spectrogram (in a real implementation, this would use a proper algorithm)
            updateSpectrogram(filteredData);
        })
        .catch(error => console.error('Error updating signal data:', error));
}

/**
 * Apply signal filtering based on settings
 */
function applyFilter(signalData, settings) {
    // This is a simplified implementation
    // In a real system, proper digital signal processing would be used
    
    if (settings.type === 'none') {
        return signalData;
    }
    
    // For demonstration, we'll just apply a simple moving average filter
    const filteredData = [];
    const windowSize = 5; // Simple filter window
    
    for (let i = 0; i < signalData.length; i++) {
        const sample = [];
        
        for (let channel = 0; channel < signalData[0].length; channel++) {
            let sum = 0;
            let count = 0;
            
            for (let j = Math.max(0, i - windowSize); j <= Math.min(signalData.length - 1, i + windowSize); j++) {
                sum += signalData[j][channel];
                count++;
            }
            
            sample.push(sum / count);
        }
        
        filteredData.push(sample);
    }
    
    return filteredData;
}

/**
 * Update the spectrum chart with frequency analysis
 */
function updateSpectrumData() {
    fetch('/api/signal-data')
        .then(response => response.json())
        .then(data => {
            const signalData = data.data;
            
            // For each channel, compute a simple "spectrum"
            // In a real implementation, this would use a proper FFT algorithm
            for (let channel = 0; channel < 4; channel++) {
                // Extract channel data
                const channelData = signalData.map(sample => sample[channel]);
                
                // Generate a dummy spectrum (for demonstration)
                const spectrum = generateDummySpectrum(channelData, 100);
                
                // Update spectrum chart
                spectrumChart.data.datasets[channel].data = spectrum;
            }
            
            // Update chart
            spectrumChart.update();
        })
        .catch(error => console.error('Error updating spectrum data:', error));
}

/**
 * Generate a dummy spectrum for demonstration
 */
function generateDummySpectrum(signalData, numPoints) {
    // This is just a placeholder that generates a plausible-looking spectrum
    // In a real implementation, this would use a proper FFT algorithm
    
    const spectrum = [];
    
    // Calculate signal energy (simple RMS)
    const energy = Math.sqrt(signalData.reduce((sum, val) => sum + val * val, 0) / signalData.length);
    
    // Generate a shape that peaks in the middle and falls off at higher frequencies
    for (let i = 0; i < numPoints; i++) {
        // Create a peak around 10-20Hz (assuming 100Hz spans 0-50Hz)
        const freqComponent = energy * Math.exp(-Math.pow(i - 15, 2) / 50);
        
        // Add some random variation
        const randomFactor = 0.2 + Math.random() * 0.2;
        
        spectrum.push(freqComponent * randomFactor);
    }
    
    return spectrum;
}

/**
 * Update the spectrogram visualization
 */
function updateSpectrogram(signalData) {
    const canvas = document.getElementById('spectrogramChart');
    const ctx = canvas.getContext('2d');
    
    // In a real implementation, this would compute a proper spectrogram
    // For this demo, we'll just create a simple visualization
    
    // Shift the existing image to the left
    const imageData = ctx.getImageData(1, 0, canvas.width - 1, canvas.height);
    ctx.putImageData(imageData, 0, 0);
    
    // Clear the right-most column
    ctx.fillStyle = 'rgb(0, 0, 0)';
    ctx.fillRect(canvas.width - 1, 0, 1, canvas.height);
    
    // Draw a new column of data on the right edge
    // Average the 4 channels
    const avgData = [];
    for (let i = 0; i < signalData.length; i++) {
        let sum = 0;
        for (let channel = 0; channel < 4; channel++) {
            if (selectedChannels[channel]) {
                sum += signalData[i][channel];
            }
        }
        avgData.push(sum / 4);
    }
    
    // Divide the height into segments
    const segmentSize = Math.floor(avgData.length / canvas.height);
    
    // Draw each segment
    for (let y = 0; y < canvas.height; y++) {
        // Get the average value for this segment
        let segmentAvg = 0;
        const startIdx = y * segmentSize;
        const endIdx = Math.min(startIdx + segmentSize, avgData.length);
        
        for (let i = startIdx; i < endIdx; i++) {
            segmentAvg += Math.abs(avgData[i]);
        }
        segmentAvg /= (endIdx - startIdx);
        
        // Map the value to a color intensity (0-255)
        const intensity = Math.min(255, Math.floor(segmentAvg * 1000));
        
        // Use a heatmap-like color scheme
        let r = 0, g = 0, b = 0;
        if (intensity < 85) {
            r = 0;
            g = intensity * 3;
            b = 255;
        } else if (intensity < 170) {
            r = (intensity - 85) * 3;
            g = 255;
            b = 255 - (intensity - 85) * 3;
        } else {
            r = 255;
            g = 255 - (intensity - 170) * 3;
            b = 0;
        }
        
        // Draw the pixel
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(canvas.width - 1, canvas.height - y - 1, 1, 1);
    }
}

/**
 * Update signal statistics
 */
function updateSignalStatistics() {
    fetch('/api/signal-data')
        .then(response => response.json())
        .then(data => {
            const signalData = data.data;
            
            // Calculate statistics for each channel
            for (let channel = 0; channel < 4; channel++) {
                // Extract channel data
                const channelData = signalData.map(sample => sample[channel]);
                
                // Calculate RMS
                const rms = Math.sqrt(channelData.reduce((sum, val) => sum + val * val, 0) / channelData.length);
                document.getElementById(`rmsChannel${channel+1}`).textContent = `${rms.toFixed(2)} µV`;
                
                // For demonstration, generate plausible-looking values for other stats
                // In a real implementation, these would be properly calculated
                
                // Dominant frequency - random value between 10-20Hz
                const dominantFreq = (10 + Math.random() * 10).toFixed(1);
                document.getElementById(`freqChannel${channel+1}`).textContent = `${dominantFreq} Hz`;
                
                // SNR - random value between 7-10dB
                const snr = (7 + Math.random() * 3).toFixed(1);
                document.getElementById(`snrChannel${channel+1}`).textContent = `${snr} dB`;
                
                // Sample entropy - random value between 1.0-1.5
                const entropy = (1.0 + Math.random() * 0.5).toFixed(2);
                document.getElementById(`entropyChannel${channel+1}`).textContent = entropy;
            }
        })
        .catch(error => console.error('Error updating signal statistics:', error));
}

/**
 * Update the signal chart parameters based on UI settings
 */
function updateSignalChart() {
    // Update chart scales based on display settings
    const samples = signalChartLarge.data.datasets[0].data.length;
    
    // Update x-axis ticks to show time in ms
    signalChartLarge.options.scales.x.ticks.callback = function(value, index, values) {
        return Math.floor(value * (displaySettings.timeWindow * 1000 / samples));
    };
    
    // Update tooltip to show time in ms
    signalChartLarge.options.plugins.tooltip.callbacks.title = function(tooltipItems) {
        const idx = tooltipItems[0].dataIndex;
        const timeMs = idx * (displaySettings.timeWindow * 1000 / samples);
        return `Time: ${timeMs.toFixed(0)} ms`;
    };
    
    // Disable auto-scaling if not selected
    if (!displaySettings.autoScale) {
        // Reset to default scale
        signalChartLarge.options.scales.y.min = -2 * displaySettings.amplitudeScale;
        signalChartLarge.options.scales.y.max = 2 * displaySettings.amplitudeScale;
    }
    
    signalChartLarge.update();
}

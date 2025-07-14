// MyMind - Comprehensive UI JavaScript
let currentSessionId = null;
let radarChart = null;
let barChart = null;

// Tab switching functionality
function switchTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    
    // Add active class to clicked tab
    event.target.classList.add('active');
}

// Utility functions
function showLoading(elementId) {
    document.getElementById(elementId).style.display = 'block';
}

function hideLoading(elementId) {
    document.getElementById(elementId).style.display = 'none';
}

function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function hideError(elementId) {
    const errorElement = document.getElementById(elementId);
    errorElement.style.display = 'none';
    errorElement.textContent = '';
}

function showSuccess(elementId, message) {
    const successElement = document.getElementById(elementId);
    successElement.textContent = message;
    successElement.style.display = 'block';
}

function hideSuccess(elementId) {
    const successElement = document.getElementById(elementId);
    successElement.style.display = 'none';
    successElement.textContent = '';
}

// Text Analysis Functions
async function analyzeText() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first.');
        return;
    }

    // Reset UI
    document.getElementById('text-results').style.display = 'none';
    hideError('text-error');
    showLoading('text-loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/simple_analyze', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Analysis failed.');
        }

        displayTextResults(data);

    } catch (error) {
        showError('text-error', 'An error occurred: ' + error.message);
    } finally {
        hideLoading('text-loading');
    }
}

function displayTextResults(data) {
    const resultsContainer = document.getElementById('text-results');
    
    let needsSection = '';
    if (data.needs_analysis) {
        needsSection = `
            <div class="result-box">
                <h3>🎯 Needs Analysis</h3>
                <pre>${JSON.stringify(data.needs_analysis, null, 2)}</pre>
            </div>
        `;
    }
    
    resultsContainer.innerHTML = `
        <div class="result-box">
            <h3>🔑 Keywords & Sentiment</h3>
            <pre>${JSON.stringify(data.keywords_analysis, null, 2)}</pre>
        </div>
        <div class="result-box">
            <h3>🧠 Cognitive Distortions (CBT)</h3>
            <pre>${JSON.stringify(data.therapeutic_analysis.cognitive_distortions, null, 2)}</pre>
        </div>
        <div class="result-box">
            <h3>🎭 Schema Patterns</h3>
            <pre>${JSON.stringify(data.therapeutic_analysis.schema_analysis, null, 2)}</pre>
        </div>
        ${needsSection}
    `;
    
    resultsContainer.style.display = 'grid';
}

// Audio Session Functions
async function uploadAudio() {
    const fileInput = document.getElementById('audioInput');
    const clientIdInput = document.getElementById('clientId');
    const numSpeakersSelect = document.getElementById('numSpeakers');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an audio file first.');
        return;
    }

    hideError('audio-error');
    hideSuccess('audio-success');
    showLoading('audio-loading');

    const formData = new FormData();
    formData.append('file', file);
    
    if (clientIdInput.value.trim()) {
        formData.append('client_id', clientIdInput.value.trim());
    }
    
    if (numSpeakersSelect.value) {
        formData.append('num_speakers', numSpeakersSelect.value);
    }

    try {
        const response = await fetch('/api/preprocess/upload-audio', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed.');
        }

        currentSessionId = data.session_id;
        showSuccess('audio-success', `✅ Audio uploaded successfully! Session ID: ${currentSessionId}`);
        document.getElementById('processCompleteButton').disabled = false;

    } catch (error) {
        showError('audio-error', 'Upload failed: ' + error.message);
    } finally {
        hideLoading('audio-loading');
    }
}

async function processComplete() {
    if (!currentSessionId) {
        alert('Please upload an audio file first.');
        return;
    }

    const numSpeakers = document.getElementById('numSpeakers').value;
    
    hideError('audio-error');
    hideSuccess('audio-success');
    showLoading('audio-loading');

    try {
        // Start complete processing (transcription + keywords + needs)
        const response = await fetch(`/api/preprocess/process-complete/${currentSessionId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                num_speakers: numSpeakers ? parseInt(numSpeakers) : null,
                chunk_size: 3
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Processing failed.');
        }

        showSuccess('audio-success', '🚀 Complete processing started! You can check progress in Session Management.');
        
        // Also trigger needs extraction
        setTimeout(async () => {
            try {
                await fetch(`/api/preprocess/needs/${currentSessionId}`, {
                    method: 'POST',
                });
            } catch (e) {
                console.warn('Needs extraction failed:', e);
            }
        }, 5000); // Wait 5 seconds for transcription to start

    } catch (error) {
        showError('audio-error', 'Processing failed: ' + error.message);
    } finally {
        hideLoading('audio-loading');
    }
}

// Session Management Functions
async function refreshSessions() {
    showLoading('sessions-loading');
    hideError('sessions-error');

    try {
        // Since we don't have a sessions list endpoint, we'll create a placeholder
        // In a real implementation, you'd fetch from /api/sessions
        showSuccess('sessions-error', 'Session management will be implemented when session list endpoint is available. For now, use the processing status endpoint with session IDs.');
        
    } catch (error) {
        showError('sessions-error', 'Failed to load sessions: ' + error.message);
    } finally {
        hideLoading('sessions-loading');
    }
}

async function checkSessionStatus(sessionId) {
    try {
        const response = await fetch(`/api/preprocess/status/${sessionId}`);
        const data = await response.json();
        
        if (response.ok) {
            return data;
        }
        return null;
    } catch (error) {
        console.error('Failed to check session status:', error);
        return null;
    }
}

// Needs Profiling Functions
async function generateProfile() {
    const clientId = document.getElementById('profileClientId').value.trim();
    
    if (!clientId) {
        alert('Please enter a Client ID.');
        return;
    }

    hideError('profile-error');
    hideSuccess('profile-success');
    showLoading('profile-loading');
    document.getElementById('profile-results').style.display = 'none';

    try {
        const response = await fetch(`/api/profiling/clients/${clientId}/analyze-needs`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_count: 10
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Profile generation failed.');
        }

        showSuccess('profile-success', '✅ Profile analysis started! Click "View Dashboard" to see results once processing is complete.');

    } catch (error) {
        showError('profile-error', 'Profile generation failed: ' + error.message);
    } finally {
        hideLoading('profile-loading');
    }
}

async function viewDashboard() {
    const clientId = document.getElementById('profileClientId').value.trim();
    
    if (!clientId) {
        alert('Please enter a Client ID.');
        return;
    }

    hideError('profile-error');
    hideSuccess('profile-success');
    showLoading('profile-loading');

    try {
        const response = await fetch(`/api/profiling/clients/${clientId}/needs-dashboard`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Dashboard loading failed.');
        }

        displayDashboard(data);
        document.getElementById('profile-results').style.display = 'block';

    } catch (error) {
        showError('profile-error', 'Dashboard loading failed: ' + error.message);
    } finally {
        hideLoading('profile-loading');
    }
}

function displayDashboard(data) {
    // Display insights
    const insightsList = document.getElementById('insights-list');
    insightsList.innerHTML = '';
    data.life_segments.insights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        insightsList.appendChild(li);
    });

    // Display recommendations
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '<h4>Therapeutic Recommendations:</h4>';
    data.recommendations.forEach(rec => {
        const div = document.createElement('div');
        div.innerHTML = `
            <strong>${rec.intervention}</strong> (Priority: ${rec.priority})
            <br><small>${rec.description}</small>
        `;
        div.style.marginBottom = '10px';
        recommendationsDiv.appendChild(div);
    });

    // Create radar chart
    createRadarChart(data.visualization_data.radar_chart);
    
    // Create bar chart
    createBarChart(data.visualization_data.bar_chart);
}

function createRadarChart(chartData) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    if (radarChart) {
        radarChart.destroy();
    }
    
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        display: false
                    },
                    suggestedMin: -1,
                    suggestedMax: 1,
                    ticks: {
                        beginAtZero: true
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Life Segments Analysis'
                }
            }
        }
    });
}

function createBarChart(chartData) {
    const ctx = document.getElementById('barChart').getContext('2d');
    
    if (barChart) {
        barChart.destroy();
    }
    
    barChart = new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Needs Fulfillment Scores'
                }
            }
        }
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Text Analysis
    document.getElementById('analyzeButton').addEventListener('click', analyzeText);
    
    // Audio Sessions
    document.getElementById('uploadAudioButton').addEventListener('click', uploadAudio);
    document.getElementById('processCompleteButton').addEventListener('click', processComplete);
    
    // Session Management
    document.getElementById('refreshSessionsButton').addEventListener('click', refreshSessions);
    
    // Needs Profiling
    document.getElementById('generateProfileButton').addEventListener('click', generateProfile);
    document.getElementById('viewDashboardButton').addEventListener('click', viewDashboard);
});
// Global variables
let currentView = 'home';
let currentClient = null;
let currentSession = null;
let clients = [];
let sessions = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    loadDashboardData();
    showView('home');
});

// Text Analysis Functions (from main app)
async function analyzeText() {
    const fileInput = document.getElementById('text-file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('Please select a .txt file first using the "Choose File" button.', 'error');
        return;
    }

    if (file.type !== 'text/plain') {
        showMessage('Please select a valid .txt file.', 'error');
        return;
    }

    // Clear previous results
    const resultsContainer = document.getElementById('text-analysis-results');
    resultsContainer.style.display = 'none';
    resultsContainer.innerHTML = '';

    showLoading(true);
    hideMessage();

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
        showMessage('Analysis failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayTextResults(data) {
    const resultsContainer = document.getElementById('text-analysis-results');
    
    let needsSection = '';
    if (data.needs_analysis) {
        needsSection = `
            <div class="result-section">
                <h4>🎯 Needs Analysis</h4>
                <pre>${JSON.stringify(data.needs_analysis, null, 2)}</pre>
            </div>
        `;
    }
    
    resultsContainer.innerHTML = `
        <div class="result-section">
            <h4>🔑 Keywords & Sentiment</h4>
            <pre>${JSON.stringify(data.keywords_analysis, null, 2)}</pre>
        </div>
        <div class="result-section">
            <h4>🧠 Cognitive Distortions (CBT)</h4>
            <pre>${JSON.stringify(data.therapeutic_analysis.cognitive_distortions, null, 2)}</pre>
        </div>
        <div class="result-section">
            <h4>🎭 Schema Patterns</h4>
            <pre>${JSON.stringify(data.therapeutic_analysis.schema_analysis, null, 2)}</pre>
        </div>
        ${needsSection}
    `;
    
    resultsContainer.style.display = 'block';
}

// Audio Processing Functions (from main app)
async function processAudioFile() {
    const fileInput = document.getElementById('audio-file');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage('Please select an audio file first.', 'error');
        return;
    }

    if (!currentSession) {
        showMessage('Please create a session first', 'error');
        return;
    }

    showLoading(true);
    hideMessage();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('client_id', currentClient.id);

    try {
        const response = await fetch('/api/preprocess/upload-audio', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed.');
        }

        currentSession.audio_session_id = data.session_id;
        showMessage('✅ Audio uploaded successfully! Session ID: ' + data.session_id, 'success');
        
        // Enable complete processing button
        document.getElementById('process-complete-btn').disabled = false;

    } catch (error) {
        showMessage('Upload failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

async function processComplete() {
    if (!currentSession || !currentSession.audio_session_id) {
        showMessage('Please upload an audio file first.', 'error');
        return;
    }

    showLoading(true);
    hideMessage();

    try {
        const response = await fetch(`/api/preprocess/process-complete/${currentSession.audio_session_id}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                num_speakers: 2,
                chunk_size: 3
            }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Processing failed.');
        }

        showMessage('🚀 Complete processing started! Check progress in session management.', 'success');
        
        // Also trigger needs extraction
        setTimeout(async () => {
            try {
                await fetch(`/api/preprocess/needs/${currentSession.audio_session_id}`, {
                    method: 'POST',
                });
            } catch (e) {
                console.warn('Needs extraction failed:', e);
            }
        }, 5000);

    } catch (error) {
        showMessage('Processing failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

// Enhanced Client Management Functions
async function loadClientsEnhanced() {
    showLoading(true);
    hideMessage();

    try {
        const response = await fetch('/api/profiling/clients');
        const clients = await response.json();

        if (!response.ok) {
            throw new Error('Failed to load clients');
        }

        displayClientsEnhanced(clients);

    } catch (error) {
        showMessage('Failed to load clients: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayClientsEnhanced(clients) {
    const container = document.getElementById('clients-enhanced-list');
    
    if (clients.length === 0) {
        container.innerHTML = '<p>No clients found.</p>';
        return;
    }

    container.innerHTML = '<h4>📋 Available Clients (click ID to copy):</h4>';
    
    clients.forEach(client => {
        const clientDiv = document.createElement('div');
        clientDiv.className = 'client-item';
        clientDiv.innerHTML = `
            <div>
                <span class="client-id" onclick="copyToClipboard('${client.id}', this)" title="Click to copy ID">${client.id}</span>
                <strong>${client.name}</strong>
                ${client.email ? `(${client.email})` : ''}
                <span style="float: right; color: #6c757d;">${client.session_count} sessions</span>
            </div>
            <div style="font-size: 12px; color: #6c757d; margin-top: 5px;">
                Created: ${new Date(client.created_at).toLocaleDateString()}
            </div>
            <div class="sessions-container" id="sessions-${client.id}" style="display: none;">
                <button onclick="loadClientSessionsEnhanced('${client.id}')" class="btn btn-sm" style="font-size: 12px;">📊 View Sessions</button>
            </div>
        `;
        
        clientDiv.addEventListener('click', (e) => {
            if (e.target.classList.contains('client-id')) return;
            toggleClientSessionsEnhanced(client.id);
        });
        
        container.appendChild(clientDiv);
    });
}

async function loadClientSessionsEnhanced(clientId) {
    const sessionsContainer = document.getElementById(`sessions-${clientId}`);
    sessionsContainer.innerHTML = '<div class="loading">Loading sessions...</div>';

    try {
        const response = await fetch(`/api/profiling/clients/${clientId}/sessions`);
        const sessions = await response.json();

        if (!response.ok) {
            throw new Error(sessions.detail || 'Failed to load sessions');
        }

        displayClientSessionsEnhanced(clientId, sessions);

    } catch (error) {
        sessionsContainer.innerHTML = `<div class="error">Failed to load sessions: ${error.message}</div>`;
    }
}

function displayClientSessionsEnhanced(clientId, sessions) {
    const sessionsContainer = document.getElementById(`sessions-${clientId}`);
    
    if (sessions.length === 0) {
        sessionsContainer.innerHTML = '<p style="font-style: italic;">No sessions found for this client.</p>';
        return;
    }

    let sessionsHtml = '<h5>📊 Sessions (click ID to copy):</h5>';
    
    sessions.forEach(session => {
        sessionsHtml += `
            <div class="session-item">
                <span class="session-id" onclick="copyToClipboard('${session.id}', this)" title="Click to copy ID">${session.id}</span>
                <strong>${session.title}</strong>
                <span class="session-status status-${session.status}">${session.status}</span>
                <div style="font-size: 11px; color: #6c757d; margin-top: 3px;">
                    ${new Date(session.created_at).toLocaleString()}
                </div>
                ${session.notes ? `<div style="font-size: 12px; margin-top: 5px;">${session.notes}</div>` : ''}
            </div>
        `;
    });
    
    sessionsContainer.innerHTML = sessionsHtml;
}

function toggleClientSessionsEnhanced(clientId) {
    const sessionsContainer = document.getElementById(`sessions-${clientId}`);
    if (sessionsContainer.style.display === 'none') {
        sessionsContainer.style.display = 'block';
        loadClientSessionsEnhanced(clientId);
    } else {
        sessionsContainer.style.display = 'none';
    }
}

// Needs Profiling Functions (from main app)
async function generateNeedsProfile() {
    const clientId = document.getElementById('profile-client-id').value.trim();
    
    if (!clientId) {
        showMessage('Please enter a Client ID.', 'error');
        return;
    }

    showLoading(true);
    hideMessage();

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

        showMessage('✅ Profile analysis started! Click "View Dashboard" to see results once processing is complete.', 'success');

    } catch (error) {
        showMessage('Profile generation failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

async function viewNeedsDashboard() {
    const clientId = document.getElementById('profile-client-id').value.trim();
    
    if (!clientId) {
        showMessage('Please enter a Client ID.', 'error');
        return;
    }

    showLoading(true);
    hideMessage();

    try {
        const response = await fetch(`/api/profiling/clients/${clientId}/needs-dashboard`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Dashboard loading failed.');
        }

        displayNeedsDashboard(data);
        document.getElementById('needs-dashboard-results').style.display = 'block';

    } catch (error) {
        showMessage('Dashboard loading failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayNeedsDashboard(data) {
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
    const ctx = document.getElementById('radar-chart').getContext('2d');
    
    if (window.radarChart) {
        window.radarChart.destroy();
    }
    
    window.radarChart = new Chart(ctx, {
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
    const ctx = document.getElementById('bar-chart').getContext('2d');
    
    if (window.barChart) {
        window.barChart.destroy();
    }
    
    window.barChart = new Chart(ctx, {
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

// Utility Functions
function copyToClipboard(text, element) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = element.textContent;
        element.textContent = 'Copied!';
        element.style.backgroundColor = '#28a745';
        element.style.color = 'white';
        
        setTimeout(() => {
            element.textContent = originalText;
            element.style.backgroundColor = '';
            element.style.color = '';
        }, 1000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
        showMessage('Failed to copy to clipboard', 'error');
    });
}

function hideMessage() {
    const messages = document.querySelectorAll('.success, .error');
    messages.forEach(msg => msg.remove());
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

// Setup event listeners
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const view = this.dataset.view;
            showView(view);
        });
    });

    // Forms
    document.getElementById('new-client-form').addEventListener('submit', handleNewClient);
    document.getElementById('new-session-form').addEventListener('submit', handleNewSession);
    document.getElementById('questionnaire-form').addEventListener('submit', handleQuestionnaire);
    document.getElementById('settings-form').addEventListener('submit', handleSettings);

    // File upload
    document.getElementById('audio-file').addEventListener('change', handleFileUpload);
    document.getElementById('text-file-input').addEventListener('change', handleTextFileSelection);

    // Chat
    document.getElementById('chat-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });

    // Drag and drop
    const uploadArea = document.getElementById('upload-area');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Text analysis drag and drop
    const textUploadArea = document.querySelector('#analysis-view .upload-area');
    if (textUploadArea) {
        textUploadArea.addEventListener('dragover', handleTextDragOver);
        textUploadArea.addEventListener('drop', handleTextDrop);
    }
}

// Navigation functions
function updateBreadcrumb(breadcrumbItems) {
    const breadcrumbElement = document.getElementById('breadcrumb-text');
    breadcrumbElement.innerHTML = '';
    
    breadcrumbItems.forEach((item, index) => {
        if (index > 0) {
            const separator = document.createElement('span');
            separator.textContent = ' > ';
            breadcrumbElement.appendChild(separator);
        }
        
        if (item.onclick) {
            const link = document.createElement('a');
            link.href = '#';
            link.textContent = item.text;
            link.onclick = (e) => {
                e.preventDefault();
                item.onclick();
            };
            breadcrumbElement.appendChild(link);
        } else {
            const span = document.createElement('span');
            span.textContent = item.text;
            breadcrumbElement.appendChild(span);
        }
    });
}

function showView(view, subview = null) {
    // Update navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelector(`[data-view="${view}"]`).classList.add('active');

    // Hide all views
    document.querySelectorAll('.view').forEach(v => {
        v.classList.add('hidden');
    });

    // Show specific view
    currentView = view;
    
    switch(view) {
        case 'home':
            document.getElementById('page-title').textContent = 'Dashboard';
            document.getElementById('breadcrumb-text').textContent = 'Home';
            document.getElementById('home-view').classList.remove('hidden');
            loadDashboardData();
            break;
        case 'clients':
            if (subview === 'new-client') {
                showNewClientForm();
            } else {
                document.getElementById('page-title').textContent = 'Clients';
                document.getElementById('breadcrumb-text').textContent = 'Clients';
                document.getElementById('clients-view').classList.remove('hidden');
                loadClients();
            }
            break;
        case 'calendar':
            document.getElementById('page-title').textContent = 'Calendar';
            document.getElementById('breadcrumb-text').textContent = 'Calendar';
            document.getElementById('calendar-view').classList.remove('hidden');
            loadCalendar();
            break;
        case 'analysis':
            document.getElementById('page-title').textContent = 'Text Analysis';
            document.getElementById('breadcrumb-text').textContent = 'Text Analysis';
            document.getElementById('analysis-view').classList.remove('hidden');
            break;
        case 'client-profiling':
            document.getElementById('page-title').textContent = 'Client Needs Profiling';
            document.getElementById('breadcrumb-text').textContent = 'Client > Needs Profiling';
            document.getElementById('client-profiling-view').classList.remove('hidden');
            break;
        case 'settings':
            document.getElementById('page-title').textContent = 'Settings';
            document.getElementById('breadcrumb-text').textContent = 'Settings';
            document.getElementById('settings-view').classList.remove('hidden');
            break;
    }
}

// Dashboard functions
async function loadDashboardData() {
    try {
        showLoading(true);
        
        // Load clients
        const clientsResponse = await fetch('/api/clients/');
        clients = await clientsResponse.json();
        
        // Load sessions
        const sessionsResponse = await fetch('/api/sessions/');
        sessions = await sessionsResponse.json();
        
        // Update stats
        updateDashboardStats();
        updateRecentClients();
        updateRecentSessions();
        
        showLoading(false);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showMessage('Error loading dashboard data', 'error');
        showLoading(false);
    }
}

function updateDashboardStats() {
    document.getElementById('total-clients').textContent = clients.length;
    document.getElementById('total-sessions').textContent = sessions.length;
    
    // Calculate sessions today
    const today = new Date().toISOString().split('T')[0];
    const sessionsToday = sessions.filter(session => 
        session.created_at.startsWith(today)
    ).length;
    document.getElementById('sessions-today').textContent = sessionsToday;
    
    // Calculate active clients (clients with sessions in last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    const activeClients = new Set();
    sessions.forEach(session => {
        if (new Date(session.created_at) > thirtyDaysAgo) {
            activeClients.add(session.client_id);
        }
    });
    document.getElementById('active-clients').textContent = activeClients.size;
}

function updateRecentClients() {
    const recentClients = clients.slice(0, 5);
    const container = document.getElementById('recent-clients');
    
    container.innerHTML = recentClients.map(client => `
        <div class="client-item" onclick="showClientProfile('${client.id}')">
            <div class="client-name">${client.name}</div>
            <div class="client-email">${client.email || 'No email'}</div>
            <div class="client-email">${client.session_count} sessions</div>
        </div>
    `).join('');
}

function updateRecentSessions() {
    const recentSessions = sessions.slice(0, 5);
    const container = document.getElementById('recent-sessions');
    
    container.innerHTML = recentSessions.map(session => `
        <div class="session-item" onclick="showSessionView('${session.id}')">
            <div class="session-title">${session.title || 'Untitled Session'}</div>
            <div class="session-date">${new Date(session.created_at).toLocaleDateString()}</div>
            <span class="session-status status-${session.status}">${session.status}</span>
        </div>
    `).join('');
}

// Client management functions
async function loadClients() {
    try {
        showLoading(true);
        const response = await fetch('/api/clients/');
        clients = await response.json();
        
        const container = document.getElementById('clients-list');
        container.innerHTML = clients.map(client => `
            <div class="client-item" onclick="showClientProfile('${client.id}')">
                <div class="client-name">${client.name}</div>
                <div class="client-email">${client.email || 'No email'}</div>
                <div class="client-email">
                    ${client.session_count} sessions • 
                    Created: ${new Date(client.created_at).toLocaleDateString()}
                </div>
            </div>
        `).join('');
        
        showLoading(false);
    } catch (error) {
        console.error('Error loading clients:', error);
        showMessage('Error loading clients', 'error');
        showLoading(false);
    }
}

function showNewClientForm() {
    document.getElementById('page-title').textContent = 'New Client';
    document.getElementById('breadcrumb-text').innerHTML = '<a href="#" onclick="showView(\'clients\')">Clients</a> > New Client';
    document.getElementById('new-client-view').classList.remove('hidden');
}

async function handleNewClient(e) {
    e.preventDefault();
    
    const name = document.getElementById('client-name').value;
    const email = document.getElementById('client-email').value;
    
    try {
        const response = await fetch('/api/clients/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                email: email || null
            })
        });
        
        if (response.ok) {
            showMessage('Client created successfully', 'success');
            document.getElementById('new-client-form').reset();
            showView('clients');
        } else {
            const error = await response.json();
            showMessage(error.detail || 'Error creating client', 'error');
        }
    } catch (error) {
        console.error('Error creating client:', error);
        showMessage('Error creating client', 'error');
    }
}

async function showClientProfile(clientId) {
    try {
        const response = await fetch(`/api/clients/${clientId}`);
        const client = await response.json();
        
        currentClient = client;
        
        // Hide all views first
        document.querySelectorAll('.view').forEach(v => {
            v.classList.add('hidden');
        });
        
        document.getElementById('page-title').textContent = client.name;
        document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > ${client.name}`;
        
        document.getElementById('client-profile-content').innerHTML = `
            <div class="grid">
                <div>
                    <h4>Client Information</h4>
                    <p><strong>Name:</strong> ${client.name}</p>
                    <p><strong>Email:</strong> ${client.email || 'Not provided'}</p>
                    <p><strong>Client ID:</strong> ${client.id}</p>
                    <p><strong>Created:</strong> ${new Date(client.created_at).toLocaleDateString()}</p>
                </div>
                <div>
                    <h4>Session Summary</h4>
                    <p><strong>Total Sessions:</strong> ${client.session_count}</p>
                    <p><strong>Last Updated:</strong> ${new Date(client.updated_at).toLocaleDateString()}</p>
                </div>
            </div>
        `;
        
        document.getElementById('client-profile-view').classList.remove('hidden');
    } catch (error) {
        console.error('Error loading client profile:', error);
        showMessage('Error loading client profile', 'error');
    }
}

function showClientStats() {
    // Hide all views first
    document.querySelectorAll('.view').forEach(v => {
        v.classList.add('hidden');
    });
    
    document.getElementById('page-title').textContent = `${currentClient.name} - Statistics`;
    document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${currentClient.id}')">${currentClient.name}</a> > Statistics`;
    document.getElementById('client-stats-view').classList.remove('hidden');
    
    // Load charts (placeholder for now)
    loadClientCharts();
}

function loadClientCharts() {
    // Sessions chart
    const sessionsCtx = document.getElementById('sessions-chart').getContext('2d');
    new Chart(sessionsCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Sessions',
                data: [2, 3, 1, 4, 2, 3],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Progress chart
    const progressCtx = document.getElementById('progress-chart').getContext('2d');
    new Chart(progressCtx, {
        type: 'radar',
        data: {
            labels: ['Mood', 'Anxiety', 'Relationships', 'Work', 'Self-esteem'],
            datasets: [{
                label: 'Progress',
                data: [65, 59, 80, 81, 56],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function showClientQuestionnaire() {
    // Hide all views first
    document.querySelectorAll('.view').forEach(v => {
        v.classList.add('hidden');
    });
    
    document.getElementById('page-title').textContent = `${currentClient.name} - Assessment`;
    document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${currentClient.id}')">${currentClient.name}</a> > Assessment`;
    document.getElementById('client-questionnaire-view').classList.remove('hidden');
}

async function handleQuestionnaire(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData);
    data.therapy_goals = document.getElementById('therapy-goals').value;
    
    // In a real application, this would save to the database
    console.log('Questionnaire data:', data);
    showMessage('Assessment saved successfully', 'success');
    showClientProfile(currentClient.id);
}

async function showClientSessions() {
    try {
        const response = await fetch(`/api/clients/${currentClient.id}/sessions`);
        const sessions = await response.json();
        
        document.getElementById('page-title').textContent = `${currentClient.name} - Sessions`;
        document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${currentClient.id}')">${currentClient.name}</a> > Sessions`;
        
        const container = document.getElementById('client-sessions-list');
        container.innerHTML = sessions.map(session => `
            <div class="session-item" onclick="showSessionView('${session.id}')">
                <div class="session-title">${session.title || 'Untitled Session'}</div>
                <div class="session-date">${new Date(session.created_at).toLocaleDateString()}</div>
                <span class="session-status status-${session.status}">${session.status}</span>
                <div style="margin-top: 10px;">
                    <button class="btn btn-sm" onclick="event.stopPropagation(); editSession('${session.id}')">Edit</button>
                    <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteSession('${session.id}')">Delete</button>
                </div>
            </div>
        `).join('');
        
        document.getElementById('client-sessions-view').classList.remove('hidden');
    } catch (error) {
        console.error('Error loading client sessions:', error);
        showMessage('Error loading client sessions', 'error');
    }
}

function showNewSessionForm() {
    document.getElementById('page-title').textContent = 'New Session';
    document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${currentClient.id}')">${currentClient.name}</a> > <a href="#" onclick="showClientSessions()">Sessions</a> > New Session`;
    document.getElementById('new-session-view').classList.remove('hidden');
}

async function handleNewSession(e) {
    e.preventDefault();
    
    const title = document.getElementById('session-title').value;
    const notes = document.getElementById('session-notes').value;
    
    try {
        const response = await fetch(`/api/clients/${currentClient.id}/sessions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title || null,
                notes: notes || null
            })
        });
        
        if (response.ok) {
            const session = await response.json();
            showMessage('Session created successfully', 'success');
            document.getElementById('new-session-form').reset();
            showSessionView(session.id);
        } else {
            const error = await response.json();
            showMessage(error.detail || 'Error creating session', 'error');
        }
    } catch (error) {
        console.error('Error creating session:', error);
        showMessage('Error creating session', 'error');
    }
}

async function showSessionView(sessionId) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}`);
        const session = await response.json();
        
        currentSession = session;
        
        document.getElementById('page-title').textContent = session.title || 'Session';
        document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${session.client_id}')">${session.client_name}</a> > <a href="#" onclick="showClientSessions()">Sessions</a> > ${session.title || 'Session'}`;
        
        document.getElementById('session-content').innerHTML = `
            <div class="grid">
                <div>
                    <h4>Session Information</h4>
                    <p><strong>Title:</strong> ${session.title || 'Untitled Session'}</p>
                    <p><strong>Status:</strong> <span class="session-status status-${session.status}">${session.status}</span></p>
                    <p><strong>Created:</strong> ${new Date(session.created_at).toLocaleDateString()}</p>
                    <p><strong>Client:</strong> ${session.client_name}</p>
                </div>
                <div>
                    <h4>Notes</h4>
                    <p>${session.notes || 'No notes available'}</p>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <button class="btn" onclick="editSession('${session.id}')">Edit Session</button>
                <button class="btn btn-secondary" onclick="loadSessionAnalysis('${session.id}')">View Analysis</button>
            </div>
        `;
        
        document.getElementById('session-view').classList.remove('hidden');
    } catch (error) {
        console.error('Error loading session:', error);
        showMessage('Error loading session', 'error');
    }
}

// File upload functions
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (file) {
        uploadAudioFile(file);
    }
}

function handleTextFileSelection(e) {
    const file = e.target.files[0];
    const uploadArea = document.querySelector('#analysis-view .upload-area');
    
    if (file) {
        if (file.type !== 'text/plain') {
            showMessage('Please select a valid .txt file', 'error');
            e.target.value = '';
            return;
        }
        
        // Update the upload area to show selected file
        const fileInfo = uploadArea.querySelector('.file-info') || document.createElement('div');
        fileInfo.className = 'file-info';
        fileInfo.innerHTML = `
            <p style="color: #28a745; margin-top: 10px;">
                ✅ Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)
            </p>
        `;
        
        if (!uploadArea.querySelector('.file-info')) {
            uploadArea.appendChild(fileInfo);
        }
        
        showMessage(`File "${file.name}" selected successfully`, 'success');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadAudioFile(files[0]);
    }
}

function handleTextDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleTextDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type === 'text/plain') {
            const fileInput = document.getElementById('text-file-input');
            // Create a new FileList-like object
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            
            // Trigger the change event
            handleTextFileSelection({ target: fileInput });
        } else {
            showMessage('Please drop a valid .txt file.', 'error');
        }
    }
}

async function uploadAudioFile(file) {
    if (!currentSession) {
        showMessage('Please create a session first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('client_id', currentClient.id);
    
    try {
        showLoading(true);
        const response = await fetch('/api/preprocess/upload-audio', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            showMessage('Audio file uploaded successfully', 'success');
            // Update session with audio file info
            currentSession.audio_file = result.file_path;
        } else {
            const error = await response.json();
            showMessage(error.detail || 'Error uploading file', 'error');
        }
        
        showLoading(false);
    } catch (error) {
        console.error('Error uploading file:', error);
        showMessage('Error uploading file', 'error');
        showLoading(false);
    }
}

function startRecording() {
    // This would implement audio recording functionality
    showMessage('Recording functionality not yet implemented', 'error');
}

// Chat functions
function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    addChatMessage(message, 'user');
    
    // Clear input
    input.value = '';
    
    // Simulate AI response
    setTimeout(() => {
        const responses = [
            "I understand. Can you tell me more about that?",
            "That's an interesting perspective. How does that make you feel?",
            "Let's explore that further. What do you think is the underlying cause?",
            "I see. This seems to be a recurring theme. Have you noticed any patterns?",
            "Thank you for sharing that. What would you like to work on next?"
        ];
        
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        addChatMessage(randomResponse, 'ai');
    }, 1000);
}

function addChatMessage(message, sender) {
    const container = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'AI Therapist'}:</strong> ${message}`;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

// Calendar functions
function loadCalendar() {
    const grid = document.getElementById('calendar-grid');
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth();
    
    // Calendar header
    const daysOfWeek = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    let calendarHTML = daysOfWeek.map(day => `<div class="calendar-day" style="font-weight: bold; background: #f0f0f0;">${day}</div>`).join('');
    
    // First day of month
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const startDate = new Date(firstDay);
    startDate.setDate(startDate.getDate() - firstDay.getDay());
    
    // Generate calendar days
    for (let i = 0; i < 42; i++) {
        const date = new Date(startDate);
        date.setDate(startDate.getDate() + i);
        
        const isCurrentMonth = date.getMonth() === month;
        const isToday = date.toDateString() === now.toDateString();
        
        let dayClass = 'calendar-day';
        if (!isCurrentMonth) dayClass += ' other-month';
        if (isToday) dayClass += ' today';
        
        calendarHTML += `
            <div class="${dayClass}" onclick="selectDate('${date.toISOString().split('T')[0]}')">
                <div>${date.getDate()}</div>
                <!-- Placeholder for events -->
            </div>
        `;
    }
    
    grid.innerHTML = calendarHTML;
}

function selectDate(dateString) {
    // Handle date selection
    console.log('Selected date:', dateString);
    showMessage(`Selected date: ${dateString}`, 'success');
}

// Settings functions
function handleSettings(e) {
    e.preventDefault();
    
    const settings = {
        clinic_name: document.getElementById('clinic-name').value,
        therapist_name: document.getElementById('therapist-name').value,
        session_duration: document.getElementById('session-duration').value,
        timezone: document.getElementById('timezone').value
    };
    
    // Save to localStorage (in a real app, this would go to the server)
    localStorage.setItem('therapySettings', JSON.stringify(settings));
    
    showMessage('Settings saved successfully', 'success');
}

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'block' : 'none';
}

function showMessage(message, type) {
    const container = document.getElementById('message-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = type;
    messageDiv.textContent = message;
    
    // Position at top of screen
    messageDiv.style.position = 'fixed';
    messageDiv.style.top = '20px';
    messageDiv.style.right = '20px';
    messageDiv.style.zIndex = '3000';
    messageDiv.style.maxWidth = '300px';
    
    container.appendChild(messageDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// Session management functions
async function showClientSessions() {
    if (!currentClient) {
        showMessage('No client selected', 'error');
        return;
    }
    
    // Hide all views first
    document.querySelectorAll('.view').forEach(v => {
        v.classList.add('hidden');
    });
    
    try {
        const response = await fetch(`/api/clients/${currentClient.id}/sessions`);
        const sessions = await response.json();
        
        document.getElementById('page-title').textContent = `${currentClient.name} - Sessions`;
        document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${currentClient.id}')">${currentClient.name}</a> > Sessions`;
        
        document.getElementById('client-sessions-content').innerHTML = `
            <div class="card">
                <h4>Sessions for ${currentClient.name}</h4>
                <div class="session-list">
                    ${sessions.length > 0 ? sessions.map(session => `
                        <div class="session-item" onclick="showSessionView('${session.id}')">
                            <div class="session-title">${session.title || 'Untitled Session'}</div>
                            <div class="session-date">${new Date(session.created_at).toLocaleDateString()}</div>
                            <span class="session-status status-${session.status}">${session.status}</span>
                        </div>
                    `).join('') : '<p>No sessions found for this client.</p>'}
                </div>
                <div style="margin-top: 20px;">
                    <button class="btn" onclick="showNewSessionForm()">➕ New Session</button>
                </div>
            </div>
        `;
        
        document.getElementById('client-sessions-view').classList.remove('hidden');
    } catch (error) {
        console.error('Error loading client sessions:', error);
        showMessage('Error loading client sessions', 'error');
    }
}

function showNewSessionForm() {
    if (!currentClient) {
        showMessage('No client selected', 'error');
        return;
    }
    
    // Hide all views first
    document.querySelectorAll('.view').forEach(v => {
        v.classList.add('hidden');
    });
    
    document.getElementById('page-title').textContent = `New Session for ${currentClient.name}`;
    document.getElementById('breadcrumb-text').innerHTML = `<a href="#" onclick="showView('clients')">Clients</a> > <a href="#" onclick="showClientProfile('${currentClient.id}')">${currentClient.name}</a> > <a href="#" onclick="showClientSessions()">Sessions</a> > New Session`;
    
    document.getElementById('new-session-view').classList.remove('hidden');
}

async function editSession(sessionId) {
    // This would open an edit form
    showMessage('Edit session functionality not yet implemented', 'error');
}

async function deleteSession(sessionId) {
    if (!confirm('Are you sure you want to delete this session?')) return;
    
    try {
        const response = await fetch(`/api/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showMessage('Session deleted successfully', 'success');
            showClientSessions();
        } else {
            const error = await response.json();
            showMessage(error.detail || 'Error deleting session', 'error');
        }
    } catch (error) {
        console.error('Error deleting session:', error);
        showMessage('Error deleting session', 'error');
    }
}

async function loadSessionAnalysis(sessionId) {
    try {
        const response = await fetch(`/api/simple_analyze?session_id=${sessionId}`);
        
        if (response.ok) {
            const analysis = await response.json();
            // Display analysis results
            showMessage('Analysis loaded successfully', 'success');
            console.log('Analysis results:', analysis);
        } else {
            showMessage('No analysis available for this session', 'error');
        }
    } catch (error) {
        console.error('Error loading analysis:', error);
        showMessage('Error loading analysis', 'error');
    }
}

// Client Profiling Functions
async function showClientProfiling() {
    if (!currentClient) {
        showMessage('No client selected', 'error');
        return;
    }
    
    // Hide all views first
    document.querySelectorAll('.view').forEach(v => {
        v.classList.add('hidden');
    });
    
    document.getElementById('client-profiling-view').classList.remove('hidden');
    
    // Update breadcrumb
    updateBreadcrumb([
        { text: 'Clients', onclick: () => showView('clients') },
        { text: currentClient.name || 'Unnamed Client', onclick: () => showClientProfile(currentClient.id) },
        { text: 'Needs Profiling' }
    ]);
    
    // Update client profiling content
    const content = document.getElementById('client-profiling-content');
    content.innerHTML = `
        <div class="card">
            <h4>📋 Client Information</h4>
            <p><strong>Name:</strong> ${currentClient.name || 'Unnamed Client'}</p>
            <p><strong>Email:</strong> ${currentClient.email || 'No email'}</p>
            <p><strong>Client ID:</strong> <span class="client-id">${currentClient.id}</span></p>
            <p><strong>Sessions:</strong> ${currentClient.session_count || 0}</p>
        </div>
    `;
}

async function generateClientNeedsProfile() {
    if (!currentClient) {
        showMessage('No client selected', 'error');
        return;
    }
    
    const sessionCount = document.getElementById('session-count').value || 10;
    
    showLoading(true);
    
    try {
        const response = await fetch(`/api/profiling/clients/${currentClient.id}/analyze-needs?session_count=${sessionCount}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate profile');
        }
        
        const result = await response.json();
        showMessage('Profile generation started successfully', 'success');
        
        // Auto-refresh dashboard after a short delay
        setTimeout(() => {
            viewClientNeedsDashboard();
        }, 3000);
        
    } catch (error) {
        console.error('Error generating profile:', error);
        showMessage('Error generating profile: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

async function viewClientNeedsDashboard() {
    if (!currentClient) {
        showMessage('No client selected', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`/api/profiling/clients/${currentClient.id}/needs-dashboard`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load dashboard');
        }
        
        const dashboardData = await response.json();
        displayClientNeedsDashboard(dashboardData);
        
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showMessage('Error loading dashboard: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayClientNeedsDashboard(data) {
    const resultsDiv = document.getElementById('client-needs-dashboard-results');
    resultsDiv.style.display = 'block';
    
    // Create radar chart for life segments
    const radarCtx = document.getElementById('client-radar-chart').getContext('2d');
    
    // Clear existing chart
    if (window.clientRadarChart) {
        window.clientRadarChart.destroy();
    }
    
    const radarData = data.visualization_data.radar_chart;
    window.clientRadarChart = new Chart(radarCtx, {
        type: 'radar',
        data: radarData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Create bar chart for needs fulfillment
    const barCtx = document.getElementById('client-bar-chart').getContext('2d');
    
    // Clear existing chart
    if (window.clientBarChart) {
        window.clientBarChart.destroy();
    }
    
    const barData = data.visualization_data.bar_chart;
    window.clientBarChart = new Chart(barCtx, {
        type: 'bar',
        data: barData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Display insights
    const insightsList = document.getElementById('client-insights-list');
    insightsList.innerHTML = '';
    data.life_segments.insights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        insightsList.appendChild(li);
    });
    
    // Display recommendations
    const recommendationsDiv = document.getElementById('client-recommendations');
    recommendationsDiv.innerHTML = '';
    data.recommendations.forEach(rec => {
        const div = document.createElement('div');
        div.innerHTML = `
            <h6>${rec.intervention} <span class="badge ${rec.priority}">${rec.priority}</span></h6>
            <p>${rec.description}</p>
        `;
        div.style.marginBottom = '10px';
        recommendationsDiv.appendChild(div);
    });
}

// Initialize dashboard when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadDashboardData);
} else {
    loadDashboardData();
}
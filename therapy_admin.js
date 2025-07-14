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
}

// Navigation functions
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

// Initialize dashboard when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadDashboardData);
} else {
    loadDashboardData();
}
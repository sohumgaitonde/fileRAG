// fileRAG Desktop - Renderer Process
const { ipcRenderer } = require('electron');
const axios = require('axios');

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentTab = 'index';
let isIndexing = false;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    console.log('fileRAG Desktop initialized');
    checkBackendStatus();
    
    // Add event listeners
    const mainSearchInput = document.getElementById('mainSearchInput');
    if (mainSearchInput) {
        // Auto-resize textarea
        mainSearchInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
        
        // Handle Enter key (but allow Shift+Enter for new lines)
        mainSearchInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                performSearch();
            }
        });
    }
});

// Tab Management
function showTab(tabName, clickedElement = null) {
    // Hide all tab content
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
        tab.style.display = 'none';
    });
    
    // Remove active class from all bottom tabs
    document.querySelectorAll('.bottom-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    const targetTab = document.getElementById(tabName);
    if (targetTab) {
        targetTab.classList.add('active');
        targetTab.style.display = 'block';
    }
    
    // Add active class to clicked bottom tab (if provided)
    if (clickedElement) {
        clickedElement.classList.add('active');
    } else {
        // Find the corresponding bottom tab and activate it
        const bottomTab = document.querySelector(`[onclick="showTab('${tabName}')"]`);
        if (bottomTab) {
            bottomTab.classList.add('active');
        }
    }
    
    currentTab = tabName;
    
    // Load tab-specific data
    if (tabName === 'status') {
        checkStatus();
    }
}

// Directory Selection
function browseDirectory() {
    // For now, use a simple prompt. In a real app, you'd use dialog.showOpenDialog
    const path = prompt('Enter directory path:');
    if (path) {
        document.getElementById('directoryPath').value = path;
    }
}

// Indexing Functions
async function startIndexing() {
    const directoryPath = document.getElementById('directoryPath').value;
    
    if (!directoryPath) {
        showStatus('indexStatus', 'Please select a directory first', 'error');
        return;
    }
    
    try {
        isIndexing = true;
        showStatus('indexStatus', 'Starting indexing process...', 'info');
        
        // Call the backend API to start indexing
        const response = await axios.post(`${API_BASE_URL}/api/index`, {
            directory_path: directoryPath
        });
        
        showStatus('indexStatus', 'Indexing started successfully! Check the Status tab for progress.', 'success');
        
        // Switch to status tab to show progress
        showTab('status');
        
    } catch (error) {
        console.error('Indexing error:', error);
        showStatus('indexStatus', `Error: ${error.message}`, 'error');
    } finally {
        isIndexing = false;
    }
}

// Search Functions
async function performSearch() {
    const query = document.getElementById('mainSearchInput').value;
    
    if (!query.trim()) {
        showStatus('searchResults', 'Please enter a search query', 'error');
        return;
    }
    
    try {
        // Automatically switch to search tab to show results
        showTab('search');
        showStatus('searchResults', 'Searching...', 'info');
        
        const response = await axios.post(`${API_BASE_URL}/api/search`, {
            query: query,
            limit: 10
        });
        
        displaySearchResults(response.data);
        
    } catch (error) {
        console.error('Search error:', error);
        showStatus('searchResults', `Search error: ${error.message}`, 'error');
    }
}

function displaySearchResults(results) {
    const container = document.getElementById('searchResults');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="status info">No results found</div>';
        return;
    }
    
    let html = '<h3>Search Results:</h3>';
    results.forEach((result, index) => {
        html += `
            <div class="search-result">
                <h4>${result.filename || 'Unknown File'}</h4>
                <p><strong>Score:</strong> <span style="color: #00ffff;">${(result.score * 100).toFixed(1)}%</span></p>
                <p><strong>Content:</strong> ${result.content || result.text || 'No preview available'}</p>
                <p><strong>Path:</strong> <span style="color: #888; font-family: monospace;">${result.file_path || 'Unknown path'}</span></p>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Status Functions
async function checkStatus() {
    try {
        // Check backend health
        const healthResponse = await axios.get(`${API_BASE_URL}/health`);
        document.getElementById('backendStatus').textContent = 'Connected';
        document.getElementById('backendStatus').style.color = 'green';
        
        // Get system stats (if available)
        try {
            const statsResponse = await axios.get(`${API_BASE_URL}/api/stats`);
            document.getElementById('fileCount').textContent = statsResponse.data.file_count || 'Unknown';
            document.getElementById('dbSize').textContent = statsResponse.data.db_size || 'Unknown';
        } catch (statsError) {
            console.log('Stats endpoint not available yet');
            document.getElementById('fileCount').textContent = 'N/A';
            document.getElementById('dbSize').textContent = 'N/A';
        }
        
    } catch (error) {
        console.error('Status check error:', error);
        document.getElementById('backendStatus').textContent = 'Disconnected';
        document.getElementById('backendStatus').style.color = 'red';
        document.getElementById('fileCount').textContent = 'N/A';
        document.getElementById('dbSize').textContent = 'N/A';
    }
}

async function checkBackendStatus() {
    try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        console.log('Backend is running');
    } catch (error) {
        console.log('Backend not available yet');
        showStatus('indexStatus', 'Backend not running. Please start the Python backend first.', 'error');
    }
}

// Utility Functions
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="status ${type}">${message}</div>`;
}

// Auto-refresh status every 30 seconds
setInterval(() => {
    if (currentTab === 'status') {
        checkStatus();
    }
}, 30000);

// Handle Enter key in search
document.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        if (currentTab === 'index') {
            startIndexing();
        }
    }
});

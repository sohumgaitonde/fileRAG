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
            limit: 10,
            result_limit: 20
        });
        
        displaySearchResults(response.data);
        
    } catch (error) {
        console.error('Search error:', error);
        showStatus('searchResults', `Search error: ${error.message}`, 'error');
    }
}

function displaySearchResults(data) {
    const container = document.getElementById('searchResults');
    
    // Handle both old format (array) and new format (object with results array)
    const results = data.results || data;
    const queryVariations = data.query_variations || [];
    const performanceMetrics = data.performance_metrics || {};
    const qualityMetrics = data.quality_metrics || {};
    const totalTime = data.total_time || 0;
    
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="status info">No results found</div>';
        return;
    }
    
    let html = '<div class="search-header">';
    html += '<h3>🔍 Search Results</h3>';
    
    // Display performance metrics
    if (totalTime > 0) {
        html += `<div class="performance-info">`;
        html += `<span class="metric">⏱️ Search Time: ${totalTime.toFixed(2)}s</span>`;
        html += `<span class="metric">📊 Results: ${results.length}</span>`;
        if (queryVariations.length > 0) {
            html += `<span class="metric">🔄 Queries: ${queryVariations.length}</span>`;
        }
        html += `</div>`;
    }
    
    // Display quality metrics if available
    if (qualityMetrics.avg_score !== undefined) {
        html += `<div class="quality-info">`;
        html += `<span class="metric">📈 Avg Score: ${(qualityMetrics.avg_score * 100).toFixed(1)}%</span>`;
        if (qualityMetrics.diversity_score !== undefined) {
            html += `<span class="metric">🎯 Diversity: ${(qualityMetrics.diversity_score * 100).toFixed(1)}%</span>`;
        }
        if (qualityMetrics.coverage_score !== undefined) {
            html += `<span class="metric">📋 Coverage: ${(qualityMetrics.coverage_score * 100).toFixed(1)}%</span>`;
        }
        html += `</div>`;
    }
    
    html += '</div>';
    
    // Display query variations if available
    if (queryVariations.length > 0) {
        html += '<div class="query-variations">';
        html += '<h4>🔍 Generated Query Variations:</h4>';
        html += '<div class="variations-list">';
        queryVariations.forEach((variation, index) => {
            html += `<div class="variation-item">${index + 1}. ${variation}</div>`;
        });
        html += '</div>';
        html += '</div>';
    }
    
    // Display search results
    html += '<div class="results-container">';
    results.forEach((result, index) => {
        const baseScore = result.score || 0;
        const weightedScore = result.weighted_score || baseScore;
        const queryImportance = result.query_importance || 1.0;
        const foundByQueries = result.found_by_queries || [];
        const totalMatches = result.total_matches || 1;
        
        html += `
            <div class="search-result enhanced">
                <div class="result-header">
                    <h4>📄 ${result.filename || 'Unknown File'}</h4>
                    <div class="score-info">
                        <span class="score primary">${(baseScore * 100).toFixed(1)}%</span>
                        ${weightedScore !== baseScore ? `<span class="score weighted">Weighted: ${(weightedScore * 100).toFixed(1)}%</span>` : ''}
                    </div>
                </div>
                
                <div class="result-metrics">
                    ${queryImportance !== 1.0 ? `<span class="metric">🎯 Importance: ${queryImportance.toFixed(2)}</span>` : ''}
                    ${totalMatches > 1 ? `<span class="metric">🔍 Found ${totalMatches} times</span>` : ''}
                    ${foundByQueries.length > 0 ? `<span class="metric">📝 By ${foundByQueries.length} queries</span>` : ''}
                </div>
                
                <div class="result-content">
                    <p><strong>Content:</strong> ${result.content || result.text || 'No preview available'}</p>
                    <p><strong>Path:</strong> <span class="file-path">${result.file_path || 'Unknown path'}</span></p>
                </div>
                
                ${foundByQueries.length > 0 ? `
                    <div class="query-attribution">
                        <strong>Found by queries:</strong>
                        <div class="query-tags">
                            ${foundByQueries.map(query => `<span class="query-tag">${query}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    });
    html += '</div>';
    
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

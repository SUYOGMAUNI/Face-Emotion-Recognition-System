/**
 * script.js — Frontend logic for Face Emotion Recognition Web UI
 * Author: Suyog Mauni | suyogmauni.com.np
 * 
 * Handles:
 * - Real-time stats updates
 * - Emotion chart rendering
 * - Camera controls
 * - UI state management
 */

const EMOTIONS = {
    'Happy': { emoji: '😊', color: '#64DC00' },
    'Sad': { emoji: '😢', color: '#C86432' },
    'Angry': { emoji: '😠', color: '#DC3232' },
    'Surprise': { emoji: '😲', color: '#DCC832' },
    'Fear': { emoji: '😨', color: '#C832C8' },
    'Disgust': { emoji: '🤢', color: '#50B450' },
    'Neutral': { emoji: '😐', color: '#B4B4B4' }
};

const EMOTION_ORDER = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'];

let isRunning = true;
let updateInterval = null;

// DOM Elements
const elements = {
    fps: document.getElementById('fps'),
    frameCount: document.getElementById('frame-count'),
    sessionTime: document.getElementById('session-time'),
    faceCount: document.getElementById('face-count'),
    totalDetections: document.getElementById('total-detections'),
    emotionChart: document.getElementById('emotion-chart'),
    noFaceOverlay: document.getElementById('no-face-overlay'),
    statusBadge: document.getElementById('status-badge'),
    statusText: document.getElementById('status-text'),
    toggleCameraBtn: document.getElementById('toggle-camera'),
    resetStatsBtn: document.getElementById('reset-stats')
};

/**
 * Format seconds to MM:SS
 */
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Fetch and update statistics from server
 */
async function updateStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        // Update header stats
        elements.fps.textContent = data.fps.toFixed(1);
        elements.frameCount.textContent = data.frame_count.toLocaleString();
        elements.sessionTime.textContent = formatTime(data.session_time);
        
        // Update face count
        elements.faceCount.textContent = data.face_count;
        
        // Calculate total detections
        const totalDetections = Object.values(data.emotion_data)
            .reduce((sum, emotion) => sum + emotion.count, 0);
        elements.totalDetections.textContent = totalDetections.toLocaleString();
        
        // Update emotion chart
        updateEmotionChart(data.emotion_data);
        
        // Update no-face overlay
        if (data.face_count === 0 && data.is_running) {
            elements.noFaceOverlay.classList.add('show');
        } else {
            elements.noFaceOverlay.classList.remove('show');
        }
        
        // Update status
        isRunning = data.is_running;
        updateStatus();
        
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

/**
 * Update emotion distribution chart
 */
function updateEmotionChart(emotionData) {
    const chart = elements.emotionChart;
    chart.innerHTML = '';
    
    EMOTION_ORDER.forEach(emotion => {
        const data = emotionData[emotion];
        const { emoji, color } = EMOTIONS[emotion];
        
        const barElement = document.createElement('div');
        barElement.className = 'emotion-bar';
        
        barElement.innerHTML = `
            <div class="emotion-header">
                <div class="emotion-label">
                    <span class="emotion-emoji">${emoji}</span>
                    <span>${emotion}</span>
                </div>
                <span class="emotion-percentage">${data.percentage.toFixed(1)}%</span>
            </div>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${data.percentage}%; background: ${color};"></div>
            </div>
        `;
        
        chart.appendChild(barElement);
    });
}

/**
 * Update status badge
 */
function updateStatus() {
    if (isRunning) {
        elements.statusBadge.classList.remove('stopped');
        elements.statusText.textContent = 'Live';
        elements.toggleCameraBtn.innerHTML = `
            <span class="btn-icon">⏸️</span>
            Stop Camera
        `;
    } else {
        elements.statusBadge.classList.add('stopped');
        elements.statusText.textContent = 'Stopped';
        elements.toggleCameraBtn.innerHTML = `
            <span class="btn-icon">▶️</span>
            Start Camera
        `;
    }
}

/**
 * Toggle camera on/off
 */
async function toggleCamera() {
    try {
        const endpoint = isRunning ? '/stop_camera' : '/start_camera';
        const response = await fetch(endpoint, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'stopped') {
            isRunning = false;
            // Reload page to restart video stream
            setTimeout(() => location.reload(), 500);
        } else if (data.status === 'started') {
            isRunning = true;
            location.reload();
        }
        
        updateStatus();
    } catch (error) {
        console.error('Error toggling camera:', error);
        alert('Failed to toggle camera. Please refresh the page.');
    }
}

/**
 * Reset statistics
 */
async function resetStats() {
    if (!confirm('Are you sure you want to reset all statistics?')) {
        return;
    }
    
    try {
        const response = await fetch('/reset_stats', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update UI immediately
            elements.frameCount.textContent = '0';
            elements.totalDetections.textContent = '0';
            elements.sessionTime.textContent = '00:00';
            
            // Clear emotion chart
            EMOTION_ORDER.forEach(emotion => {
                const data = { count: 0, percentage: 0 };
                updateEmotionChart(
                    Object.fromEntries(
                        EMOTION_ORDER.map(e => [e, { count: 0, percentage: 0, color: EMOTIONS[e].color }])
                    )
                );
            });
        }
    } catch (error) {
        console.error('Error resetting stats:', error);
        alert('Failed to reset statistics. Please try again.');
    }
}

/**
 * Initialize the application
 */
function init() {
    console.log('🎭 Face Emotion Recognition - Web UI initialized');
    
    // Set up event listeners
    elements.toggleCameraBtn.addEventListener('click', toggleCamera);
    elements.resetStatsBtn.addEventListener('click', resetStats);
    
    // Initial status update
    updateStatus();
    
    // Start periodic updates (every 500ms for smooth updates)
    updateStats();
    updateInterval = setInterval(updateStats, 500);
    
    // Handle page visibility changes to pause/resume updates
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            if (updateInterval) {
                clearInterval(updateInterval);
                updateInterval = null;
            }
        } else {
            if (!updateInterval) {
                updateStats();
                updateInterval = setInterval(updateStats, 500);
            }
        }
    });
    
    // Handle page unload
    window.addEventListener('beforeunload', () => {
        if (updateInterval) {
            clearInterval(updateInterval);
        }
    });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Space to toggle camera
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        toggleCamera();
    }
    
    // R to reset stats (with Ctrl/Cmd)
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        resetStats();
    }
});

// Expose functions globally for debugging
window.emotionApp = {
    updateStats,
    toggleCamera,
    resetStats,
    isRunning: () => isRunning
};
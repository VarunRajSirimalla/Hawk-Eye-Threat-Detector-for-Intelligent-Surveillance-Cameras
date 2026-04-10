// API Configuration for HawkEye CNN Frontend
// This file defines the backend API endpoint for all frontend pages

// Default to localhost, but can be overridden for production deployment
window.API_BASE = window.API_BASE || 'http://localhost:8001';

// Log configuration on page load
console.log('HawkEye CNN - API Base:', window.API_BASE);

// Optional: Detect if backend is available
(async function checkBackendHealth() {
    try {
        const response = await fetch(`${window.API_BASE}/health`, { 
            method: 'GET',
            timeout: 5000 
        });
        if (response.ok) {
            console.log('✓ Backend is healthy');
        } else {
            console.warn('⚠ Backend responded with status:', response.status);
        }
    } catch (error) {
        console.error('✗ Backend is not accessible:', error.message);
        console.log('Make sure the backend server is running on', window.API_BASE);
    }
})();

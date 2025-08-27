import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

// Remove loading screen when React takes over
const removeLoadingScreen = () => {
  const loadingContainer = document.querySelector('.loading-container');
  if (loadingContainer) {
    loadingContainer.style.opacity = '0';
    setTimeout(() => {
      loadingContainer.remove();
    }, 300);
  }
};

// Initialize app
const container = document.getElementById('root');
if (!container) {
  throw new Error('Failed to find root element');
}

const root = createRoot(container);

// Render app with error handling
try {
  root.render(<App />);
  
  // Remove loading screen after React renders
  setTimeout(removeLoadingScreen, 100);
  
  // Add loaded class for animations
  document.body.classList.add('app-loaded');
  
} catch (error) {
  console.error('Failed to render app:', error);
  
  // Show fallback error UI
  container.innerHTML = `
    <div class="error-boundary">
      <div class="error-content">
        <h2 class="error-title">Application Error</h2>
        <p class="error-message">
          SuperNova AI failed to load. Please refresh the page or contact support if the problem persists.
        </p>
        <button class="error-button" onclick="window.location.reload()">
          Reload Application
        </button>
      </div>
    </div>
  `;
}

// Performance monitoring
if ('performance' in window && 'measure' in window.performance) {
  window.addEventListener('load', () => {
    // Measure React hydration time
    performance.mark('react-hydration-start');
    
    setTimeout(() => {
      performance.mark('react-hydration-end');
      performance.measure('react-hydration', 'react-hydration-start', 'react-hydration-end');
      
      const measure = performance.getEntriesByName('react-hydration')[0];
      if (measure) {
        console.log(`React hydration took ${measure.duration.toFixed(2)}ms`);
      }
    }, 0);
  });
}

// Hot module replacement for development
if (module.hot && process.env.NODE_ENV === 'development') {
  module.hot.accept('./App', () => {
    const NextApp = require('./App').default;
    root.render(<NextApp />);
  });
}
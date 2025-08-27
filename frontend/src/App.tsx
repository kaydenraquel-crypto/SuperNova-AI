import React, { Suspense, lazy, useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { HelmetProvider } from 'react-helmet-async';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';

// Theme
import { createSuperNovaTheme } from '@/theme';
import { ThemeContextProvider, useTheme } from '@/hooks/useTheme';

// Components
import LoadingScreen from '@/components/common/LoadingScreen';
import ErrorBoundary from '@/components/common/ErrorBoundary';
import ProgressBar from '@/components/common/ProgressBar';
import NotificationProvider from '@/components/common/NotificationProvider';

// Lazy-loaded pages
const DashboardPage = lazy(() => import('@/pages/DashboardPage'));
const PortfolioPage = lazy(() => import('@/pages/PortfolioPage'));
const ChatPage = lazy(() => import('@/pages/ChatPage'));
const MarketPage = lazy(() => import('@/pages/MarketPage'));
const BacktestPage = lazy(() => import('@/pages/BacktestPage'));
const SettingsPage = lazy(() => import('@/pages/SettingsPage'));
const LoginPage = lazy(() => import('@/pages/LoginPage'));
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'));

// Layout
import MainLayout from '@/components/layout/MainLayout';
import AuthLayout from '@/components/layout/AuthLayout';

// Hooks and services
import { useAuth } from '@/hooks/useAuth';
import { useWebSocket } from '@/hooks/useWebSocket';
import { ApiProvider } from '@/services/api';

// Types
import type { PaletteMode } from '@mui/material';

// React Query client configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Protected Route Component
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredAuth?: boolean;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  children, 
  requiredAuth = true 
}) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingScreen message="Verifying authentication..." />;
  }

  if (requiredAuth && !isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (!requiredAuth && isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// Main App Content Component
const AppContent: React.FC = () => {
  const { mode } = useTheme();
  const theme = createSuperNovaTheme(mode);
  const { isAuthenticated } = useAuth();

  // Initialize WebSocket connection for authenticated users
  const { connectionStatus, marketData, notifications } = useWebSocket({
    enabled: isAuthenticated,
    reconnectAttempts: 5,
    reconnectInterval: 3000,
  });

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ProgressBar />
      <NotificationProvider>
        <Routes>
          {/* Public Routes */}
          <Route
            path="/login"
            element={
              <ProtectedRoute requiredAuth={false}>
                <AuthLayout>
                  <Suspense fallback={<LoadingScreen />}>
                    <LoginPage />
                  </Suspense>
                </AuthLayout>
              </ProtectedRoute>
            }
          />

          {/* Protected Routes */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <MainLayout 
                  connectionStatus={connectionStatus}
                  notifications={notifications}
                >
                  <Suspense fallback={<LoadingScreen message="Loading page..." />}>
                    <Routes>
                      <Route path="/dashboard" element={<DashboardPage />} />
                      <Route path="/portfolio" element={<PortfolioPage />} />
                      <Route path="/chat" element={<ChatPage />} />
                      <Route path="/market" element={<MarketPage />} />
                      <Route path="/backtest" element={<BacktestPage />} />
                      <Route path="/settings" element={<SettingsPage />} />
                      <Route path="/404" element={<NotFoundPage />} />
                      <Route path="/" element={<Navigate to="/dashboard" replace />} />
                      <Route path="*" element={<Navigate to="/404" replace />} />
                    </Routes>
                  </Suspense>
                </MainLayout>
              </ProtectedRoute>
            }
          />
        </Routes>
      </NotificationProvider>
    </ThemeProvider>
  );
};

// Theme preference detection hook
const useThemePreference = (): PaletteMode => {
  const [mode, setMode] = useState<PaletteMode>(() => {
    const savedMode = localStorage.getItem('superNovaTheme') as PaletteMode | null;
    if (savedMode && ['light', 'dark'].includes(savedMode)) {
      return savedMode;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem('superNovaTheme')) {
        setMode(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return mode;
};

// Main App Component
const App: React.FC = () => {
  const defaultThemeMode = useThemePreference();

  // Performance monitoring
  useEffect(() => {
    // Log initial performance metrics
    if ('performance' in window) {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigation) {
        console.log('App Performance Metrics:', {
          domContentLoaded: navigation.domContentLoadedEventEnd - navigation.fetchStart,
          firstContentfulPaint: navigation.loadEventEnd - navigation.fetchStart,
          totalLoadTime: navigation.loadEventEnd - navigation.fetchStart,
        });
      }
    }

    // Error tracking
    const handleUnhandledError = (event: ErrorEvent) => {
      console.error('Unhandled error:', event.error);
      // Here you could send to your error tracking service
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason);
      // Here you could send to your error tracking service
    };

    window.addEventListener('error', handleUnhandledError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleUnhandledError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  return (
    <ErrorBoundary>
      <HelmetProvider>
        <QueryClientProvider client={queryClient}>
          <ApiProvider>
            <ThemeContextProvider defaultMode={defaultThemeMode}>
              <Router>
                <AppContent />
              </Router>
            </ThemeContextProvider>
          </ApiProvider>
          {process.env.NODE_ENV === 'development' && (
            <ReactQueryDevtools initialIsOpen={false} />
          )}
        </QueryClientProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
};

export default App;
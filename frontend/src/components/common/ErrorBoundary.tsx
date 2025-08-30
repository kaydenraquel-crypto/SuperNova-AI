import React, { Component, ErrorInfo, ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Container,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  ExpandMore,
  BugReport,
  Home,
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  eventId: string | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      eventId: null,
    };
  }

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Generate a unique event ID for this error
    const eventId = Date.now().toString(36) + Math.random().toString(36).substr(2);
    
    this.setState({
      errorInfo,
      eventId,
    });

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.group('🚨 Error Boundary Caught an Error');
      console.error('Error:', error);
      console.error('Error Info:', errorInfo);
      console.error('Component Stack:', errorInfo.componentStack);
      console.groupEnd();
    }

    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);

    // In production, you would send this to your error tracking service
    if (process.env.NODE_ENV === 'production') {
      this.reportError(error, errorInfo, eventId);
    }
  }

  private reportError = (error: Error, errorInfo: ErrorInfo, eventId: string) => {
    // Send error to monitoring service (e.g., Sentry, LogRocket, etc.)
    console.log('Reporting error to monitoring service:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      eventId,
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
      url: typeof window !== 'undefined' ? window.location.href : 'unknown',
      timestamp: new Date().toISOString(),
    });
  };

  private handleReload = () => {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  private handleGoHome = () => {
    if (typeof window !== 'undefined') {
      window.location.href = '/dashboard';
    }
  };

  private handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      eventId: null,
    });
  };

  public render() {
    if (this.state.hasError) {
      // Custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, eventId } = this.state;
      const isDevelopment = process.env.NODE_ENV === 'development';

      return (
        <Container maxWidth="md" sx={{ py: 4 }}>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              textAlign: 'center',
              gap: 3,
            }}
          >
            {/* Error Icon */}
            <Box
              sx={{
                width: 100,
                height: 100,
                borderRadius: '50%',
                bgcolor: 'error.light',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              <ErrorIcon sx={{ fontSize: 48, color: 'error.contrastText' }} />
            </Box>

            {/* Main Error Message */}
            <Box>
              <Typography variant="h4" component="h1" gutterBottom>
                Oops! Something went wrong
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3, maxWidth: 600 }}>
                We're sorry, but an unexpected error occurred in the SuperNova AI application. 
                Our team has been notified and is working to fix the issue.
              </Typography>
              
              {eventId && (
                <Chip
                  label={`Error ID: ${eventId}`}
                  size="small"
                  variant="outlined"
                  sx={{ fontFamily: 'monospace' }}
                />
              )}
            </Box>

            {/* Action Buttons */}
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={this.handleReload}
                size="large"
              >
                Reload Page
              </Button>
              <Button
                variant="outlined"
                startIcon={<Home />}
                onClick={this.handleGoHome}
                size="large"
              >
                Go to Dashboard
              </Button>
              {isDevelopment && (
                <Button
                  variant="text"
                  onClick={this.handleReset}
                  size="large"
                >
                  Reset Error Boundary
                </Button>
              )}
            </Box>

            {/* Error Details (Development Only) */}
            {isDevelopment && error && (
              <Card sx={{ width: '100%', mt: 4 }}>
                <CardContent>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      The following error details are only shown in development mode:
                    </Typography>
                  </Alert>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <BugReport />
                        <Typography variant="h6">Error Details</Typography>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box sx={{ textAlign: 'left' }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Error Message:
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{
                            fontFamily: 'monospace',
                            bgcolor: 'background.paper',
                            p: 2,
                            borderRadius: 1,
                            mb: 2,
                            border: '1px solid',
                            borderColor: 'divider',
                          }}
                        >
                          {error.message}
                        </Typography>

                        {error.stack && (
                          <>
                            <Typography variant="subtitle2" gutterBottom>
                              Stack Trace:
                            </Typography>
                            <Typography
                              variant="body2"
                              component="pre"
                              sx={{
                                fontFamily: 'monospace',
                                fontSize: '0.75rem',
                                bgcolor: 'background.paper',
                                p: 2,
                                borderRadius: 1,
                                mb: 2,
                                border: '1px solid',
                                borderColor: 'divider',
                                overflow: 'auto',
                                maxHeight: 200,
                              }}
                            >
                              {error.stack}
                            </Typography>
                          </>
                        )}

                        {errorInfo?.componentStack && (
                          <>
                            <Typography variant="subtitle2" gutterBottom>
                              Component Stack:
                            </Typography>
                            <Typography
                              variant="body2"
                              component="pre"
                              sx={{
                                fontFamily: 'monospace',
                                fontSize: '0.75rem',
                                bgcolor: 'background.paper',
                                p: 2,
                                borderRadius: 1,
                                border: '1px solid',
                                borderColor: 'divider',
                                overflow: 'auto',
                                maxHeight: 200,
                              }}
                            >
                              {errorInfo.componentStack}
                            </Typography>
                          </>
                        )}
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            )}

            {/* Help Information */}
            <Box sx={{ mt: 4, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                If this error persists, please contact our support team or check our status page.
              </Typography>
            </Box>
          </Box>
        </Container>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import {
  Snackbar,
  Alert,
  AlertColor,
  Slide,
  SlideProps,
} from '@mui/material';

interface Notification {
  id: string;
  message: string;
  severity: AlertColor;
  duration?: number;
  action?: React.ReactNode;
}

interface NotificationContextType {
  showNotification: (message: string, severity?: AlertColor, duration?: number) => void;
  showSuccess: (message: string, duration?: number) => void;
  showError: (message: string, duration?: number) => void;
  showWarning: (message: string, duration?: number) => void;
  showInfo: (message: string, duration?: number) => void;
  dismissNotification: (id: string) => void;
  dismissAll: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

interface NotificationProviderProps {
  children: ReactNode;
  maxNotifications?: number;
  defaultDuration?: number;
}

const SlideTransition = (props: SlideProps) => {
  return <Slide {...props} direction="up" />;
};

export const NotificationProvider: React.FC<NotificationProviderProps> = ({
  children,
  maxNotifications = 3,
  defaultDuration = 6000,
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const generateId = () => Math.random().toString(36).substr(2, 9);

  const showNotification = useCallback((
    message: string,
    severity: AlertColor = 'info',
    duration: number = defaultDuration
  ) => {
    const id = generateId();
    const notification: Notification = {
      id,
      message,
      severity,
      duration,
    };

    setNotifications(prev => {
      const newNotifications = [notification, ...prev];
      // Keep only the most recent notifications
      return newNotifications.slice(0, maxNotifications);
    });

    // Auto-dismiss after duration
    if (duration > 0) {
      setTimeout(() => {
        dismissNotification(id);
      }, duration);
    }
  }, [defaultDuration, maxNotifications]);

  const showSuccess = useCallback((message: string, duration?: number) => {
    showNotification(message, 'success', duration);
  }, [showNotification]);

  const showError = useCallback((message: string, duration?: number) => {
    showNotification(message, 'error', duration);
  }, [showNotification]);

  const showWarning = useCallback((message: string, duration?: number) => {
    showNotification(message, 'warning', duration);
  }, [showNotification]);

  const showInfo = useCallback((message: string, duration?: number) => {
    showNotification(message, 'info', duration);
  }, [showNotification]);

  const dismissNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  const dismissAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const contextValue: NotificationContextType = {
    showNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    dismissNotification,
    dismissAll,
  };

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
      
      {/* Render notifications */}
      {notifications.map((notification, index) => (
        <Snackbar
          key={notification.id}
          open={true}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          TransitionComponent={SlideTransition}
          sx={{
            // Stack notifications vertically
            bottom: `${16 + (index * 70)}px !important`,
          }}
        >
          <Alert
            onClose={() => dismissNotification(notification.id)}
            severity={notification.severity}
            variant="filled"
            sx={{
              minWidth: 300,
              maxWidth: 500,
              boxShadow: 3,
            }}
            action={notification.action}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </NotificationContext.Provider>
  );
};
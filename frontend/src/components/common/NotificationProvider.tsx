import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import {
  Snackbar,
  Alert,
  AlertProps,
  Slide,
  SlideProps,
} from '@mui/material';

interface Notification {
  id: string;
  type: AlertProps['severity'];
  title?: string;
  message: string;
  duration?: number;
  action?: ReactNode;
}

interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

function SlideTransition(props: SlideProps) {
  return <Slide {...props} direction="up" />;
}

interface NotificationProviderProps {
  children: ReactNode;
  maxNotifications?: number;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({
  children,
  maxNotifications = 5,
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((notification: Omit<Notification, 'id'>) => {
    const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
    const newNotification: Notification = {
      id,
      duration: 6000, // Default 6 seconds
      ...notification,
    };

    setNotifications(prev => {
      // Remove oldest if we exceed max
      const updated = prev.length >= maxNotifications 
        ? prev.slice(1)
        : prev;
      return [...updated, newNotification];
    });

    // Auto-remove after duration
    if (newNotification.duration && newNotification.duration > 0) {
      setTimeout(() => {
        removeNotification(id);
      }, newNotification.duration);
    }

    return id;
  }, [maxNotifications]);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  const contextValue: NotificationContextType = {
    notifications,
    addNotification,
    removeNotification,
    clearNotifications,
  };

  // Get the most recent notification to display
  const currentNotification = notifications[notifications.length - 1];

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
      
      <Snackbar
        open={!!currentNotification}
        autoHideDuration={currentNotification?.duration || 6000}
        onClose={() => currentNotification && removeNotification(currentNotification.id)}
        TransitionComponent={SlideTransition}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        {currentNotification && (
          <Alert
            onClose={() => removeNotification(currentNotification.id)}
            severity={currentNotification.type}
            variant="filled"
            sx={{ width: '100%' }}
            action={currentNotification.action}
          >
            {currentNotification.title && (
              <strong>{currentNotification.title}: </strong>
            )}
            {currentNotification.message}
          </Alert>
        )}
      </Snackbar>
    </NotificationContext.Provider>
  );
};

export const useNotification = (): NotificationContextType => {
  const context = useContext(NotificationContext);
  if (context === undefined) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

// Convenience hooks for different notification types
export const useNotifications = () => {
  const { addNotification, removeNotification, clearNotifications } = useNotification();

  return {
    success: (message: string, title?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'success', message, title, ...options }),
    
    error: (message: string, title?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'error', message, title, ...options }),
    
    warning: (message: string, title?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'warning', message, title, ...options }),
    
    info: (message: string, title?: string, options?: Partial<Notification>) =>
      addNotification({ type: 'info', message, title, ...options }),
    
    remove: removeNotification,
    clear: clearNotifications,
  };
};

export default NotificationProvider;
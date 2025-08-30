import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Chip,
  Badge,
  Menu,
  MenuItem,
  Avatar,
  Button,
} from '@mui/material';
import {
  NotificationsActive,
  Warning,
  Info,
  CheckCircle,
  Error,
  MoreVert,
  Close,
  Settings,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface Alert {
  id: string;
  type: 'price' | 'news' | 'earnings' | 'technical';
  severity: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  symbol?: string;
  read: boolean;
}

interface AlertsCardProps {
  loading?: boolean;
}

const AlertsCard: React.FC<AlertsCardProps> = ({
  loading = false,
}) => {
  const theme = useTheme();
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: '1',
      type: 'price',
      severity: 'warning',
      title: 'Price Alert: AAPL',
      message: 'Apple stock has dropped below $175.00',
      timestamp: '2024-01-15T10:30:00Z',
      symbol: 'AAPL',
      read: false,
    },
    {
      id: '2',
      type: 'earnings',
      severity: 'info',
      title: 'Earnings Report: TSLA',
      message: 'Tesla Q4 2023 earnings report available',
      timestamp: '2024-01-15T09:15:00Z',
      symbol: 'TSLA',
      read: false,
    },
    {
      id: '3',
      type: 'technical',
      severity: 'success',
      title: 'Technical Signal: NVDA',
      message: 'Golden cross pattern detected',
      timestamp: '2024-01-14T16:45:00Z',
      symbol: 'NVDA',
      read: true,
    },
    {
      id: '4',
      type: 'news',
      severity: 'error',
      title: 'Market Alert',
      message: 'Significant market volatility detected',
      timestamp: '2024-01-14T14:20:00Z',
      read: true,
    },
  ]);

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'info':
        return <Info />;
      case 'warning':
        return <Warning />;
      case 'error':
        return <Error />;
      case 'success':
        return <CheckCircle />;
      default:
        return <NotificationsActive />;
    }
  };

  const getAlertColor = (severity: string) => {
    switch (severity) {
      case 'info':
        return theme.palette.info.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      case 'success':
        return theme.palette.success.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const getSeverityLabel = (severity: string) => {
    return severity.charAt(0).toUpperCase() + severity.slice(1);
  };

  const dismissAlert = (id: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id));
  };

  const markAsRead = (id: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === id ? { ...alert, read: true } : alert
      )
    );
  };

  const unreadCount = alerts.filter(alert => !alert.read).length;

  const formatTime = (timestamp: string) => {
    const now = new Date();
    const alertTime = new Date(timestamp);
    const diffInHours = Math.floor((now.getTime() - alertTime.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) {
      return 'Just now';
    } else if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    } else {
      return alertTime.toLocaleDateString();
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6" component="h3">
              Alerts
            </Typography>
            {unreadCount > 0 && (
              <Badge badgeContent={unreadCount} color="error">
                <NotificationsActive />
              </Badge>
            )}
          </Box>
          <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
            <MoreVert />
          </IconButton>
        </Box>

        <List disablePadding>
          {alerts.map((alert) => (
            <ListItem
              key={alert.id}
              sx={{
                px: 0,
                py: 1.5,
                borderBottom: 1,
                borderColor: 'divider',
                bgcolor: alert.read ? 'transparent' : theme.palette.action.hover,
                borderRadius: 1,
                mb: 1,
                cursor: 'pointer',
                '&:last-child': {
                  borderBottom: 0,
                  mb: 0,
                },
              }}
              onClick={() => !alert.read && markAsRead(alert.id)}
            >
              <Avatar
                sx={{ 
                  bgcolor: getAlertColor(alert.severity),
                  width: 32,
                  height: 32,
                  mr: 2,
                }}
              >
                {getAlertIcon(alert.severity)}
              </Avatar>

              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography 
                      variant="subtitle2" 
                      fontWeight={alert.read ? 'normal' : 'bold'}
                    >
                      {alert.title}
                    </Typography>
                    <Chip
                      label={getSeverityLabel(alert.severity)}
                      size="small"
                      color={alert.severity as any}
                      variant="outlined"
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ opacity: alert.read ? 0.7 : 1 }}
                    >
                      {alert.message}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTime(alert.timestamp)}
                    </Typography>
                  </Box>
                }
              />

              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  dismissAlert(alert.id);
                }}
                sx={{ ml: 1 }}
              >
                <Close sx={{ fontSize: 18 }} />
              </IconButton>
            </ListItem>
          ))}
        </List>

        {alerts.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              No active alerts
            </Typography>
            <Button
              startIcon={<Settings />}
              size="small"
              variant="outlined"
              sx={{ mt: 1 }}
            >
              Configure Alerts
            </Button>
          </Box>
        )}

        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Mark All as Read
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Clear All Alerts
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Alert Settings
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default AlertsCard;
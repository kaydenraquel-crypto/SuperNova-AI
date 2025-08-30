import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Button,
  IconButton,
  Tooltip,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  Settings,
  Refresh,
  Download,
  Upload,
  Search,
  Notifications,
  AccountBalance,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface QuickActionsCardProps {
  loading?: boolean;
}

const QuickActionsCard: React.FC<QuickActionsCardProps> = ({
  loading = false,
}) => {
  const theme = useTheme();

  const primaryActions = [
    {
      label: 'Buy Order',
      icon: <TrendingUp />,
      color: theme.palette.success.main,
      action: () => console.log('Buy order clicked'),
    },
    {
      label: 'Sell Order',
      icon: <TrendingDown />,
      color: theme.palette.error.main,
      action: () => console.log('Sell order clicked'),
    },
    {
      label: 'Portfolio Analysis',
      icon: <Assessment />,
      color: theme.palette.info.main,
      action: () => console.log('Portfolio analysis clicked'),
    },
    {
      label: 'Market Research',
      icon: <Search />,
      color: theme.palette.warning.main,
      action: () => console.log('Market research clicked'),
    },
  ];

  const secondaryActions = [
    {
      label: 'Refresh Data',
      icon: <Refresh />,
      tooltip: 'Refresh all market data',
      action: () => console.log('Refresh clicked'),
    },
    {
      label: 'Export Data',
      icon: <Download />,
      tooltip: 'Export portfolio data',
      action: () => console.log('Export clicked'),
    },
    {
      label: 'Import Data',
      icon: <Upload />,
      tooltip: 'Import transactions',
      action: () => console.log('Import clicked'),
    },
    {
      label: 'Account Settings',
      icon: <AccountBalance />,
      tooltip: 'Manage account settings',
      action: () => console.log('Account settings clicked'),
    },
    {
      label: 'Notifications',
      icon: <Notifications />,
      tooltip: 'Notification preferences',
      action: () => console.log('Notifications clicked'),
    },
    {
      label: 'Settings',
      icon: <Settings />,
      tooltip: 'Application settings',
      action: () => console.log('Settings clicked'),
    },
  ];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" component="h3" gutterBottom>
          Quick Actions
        </Typography>

        {/* Primary Actions */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Trading
          </Typography>
          <Grid container spacing={1}>
            {primaryActions.map((action, index) => (
              <Grid item xs={6} key={index}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={action.icon}
                  onClick={action.action}
                  sx={{
                    py: 1.5,
                    borderColor: action.color,
                    color: action.color,
                    '&:hover': {
                      backgroundColor: `${action.color}10`,
                      borderColor: action.color,
                    },
                    flexDirection: 'column',
                    gap: 0.5,
                  }}
                >
                  <Typography variant="caption" sx={{ textTransform: 'none' }}>
                    {action.label}
                  </Typography>
                </Button>
              </Grid>
            ))}
          </Grid>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Secondary Actions */}
        <Box>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Tools & Settings
          </Typography>
          <Grid container spacing={1}>
            {secondaryActions.map((action, index) => (
              <Grid item xs={4} key={index}>
                <Tooltip title={action.tooltip} arrow>
                  <IconButton
                    onClick={action.action}
                    sx={{
                      width: '100%',
                      height: 56,
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      flexDirection: 'column',
                      gap: 0.5,
                      '&:hover': {
                        backgroundColor: theme.palette.action.hover,
                        borderColor: theme.palette.primary.main,
                      },
                    }}
                  >
                    {action.icon}
                    <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
                      {action.label}
                    </Typography>
                  </IconButton>
                </Tooltip>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Quick Stats */}
        <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Quick Stats
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="success.main">
                  +2.4%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Today's P&L
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="primary.main">
                  $12,450
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Buying Power
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default QuickActionsCard;
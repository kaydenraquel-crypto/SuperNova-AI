import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Avatar,
  Chip,
  Button,
  IconButton,
  Menu,
  MenuItem,
  LinearProgress,
  Skeleton,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  MoreVert,
  Refresh,
  Add,
  AccountBalance,
  ShowChart,
  Assessment,
  Notifications,
  Speed,
  AttachMoney,
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { useQuery } from 'react-query';

// Components
import PortfolioOverviewCard from '../components/dashboard/PortfolioOverviewCard';
import PerformanceChartCard from '../components/dashboard/PerformanceChartCard';

// Hooks
import { useAuth } from '../hooks/useAuth';
import { useMarketData } from '../hooks/useWebSocket';
import { useThemeColors } from '../hooks/useTheme';

// Services
import { apiService } from '../services/api';

// Types
interface DashboardData {
  portfolio: {
    totalValue: number;
    dayChange: number;
    dayChangePercent: number;
    totalReturn: number;
    totalReturnPercent: number;
    cash: number;
    positions: number;
  };
  marketIndices: {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
  }[];
  watchlist: {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
  }[];
  recentTransactions: {
    id: string;
    symbol: string;
    type: 'buy' | 'sell';
    quantity: number;
    price: number;
    timestamp: string;
  }[];
  alerts: {
    id: string;
    type: 'price' | 'volume' | 'news' | 'ai_signal';
    title: string;
    message: string;
    symbol?: string;
    timestamp: string;
    read: boolean;
  }[];
  performance: {
    dates: string[];
    values: number[];
    benchmark?: number[];
  };
  news: {
    id: string;
    headline: string;
    summary: string;
    source: string;
    timestamp: string;
    sentiment?: 'positive' | 'negative' | 'neutral';
    symbols?: string[];
  }[];
}

const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const { user } = useAuth();
  const { getFinancialColor } = useThemeColors();
  
  const [refreshing, setRefreshing] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  // Fetch dashboard data
  const {
    data: dashboardData,
    isLoading,
    error,
    refetch,
  } = useQuery<DashboardData>(
    'dashboard',
    async () => {
      const response = await apiService.getDashboardData();
      return response;
    },
    {
      refetchInterval: 30000, // Refresh every 30 seconds
      staleTime: 15000, // Data is fresh for 15 seconds
    }
  );

  // Get real-time market data for major indices
  const majorIndices = ['SPY', 'QQQ', 'IWM', 'VTI'];
  const { data: marketData, connectionStatus } = useMarketData(majorIndices);

  const handleRefresh = async () => {
    setRefreshing(true);
    await refetch();
    setTimeout(() => setRefreshing(false), 1000);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const renderGreeting = () => {
    const hour = new Date().getHours();
    let greeting = 'Good morning';
    if (hour >= 12) greeting = 'Good afternoon';
    if (hour >= 17) greeting = 'Good evening';

    return `${greeting}, ${user?.name || 'Trader'}`;
  };

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">
          Error loading dashboard data. Please try again.
        </Typography>
        <Button onClick={handleRefresh} sx={{ mt: 2 }}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <>
      <Helmet>
        <title>Dashboard - SuperNova AI</title>
        <meta name="description" content="Your personalized financial dashboard with portfolio overview, market data, and AI insights." />
      </Helmet>

      <Box>
        {/* Header */}
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              {renderGreeting()}
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Here's your financial overview for today
            </Typography>
            {connectionStatus !== 'connected' && (
              <Chip 
                label={`Market data: ${connectionStatus}`}
                size="small"
                color={connectionStatus === 'error' ? 'error' : 'warning'}
                sx={{ mt: 1 }}
              />
            )}
          </Box>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <IconButton 
              onClick={handleRefresh} 
              disabled={refreshing || isLoading}
              title="Refresh dashboard"
            >
              <Refresh sx={{ 
                animation: refreshing ? 'spin 1s linear infinite' : 'none',
                '@keyframes spin': {
                  from: { transform: 'rotate(0deg)' },
                  to: { transform: 'rotate(360deg)' },
                },
              }} />
            </IconButton>
            
            <IconButton onClick={handleMenuOpen} title="Dashboard options">
              <MoreVert />
            </IconButton>
          </Box>
        </Box>

        {/* Loading indicator */}
        {(isLoading || refreshing) && (
          <LinearProgress sx={{ mb: 2 }} />
        )}

        {/* Dashboard Grid */}
        <Grid container spacing={3}>
          {/* Portfolio Overview */}
          <Grid item xs={12} lg={8}>
            <PortfolioOverviewCard
              data={dashboardData?.portfolio}
              isLoading={isLoading}
            />
          </Grid>

          {/* Quick Actions */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6">Quick Actions</Typography>
                <Typography variant="body2" color="text.secondary">
                  Quick actions will be available here.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Performance Chart */}
          <Grid item xs={12} lg={8}>
            <PerformanceChartCard
              data={dashboardData?.performance}
              isLoading={isLoading}
            />
          </Grid>

          {/* Market Overview */}
          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6">Market Overview</Typography>
                <Typography variant="body2" color="text.secondary">
                  Market data will be displayed here.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Watchlist */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6">Watchlist</Typography>
                <Typography variant="body2" color="text.secondary">
                  Your watchlist will be displayed here.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Transactions */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6">Recent Transactions</Typography>
                <Typography variant="body2" color="text.secondary">
                  Recent transactions will be displayed here.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Alerts */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6">Alerts</Typography>
                <Typography variant="body2" color="text.secondary">
                  Your alerts will be displayed here.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* News */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6">Financial News</Typography>
                <Typography variant="body2" color="text.secondary">
                  Latest financial news will be displayed here.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Dashboard Options Menu */}
        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={handleMenuClose}
        >
          <MenuItem onClick={() => { handleMenuClose(); handleRefresh(); }}>
            <Refresh sx={{ mr: 2 }} />
            Refresh All Data
          </MenuItem>
          <MenuItem onClick={handleMenuClose}>
            <Add sx={{ mr: 2 }} />
            Add Widget
          </MenuItem>
          <MenuItem onClick={handleMenuClose}>
            <Assessment sx={{ mr: 2 }} />
            Export Report
          </MenuItem>
        </Menu>
      </Box>
    </>
  );
};

export default DashboardPage;
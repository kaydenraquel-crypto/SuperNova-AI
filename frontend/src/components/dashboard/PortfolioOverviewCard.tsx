import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Skeleton,
  Avatar,
  Divider,
  Chip,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  AttachMoney,
  ShowChart,
  Assessment,
} from '@mui/icons-material';
import { useThemeColors } from '../../hooks/useTheme';

// Utility function for formatting financial numbers
const formatFinancialNumber = (value: number, options?: { currency?: boolean; compact?: boolean }) => {
  const { currency = true, compact = false } = options || {};
  
  if (compact && Math.abs(value) >= 1000000) {
    return (currency ? '$' : '') + (value / 1000000).toFixed(1) + 'M';
  } else if (compact && Math.abs(value) >= 1000) {
    return (currency ? '$' : '') + (value / 1000).toFixed(1) + 'K';
  }
  
  return (currency ? '$' : '') + value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
};

interface PortfolioData {
  totalValue: number;
  dayChange: number;
  dayChangePercent: number;
  totalReturn: number;
  totalReturnPercent: number;
  cash: number;
  positions: number;
}

interface PortfolioOverviewCardProps {
  data?: PortfolioData;
  isLoading?: boolean;
}

const PortfolioOverviewCard: React.FC<PortfolioOverviewCardProps> = ({
  data,
  isLoading = false,
}) => {
  const theme = useTheme();
  const { getFinancialColor } = useThemeColors();

  if (isLoading || !data) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Avatar sx={{ bgcolor: theme.palette.primary.main, mr: 2 }}>
              <AccountBalance />
            </Avatar>
            <Box sx={{ flex: 1 }}>
              <Skeleton variant="text" width="40%" height={32} />
              <Skeleton variant="text" width="60%" height={24} />
            </Box>
          </Box>

          <Grid container spacing={3}>
            {[...Array(4)].map((_, index) => (
              <Grid item xs={6} md={3} key={index}>
                <Box>
                  <Skeleton variant="text" width="80%" height={24} />
                  <Skeleton variant="text" width="60%" height={32} />
                  <Skeleton variant="text" width="40%" height={20} />
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    );
  }

  const dayChangeColor = getFinancialColor(data.dayChange);
  const totalReturnColor = getFinancialColor(data.totalReturn);

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Avatar sx={{ bgcolor: theme.palette.primary.main, mr: 2 }}>
            <AccountBalance />
          </Avatar>
          <Box>
            <Typography variant="h6" component="div">
              Portfolio Overview
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Real-time portfolio performance
            </Typography>
          </Box>
        </Box>

        {/* Total Portfolio Value */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="h3" component="div" sx={{ fontWeight: 700, mb: 1 }}>
            {formatFinancialNumber(data.totalValue, { currency: true, compact: true })}
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {data.dayChange >= 0 ? (
                <TrendingUp sx={{ color: dayChangeColor, fontSize: 20 }} />
              ) : (
                <TrendingDown sx={{ color: dayChangeColor, fontSize: 20 }} />
              )}
              <Typography
                variant="h6"
                sx={{ 
                  color: dayChangeColor, 
                  fontWeight: 600,
                }}
              >
                {data.dayChange >= 0 ? '+' : ''}
                {formatFinancialNumber(data.dayChange, { currency: true, decimals: 2 })}
              </Typography>
              <Typography
                variant="body1"
                sx={{ 
                  color: dayChangeColor, 
                  fontWeight: 500,
                }}
              >
                ({data.dayChangePercent >= 0 ? '+' : ''}
                {formatFinancialNumber(data.dayChangePercent, { percentage: true, decimals: 2 })})
              </Typography>
            </Box>
            
            <Chip
              label="Today"
              size="small"
              variant="outlined"
              sx={{ color: dayChangeColor, borderColor: dayChangeColor }}
            />
          </Box>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Metrics Grid */}
        <Grid container spacing={3}>
          {/* Total Return */}
          <Grid item xs={6} md={3}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Total Return
              </Typography>
              <Typography
                variant="h6"
                sx={{ 
                  color: totalReturnColor, 
                  fontWeight: 600,
                  mb: 0.5,
                }}
              >
                {data.totalReturn >= 0 ? '+' : ''}
                {formatFinancialNumber(data.totalReturn, { currency: true, compact: true })}
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: totalReturnColor }}
              >
                {data.totalReturnPercent >= 0 ? '+' : ''}
                {formatFinancialNumber(data.totalReturnPercent, { percentage: true, decimals: 1 })}
              </Typography>
            </Box>
          </Grid>

          {/* Available Cash */}
          <Grid item xs={6} md={3}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Available Cash
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                {formatFinancialNumber(data.cash, { currency: true, compact: true })}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {formatFinancialNumber((data.cash / data.totalValue) * 100, { percentage: true, decimals: 1 })} of portfolio
              </Typography>
            </Box>
          </Grid>

          {/* Positions */}
          <Grid item xs={6} md={3}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Positions
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                {data.positions}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active holdings
              </Typography>
            </Box>
          </Grid>

          {/* Invested Amount */}
          <Grid item xs={6} md={3}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Invested
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                {formatFinancialNumber(data.totalValue - data.cash, { currency: true, compact: true })}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {formatFinancialNumber(((data.totalValue - data.cash) / data.totalValue) * 100, { percentage: true, decimals: 1 })} allocated
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Action Icons */}
        <Box sx={{ 
          mt: 3, 
          pt: 2, 
          borderTop: `1px solid ${theme.palette.divider}`,
          display: 'flex',
          gap: 2,
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ShowChart sx={{ fontSize: 20, color: theme.palette.text.secondary }} />
            <Typography variant="body2" color="text.secondary">
              Performance
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AttachMoney sx={{ fontSize: 20, color: theme.palette.text.secondary }} />
            <Typography variant="body2" color="text.secondary">
              Holdings
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Assessment sx={{ fontSize: 20, color: theme.palette.text.secondary }} />
            <Typography variant="body2" color="text.secondary">
              Analytics
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PortfolioOverviewCard;
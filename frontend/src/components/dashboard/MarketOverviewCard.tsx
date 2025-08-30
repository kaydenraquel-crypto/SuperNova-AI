import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  MoreVert,
  ShowChart,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface MarketData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
}

interface MarketOverviewCardProps {
  loading?: boolean;
}

const MarketOverviewCard: React.FC<MarketOverviewCardProps> = ({
  loading = false,
}) => {
  const theme = useTheme();
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  useEffect(() => {
    // Mock data - replace with actual API call
    setMarketData([
      {
        symbol: 'SPY',
        name: 'SPDR S&P 500 ETF',
        price: 445.23,
        change: 2.14,
        changePercent: 0.48,
        volume: 45234567,
      },
      {
        symbol: 'QQQ',
        name: 'Invesco QQQ Trust',
        price: 382.45,
        change: -1.23,
        changePercent: -0.32,
        volume: 32145678,
      },
      {
        symbol: 'AAPL',
        name: 'Apple Inc.',
        price: 175.84,
        change: 0.92,
        changePercent: 0.53,
        volume: 12345678,
      },
    ]);
  }, []);

  const getChangeColor = (change: number) => {
    if (change > 0) return theme.palette.success.main;
    if (change < 0) return theme.palette.error.main;
    return theme.palette.text.secondary;
  };

  const formatNumber = (num: number, currency = false) => {
    if (currency) {
      return `$${num.toFixed(2)}`;
    }
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toFixed(2);
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="h3">
            Market Overview
          </Typography>
          <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
            <MoreVert />
          </IconButton>
        </Box>

        <Grid container spacing={2}>
          {marketData.map((stock) => (
            <Grid item xs={12} key={stock.symbol}>
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                p: 1,
                borderRadius: 1,
                '&:hover': {
                  bgcolor: theme.palette.action.hover,
                },
              }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle2" fontWeight="bold">
                    {stock.symbol}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {stock.name}
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'right', minWidth: 80 }}>
                  <Typography variant="body2" fontWeight="bold">
                    {formatNumber(stock.price, true)}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                    {stock.change >= 0 ? (
                      <TrendingUp sx={{ fontSize: 16, color: getChangeColor(stock.change) }} />
                    ) : (
                      <TrendingDown sx={{ fontSize: 16, color: getChangeColor(stock.change) }} />
                    )}
                    <Typography 
                      variant="caption" 
                      sx={{ color: getChangeColor(stock.change), fontWeight: 500 }}
                    >
                      {stock.change >= 0 ? '+' : ''}{formatNumber(stock.changePercent)}%
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="caption" color="text.secondary">
            Last updated: {new Date().toLocaleTimeString()}
          </Typography>
        </Box>

        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={() => setMenuAnchor(null)}>
            View Details
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Add to Watchlist
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Settings
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default MarketOverviewCard;
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
  Button,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Add,
  Remove,
  Star,
  StarBorder,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface WatchlistItem {
  id: string;
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  favorite: boolean;
}

interface WatchlistCardProps {
  loading?: boolean;
}

const WatchlistCard: React.FC<WatchlistCardProps> = ({
  loading = false,
}) => {
  const theme = useTheme();
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([
    {
      id: '1',
      symbol: 'TSLA',
      name: 'Tesla, Inc.',
      price: 248.42,
      change: 12.34,
      changePercent: 5.23,
      favorite: true,
    },
    {
      id: '2',
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      price: 721.33,
      change: -8.45,
      changePercent: -1.16,
      favorite: false,
    },
    {
      id: '3',
      symbol: 'AMZN',
      name: 'Amazon.com, Inc.',
      price: 153.61,
      change: 2.17,
      changePercent: 1.43,
      favorite: true,
    },
  ]);

  const getChangeColor = (change: number) => {
    if (change > 0) return theme.palette.success.main;
    if (change < 0) return theme.palette.error.main;
    return theme.palette.text.secondary;
  };

  const toggleFavorite = (id: string) => {
    setWatchlist(prev => 
      prev.map(item => 
        item.id === id ? { ...item, favorite: !item.favorite } : item
      )
    );
  };

  const removeFromWatchlist = (id: string) => {
    setWatchlist(prev => prev.filter(item => item.id !== id));
  };

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="h3">
            Watchlist
          </Typography>
          <Button
            startIcon={<Add />}
            size="small"
            variant="outlined"
            sx={{ minWidth: 'auto' }}
          >
            Add
          </Button>
        </Box>

        <List disablePadding>
          {watchlist.map((item) => (
            <ListItem
              key={item.id}
              sx={{
                px: 0,
                borderBottom: 1,
                borderColor: 'divider',
                '&:last-child': {
                  borderBottom: 0,
                },
              }}
            >
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {item.symbol}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={() => toggleFavorite(item.id)}
                      sx={{ p: 0.25 }}
                    >
                      {item.favorite ? (
                        <Star sx={{ fontSize: 16, color: theme.palette.warning.main }} />
                      ) : (
                        <StarBorder sx={{ fontSize: 16 }} />
                      )}
                    </IconButton>
                  </Box>
                }
                secondary={
                  <Typography variant="caption" color="text.secondary">
                    {item.name}
                  </Typography>
                }
              />
              
              <Box sx={{ textAlign: 'right', minWidth: 100 }}>
                <Typography variant="body2" fontWeight="bold">
                  {formatPrice(item.price)}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                  {item.change >= 0 ? (
                    <TrendingUp sx={{ fontSize: 14, color: getChangeColor(item.change) }} />
                  ) : (
                    <TrendingDown sx={{ fontSize: 14, color: getChangeColor(item.change) }} />
                  )}
                  <Typography 
                    variant="caption" 
                    sx={{ color: getChangeColor(item.change), fontWeight: 500 }}
                  >
                    {item.change >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
                  </Typography>
                </Box>
              </Box>

              <IconButton
                size="small"
                onClick={() => removeFromWatchlist(item.id)}
                sx={{ ml: 1 }}
              >
                <Remove sx={{ fontSize: 18 }} />
              </IconButton>
            </ListItem>
          ))}
        </List>

        {watchlist.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              No items in watchlist
            </Typography>
            <Button
              startIcon={<Add />}
              size="small"
              variant="outlined"
              sx={{ mt: 1 }}
            >
              Add Your First Stock
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default WatchlistCard;
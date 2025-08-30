import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  Chip,
  Avatar,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  MoreVert,
  Receipt,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import dayjs from 'dayjs';

interface Transaction {
  id: string;
  type: 'buy' | 'sell';
  symbol: string;
  quantity: number;
  price: number;
  total: number;
  timestamp: string;
  status: 'completed' | 'pending' | 'failed';
}

interface RecentTransactionsCardProps {
  loading?: boolean;
}

const RecentTransactionsCard: React.FC<RecentTransactionsCardProps> = ({
  loading = false,
}) => {
  const theme = useTheme();
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [transactions] = useState<Transaction[]>([
    {
      id: '1',
      type: 'buy',
      symbol: 'AAPL',
      quantity: 10,
      price: 175.84,
      total: 1758.40,
      timestamp: '2024-01-15T10:30:00Z',
      status: 'completed',
    },
    {
      id: '2',
      type: 'sell',
      symbol: 'TSLA',
      quantity: 5,
      price: 248.42,
      total: 1242.10,
      timestamp: '2024-01-15T09:15:00Z',
      status: 'completed',
    },
    {
      id: '3',
      type: 'buy',
      symbol: 'NVDA',
      quantity: 2,
      price: 721.33,
      total: 1442.66,
      timestamp: '2024-01-14T16:45:00Z',
      status: 'pending',
    },
    {
      id: '4',
      type: 'sell',
      symbol: 'AMZN',
      quantity: 8,
      price: 153.61,
      total: 1228.88,
      timestamp: '2024-01-14T14:20:00Z',
      status: 'completed',
    },
  ]);

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case 'buy':
        return <TrendingUp />;
      case 'sell':
        return <TrendingDown />;
      default:
        return <Receipt />;
    }
  };

  const getTransactionColor = (type: string) => {
    switch (type) {
      case 'buy':
        return theme.palette.success.main;
      case 'sell':
        return theme.palette.error.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'pending':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatTime = (timestamp: string) => {
    return dayjs(timestamp).format('MMM DD, HH:mm');
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" component="h3">
            Recent Transactions
          </Typography>
          <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
            <MoreVert />
          </IconButton>
        </Box>

        <List disablePadding>
          {transactions.map((transaction) => (
            <ListItem
              key={transaction.id}
              sx={{
                px: 0,
                py: 1.5,
                borderBottom: 1,
                borderColor: 'divider',
                '&:last-child': {
                  borderBottom: 0,
                },
              }}
            >
              <Avatar
                sx={{ 
                  bgcolor: getTransactionColor(transaction.type),
                  width: 36,
                  height: 36,
                  mr: 2,
                }}
              >
                {getTransactionIcon(transaction.type)}
              </Avatar>

              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {transaction.type.toUpperCase()} {transaction.symbol}
                    </Typography>
                    <Chip
                      label={transaction.status}
                      size="small"
                      color={getStatusColor(transaction.status) as any}
                      variant="outlined"
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {transaction.quantity} shares @ {formatCurrency(transaction.price)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTime(transaction.timestamp)}
                    </Typography>
                  </Box>
                }
              />

              <Box sx={{ textAlign: 'right' }}>
                <Typography 
                  variant="body2" 
                  fontWeight="bold"
                  sx={{ color: getTransactionColor(transaction.type) }}
                >
                  {transaction.type === 'buy' ? '-' : '+'}{formatCurrency(transaction.total)}
                </Typography>
              </Box>
            </ListItem>
          ))}
        </List>

        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={() => setMenuAnchor(null)}>
            View All Transactions
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Export to CSV
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Transaction Settings
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default RecentTransactionsCard;
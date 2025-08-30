import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Alert,
  Breadcrumbs,
  Link,
  Grid,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Fab
} from '@mui/material';
import {
  Analytics,
  Home,
  Add,
  TrendingUp,
  Assessment,
  PieChart
} from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import AdvancedAnalyticsDashboard from '../components/analytics/AdvancedAnalyticsDashboard';

interface Portfolio {
  id: number;
  name: string;
  currency: string;
  initial_value: number;
  is_active: boolean;
  is_paper_trading: boolean;
}

const AnalyticsPage: React.FC = () => {
  const { user } = useAuth();
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newPortfolioName, setNewPortfolioName] = useState('');
  const [newPortfolioInitialValue, setNewPortfolioInitialValue] = useState('10000');
  const [newPortfolioCurrency, setNewPortfolioCurrency] = useState('USD');

  // Fetch user portfolios
  const fetchPortfolios = async () => {
    try {
      setLoading(true);
      setError(null);

      // For demonstration, create synthetic portfolio data
      // In production, this would fetch from /api/portfolios or similar
      const syntheticPortfolios: Portfolio[] = [
        {
          id: 1,
          name: 'Growth Portfolio',
          currency: 'USD',
          initial_value: 50000,
          is_active: true,
          is_paper_trading: false
        },
        {
          id: 2,
          name: 'Conservative Portfolio', 
          currency: 'USD',
          initial_value: 25000,
          is_active: true,
          is_paper_trading: true
        },
        {
          id: 3,
          name: 'Tech Stocks',
          currency: 'USD',
          initial_value: 15000,
          is_active: true,
          is_paper_trading: false
        }
      ];

      setTimeout(() => {
        setPortfolios(syntheticPortfolios);
        if (syntheticPortfolios.length > 0 && !selectedPortfolio) {
          setSelectedPortfolio(syntheticPortfolios[0].id);
        }
        setLoading(false);
      }, 1000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load portfolios');
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolios();
  }, []);

  const handleCreatePortfolio = async () => {
    try {
      // In production, this would create a new portfolio via API
      const newId = Math.max(...portfolios.map(p => p.id)) + 1;
      const newPortfolio: Portfolio = {
        id: newId,
        name: newPortfolioName,
        currency: newPortfolioCurrency,
        initial_value: parseFloat(newPortfolioInitialValue),
        is_active: true,
        is_paper_trading: true
      };

      setPortfolios(prev => [...prev, newPortfolio]);
      setSelectedPortfolio(newId);
      setCreateDialogOpen(false);
      setNewPortfolioName('');
      setNewPortfolioInitialValue('10000');
      setNewPortfolioCurrency('USD');

    } catch (err) {
      console.error('Error creating portfolio:', err);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      {/* Breadcrumbs */}
      <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 3 }}>
        <Link
          component={RouterLink}
          to="/dashboard"
          sx={{ display: 'flex', alignItems: 'center' }}
          color="inherit"
        >
          <Home sx={{ mr: 0.5 }} fontSize="inherit" />
          Dashboard
        </Link>
        <Typography
          sx={{ display: 'flex', alignItems: 'center' }}
          color="text.primary"
        >
          <Analytics sx={{ mr: 0.5 }} fontSize="inherit" />
          Advanced Analytics
        </Typography>
      </Breadcrumbs>

      {/* Page Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <Analytics sx={{ mr: 2, verticalAlign: 'middle' }} />
          Advanced Analytics
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Comprehensive portfolio analysis, risk metrics, and performance attribution
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Portfolio Selection */}
      {portfolios.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <Typography variant="h6">
                Select Portfolio:
              </Typography>
            </Grid>
            {portfolios.map((portfolio) => (
              <Grid item key={portfolio.id}>
                <Card
                  sx={{
                    cursor: 'pointer',
                    border: selectedPortfolio === portfolio.id ? 2 : 1,
                    borderColor: selectedPortfolio === portfolio.id ? 'primary.main' : 'divider',
                    '&:hover': {
                      borderColor: 'primary.main',
                      boxShadow: 1
                    }
                  }}
                  onClick={() => setSelectedPortfolio(portfolio.id)}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {portfolio.is_paper_trading ? <Assessment /> : <TrendingUp />}
                      <Box>
                        <Typography variant="body2" fontWeight="medium">
                          {portfolio.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {portfolio.currency} {portfolio.initial_value.toLocaleString()}
                          {portfolio.is_paper_trading && ' (Paper)'}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Analytics Dashboard */}
      {selectedPortfolio ? (
        <AdvancedAnalyticsDashboard
          portfolioId={selectedPortfolio}
          refreshInterval={300000} // 5 minutes
        />
      ) : portfolios.length === 0 && !loading ? (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 8 }}>
            <PieChart sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No Portfolios Found
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Create your first portfolio to start using advanced analytics
            </Typography>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setCreateDialogOpen(true)}
            >
              Create Portfolio
            </Button>
          </CardContent>
        </Card>
      ) : null}

      {/* Create Portfolio FAB */}
      <Fab
        color="primary"
        aria-label="create portfolio"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        onClick={() => setCreateDialogOpen(true)}
      >
        <Add />
      </Fab>

      {/* Create Portfolio Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create New Portfolio</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              autoFocus
              margin="dense"
              label="Portfolio Name"
              fullWidth
              variant="outlined"
              value={newPortfolioName}
              onChange={(e) => setNewPortfolioName(e.target.value)}
              sx={{ mb: 2 }}
            />
            
            <TextField
              margin="dense"
              label="Initial Value"
              fullWidth
              variant="outlined"
              type="number"
              value={newPortfolioInitialValue}
              onChange={(e) => setNewPortfolioInitialValue(e.target.value)}
              sx={{ mb: 2 }}
            />
            
            <FormControl fullWidth variant="outlined">
              <InputLabel>Currency</InputLabel>
              <Select
                value={newPortfolioCurrency}
                onChange={(e) => setNewPortfolioCurrency(e.target.value)}
                label="Currency"
              >
                <MenuItem value="USD">USD - US Dollar</MenuItem>
                <MenuItem value="EUR">EUR - Euro</MenuItem>
                <MenuItem value="GBP">GBP - British Pound</MenuItem>
                <MenuItem value="JPY">JPY - Japanese Yen</MenuItem>
                <MenuItem value="CAD">CAD - Canadian Dollar</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleCreatePortfolio}
            variant="contained"
            disabled={!newPortfolioName.trim()}
          >
            Create Portfolio
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AnalyticsPage;
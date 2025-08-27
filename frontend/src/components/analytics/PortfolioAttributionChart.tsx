import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { PieChart, Assessment } from '@mui/icons-material';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart as RechartsPieChart,
  Pie,
  Cell
} from 'recharts';
import { formatFinancialNumber } from '@/theme';

interface AttributionData {
  total_excess_return: number;
  asset_allocation_effect: number;
  security_selection_effect: number;
  interaction_effect: number;
  attribution_quality: number;
  portfolio_return: number;
  benchmark_return: number;
}

interface PortfolioAttributionChartProps {
  portfolioId: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const PortfolioAttributionChart: React.FC<PortfolioAttributionChartProps> = ({ portfolioId }) => {
  const [attributionData, setAttributionData] = useState<AttributionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch attribution data
  useEffect(() => {
    const fetchAttributionData = async () => {
      try {
        setLoading(true);
        setError(null);

        // For demonstration, create synthetic attribution data
        // In production, this would fetch from /api/analytics/portfolio/{id}/attribution
        const syntheticData: AttributionData = {
          total_excess_return: 0.034,
          asset_allocation_effect: 0.012,
          security_selection_effect: 0.018,
          interaction_effect: 0.004,
          attribution_quality: 0.85,
          portfolio_return: 0.087,
          benchmark_return: 0.053
        };

        setTimeout(() => {
          setAttributionData(syntheticData);
          setLoading(false);
        }, 1000);

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load attribution data');
        setLoading(false);
      }
    };

    fetchAttributionData();
  }, [portfolioId]);

  // Prepare data for charts
  const attributionBreakdown = attributionData ? [
    {
      name: 'Asset Allocation',
      value: attributionData.asset_allocation_effect * 100,
      description: 'Effect of over/underweighting asset classes'
    },
    {
      name: 'Security Selection',
      value: attributionData.security_selection_effect * 100,
      description: 'Effect of picking outperforming securities'
    },
    {
      name: 'Interaction',
      value: attributionData.interaction_effect * 100,
      description: 'Interaction between allocation and selection'
    }
  ] : [];

  const performanceComparison = attributionData ? [
    {
      name: 'Portfolio',
      return: attributionData.portfolio_return * 100,
      type: 'portfolio'
    },
    {
      name: 'Benchmark',
      return: attributionData.benchmark_return * 100,
      type: 'benchmark'
    },
    {
      name: 'Excess Return',
      return: attributionData.total_excess_return * 100,
      type: 'excess'
    }
  ] : [];

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
            <CircularProgress />
            <Typography variant="body2" sx={{ ml: 2 }}>
              Loading attribution analysis...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error || !attributionData) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            {error || 'No attribution data available'}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Header Card */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6" component="h3">
                <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                Performance Attribution Analysis
              </Typography>
              
              <Chip
                label={`Quality: ${(attributionData.attribution_quality * 100).toFixed(0)}%`}
                color={attributionData.attribution_quality > 0.8 ? 'success' : 'warning'}
              />
            </Box>
            
            <Box sx={{ mt: 2, display: 'flex', gap: 4 }}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Portfolio Return
                </Typography>
                <Typography variant="h6" color="primary">
                  {formatFinancialNumber(attributionData.portfolio_return, { percentage: true, decimals: 2 })}
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Benchmark Return
                </Typography>
                <Typography variant="h6">
                  {formatFinancialNumber(attributionData.benchmark_return, { percentage: true, decimals: 2 })}
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Excess Return
                </Typography>
                <Typography 
                  variant="h6" 
                  color={attributionData.total_excess_return >= 0 ? 'success.main' : 'error.main'}
                >
                  {attributionData.total_excess_return >= 0 ? '+' : ''}
                  {formatFinancialNumber(attributionData.total_excess_return, { percentage: true, decimals: 2 })}
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Attribution Breakdown Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Attribution Breakdown
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={attributionBreakdown}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Contribution (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  formatter={(value: number, name) => [`${value.toFixed(3)}%`, name]}
                  labelFormatter={(label) => `${label} Effect`}
                />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
            
            <Box sx={{ mt: 2 }}>
              <Alert severity="info">
                <Typography variant="body2">
                  Attribution shows how portfolio construction decisions contributed to performance 
                  relative to benchmark.
                </Typography>
              </Alert>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Attribution Pie Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Return Attribution Sources
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              <RechartsPieChart>
                <Pie
                  data={attributionBreakdown.filter(item => item.value !== 0)}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value.toFixed(2)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {attributionBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => [`${value.toFixed(3)}%`, 'Contribution']} />
              </RechartsPieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Detailed Attribution Table */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Detailed Attribution Analysis
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Attribution Factor</TableCell>
                    <TableCell align="right">Contribution</TableCell>
                    <TableCell align="right">Impact</TableCell>
                    <TableCell>Description</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {attributionBreakdown.map((row, index) => (
                    <TableRow key={row.name}>
                      <TableCell component="th" scope="row">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              bgcolor: COLORS[index % COLORS.length],
                              borderRadius: '50%',
                              mr: 1
                            }}
                          />
                          {row.name}
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Typography
                          sx={{
                            color: row.value >= 0 ? 'success.main' : 'error.main',
                            fontWeight: 'medium'
                          }}
                        >
                          {row.value >= 0 ? '+' : ''}{row.value.toFixed(3)}%
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          size="small"
                          label={Math.abs(row.value) > 1 ? 'High' : Math.abs(row.value) > 0.5 ? 'Medium' : 'Low'}
                          color={Math.abs(row.value) > 1 ? 'error' : Math.abs(row.value) > 0.5 ? 'warning' : 'success'}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {row.description}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Attribution Quality Assessment
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="body2" sx={{ minWidth: 120 }}>
                  Explained Return:
                </Typography>
                <Box sx={{ flex: 1, mx: 2 }}>
                  <Box sx={{ 
                    height: 8, 
                    bgcolor: 'grey.300', 
                    borderRadius: 4,
                    overflow: 'hidden'
                  }}>
                    <Box sx={{ 
                      height: '100%', 
                      bgcolor: attributionData.attribution_quality > 0.8 ? 'success.main' : 'warning.main',
                      width: `${attributionData.attribution_quality * 100}%`,
                      borderRadius: 4
                    }} />
                  </Box>
                </Box>
                <Typography variant="body2">
                  {(attributionData.attribution_quality * 100).toFixed(1)}%
                </Typography>
              </Box>
              
              {attributionData.attribution_quality < 0.7 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    Low attribution quality may indicate incomplete data or model limitations. 
                    Results should be interpreted with caution.
                  </Typography>
                </Alert>
              )}
              
              {attributionData.attribution_quality >= 0.8 && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    High attribution quality indicates reliable analysis. 
                    The model explains most of the excess return variance.
                  </Typography>
                </Alert>
              )}
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Performance Comparison */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Performance Comparison
            </Typography>
            
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={performanceComparison} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip formatter={(value: number) => [`${value.toFixed(2)}%`, 'Return']} />
                <Bar 
                  dataKey="return" 
                  fill={(entry) => {
                    if (entry.type === 'portfolio') return '#8884d8';
                    if (entry.type === 'benchmark') return '#82ca9d';
                    return entry.return >= 0 ? '#00C49F' : '#FF8042';
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default PortfolioAttributionChart;
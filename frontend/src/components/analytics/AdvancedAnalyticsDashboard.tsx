import React, { useState, useEffect, useMemo } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Tab,
  Tabs,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  LinearProgress,
  Alert,
  Tooltip,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  useTheme,
  alpha
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Analytics,
  Assessment,
  PieChart,
  Timeline,
  Download,
  Refresh,
  Settings,
  Info,
  Warning,
  CheckCircle,
  Error as ErrorIcon
} from '@mui/icons-material';
import { useAuth } from '@/hooks/useAuth';
import { formatFinancialNumber } from '@/theme';
import FinancialChart from '@/components/charts/FinancialChart';
import PerformanceMetricsCard from './PerformanceMetricsCard';
import RiskAnalysisCard from './RiskAnalysisCard';
import PortfolioAttributionChart from './PortfolioAttributionChart';
import MarketSentimentWidget from './MarketSentimentWidget';

interface PortfolioPerformance {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  beta: number;
  alpha: number;
  var_95: number;
  win_rate: number;
}

interface RiskMetrics {
  value_at_risk_95: number;
  expected_shortfall_95: number;
  volatility_forecast: number;
  correlation_risk: number;
  concentration_risk: number;
}

interface AnalyticsData {
  portfolio: {
    id: number;
    name: string;
    currency: string;
    initial_value: number;
  };
  performance_metrics: PerformanceMetrics;
  risk_analysis: {
    portfolio_var: number;
    diversification_ratio: number;
    concentration_risk: number;
  };
  time_series_analysis: {
    trend_strength: number;
    trend_direction: string;
    volatility: number;
    data_quality: string;
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 0 }}>{children}</Box>}
    </div>
  );
}

interface AdvancedAnalyticsDashboardProps {
  portfolioId: number;
  refreshInterval?: number;
}

const AdvancedAnalyticsDashboard: React.FC<AdvancedAnalyticsDashboardProps> = ({
  portfolioId,
  refreshInterval = 300000 // 5 minutes
}) => {
  const theme = useTheme();
  const { user } = useAuth();
  const [tabValue, setTabValue] = useState(0);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  
  // Fetch analytics data
  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch performance metrics
      const performanceResponse = await fetch(
        `/api/analytics/portfolio/${portfolioId}/performance`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (!performanceResponse.ok) {
        throw new Error('Failed to fetch performance data');
      }
      
      const performanceData = await performanceResponse.json();
      
      // Fetch risk analysis
      const riskResponse = await fetch(
        `/api/analytics/portfolio/${portfolioId}/risk`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (!riskResponse.ok) {
        throw new Error('Failed to fetch risk data');
      }
      
      const riskData = await riskResponse.json();
      
      // Combine data
      setAnalyticsData({
        portfolio: performanceData.portfolio,
        performance_metrics: performanceData.performance_metrics,
        risk_analysis: riskData.risk_analysis,
        time_series_analysis: performanceData.time_series_analysis
      });
      
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };
  
  // Initial load and refresh interval
  useEffect(() => {
    fetchAnalyticsData();
    
    const interval = setInterval(fetchAnalyticsData, refreshInterval);
    return () => clearInterval(interval);
  }, [portfolioId, refreshInterval]);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Generate report
  const handleGenerateReport = async (reportType: string, format: string) => {
    try {
      const response = await fetch('/api/analytics/reports/generate', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          portfolio_id: portfolioId,
          report_type: reportType,
          format: format,
          include_benchmarks: true,
          include_attribution: true
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate report');
      }
      
      const result = await response.json();
      
      // Show success message and provide download link
      // This would typically show a notification
      console.log('Report generation started:', result);
    } catch (err) {
      console.error('Error generating report:', err);
    }
    
    setMenuAnchor(null);
  };
  
  // Memoized performance summary
  const performanceSummary = useMemo(() => {
    if (!analyticsData) return null;
    
    const { performance_metrics } = analyticsData;
    
    return {
      totalReturn: performance_metrics.total_return,
      sharpeRatio: performance_metrics.sharpe_ratio,
      maxDrawdown: performance_metrics.max_drawdown,
      volatility: performance_metrics.volatility,
      isPositive: performance_metrics.total_return >= 0
    };
  }, [analyticsData]);
  
  // Data quality indicator
  const getDataQualityColor = (quality: string) => {
    switch (quality) {
      case 'high': return theme.palette.success.main;
      case 'medium': return theme.palette.warning.main;
      case 'low': return theme.palette.error.main;
      default: return theme.palette.grey[500];
    }
  };
  
  const getDataQualityIcon = (quality: string) => {
    switch (quality) {
      case 'high': return <CheckCircle />;
      case 'medium': return <Warning />;
      case 'low': return <ErrorIcon />;
      default: return <Info />;
    }
  };
  
  if (loading && !analyticsData) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
          Loading advanced analytics...
        </Typography>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Alert 
        severity="error" 
        action={
          <IconButton size="small" onClick={fetchAnalyticsData}>
            <Refresh />
          </IconButton>
        }
      >
        {error}
      </Alert>
    );
  }
  
  if (!analyticsData) {
    return (
      <Alert severity="info">
        No analytics data available for this portfolio.
      </Alert>
    );
  }
  
  return (
    <Box sx={{ width: '100%', bgcolor: 'background.default', minHeight: '100vh' }}>
      {/* Header */}
      <Box sx={{ 
        background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
        color: 'white',
        p: 3,
        mb: 3
      }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              <Analytics sx={{ mr: 2, verticalAlign: 'middle' }} />
              Advanced Analytics
            </Typography>
            <Typography variant="h6" sx={{ opacity: 0.9 }}>
              {analyticsData.portfolio.name}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Data Quality Indicator */}
            <Tooltip title={`Data Quality: ${analyticsData.time_series_analysis.data_quality}`}>
              <Chip
                icon={getDataQualityIcon(analyticsData.time_series_analysis.data_quality)}
                label={analyticsData.time_series_analysis.data_quality.toUpperCase()}
                sx={{
                  bgcolor: alpha(getDataQualityColor(analyticsData.time_series_analysis.data_quality), 0.2),
                  color: 'white',
                  '& .MuiChip-icon': { color: 'white' }
                }}
              />
            </Tooltip>
            
            {/* Last Update */}
            {lastUpdate && (
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                Last updated: {lastUpdate.toLocaleTimeString()}
              </Typography>
            )}
            
            {/* Actions */}
            <IconButton 
              color="inherit" 
              onClick={fetchAnalyticsData}
              disabled={loading}
            >
              <Refresh />
            </IconButton>
            
            <IconButton 
              color="inherit"
              onClick={(e) => setMenuAnchor(e.currentTarget)}
            >
              <Download />
            </IconButton>
            
            <IconButton 
              color="inherit"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings />
            </IconButton>
          </Box>
        </Box>
        
        {/* Performance Summary Cards */}
        {performanceSummary && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ bgcolor: alpha('white', 0.1), color: 'white' }}>
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" sx={{ opacity: 0.8 }}>
                        Total Return
                      </Typography>
                      <Typography variant="h5">
                        {formatFinancialNumber(performanceSummary.totalReturn, { percentage: true, decimals: 2 })}
                      </Typography>
                    </Box>
                    {performanceSummary.isPositive ? <TrendingUp /> : <TrendingDown />}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ bgcolor: alpha('white', 0.1), color: 'white' }}>
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>
                    Sharpe Ratio
                  </Typography>
                  <Typography variant="h5">
                    {performanceSummary.sharpeRatio.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ bgcolor: alpha('white', 0.1), color: 'white' }}>
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>
                    Max Drawdown
                  </Typography>
                  <Typography variant="h5">
                    {formatFinancialNumber(performanceSummary.maxDrawdown, { percentage: true, decimals: 2 })}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ bgcolor: alpha('white', 0.1), color: 'white' }}>
                <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                  <Typography variant="body2" sx={{ opacity: 0.8 }}>
                    Volatility
                  </Typography>
                  <Typography variant="h5">
                    {formatFinancialNumber(performanceSummary.volatility, { percentage: true, decimals: 1 })}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </Box>
      
      {/* Main Content */}
      <Box sx={{ px: 3 }}>
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={tabValue} onChange={handleTabChange} variant="scrollable">
            <Tab icon={<Assessment />} label="Performance" />
            <Tab icon={<Timeline />} label="Risk Analysis" />
            <Tab icon={<PieChart />} label="Attribution" />
            <Tab icon={<TrendingUp />} label="Market Sentiment" />
          </Tabs>
        </Box>
        
        {/* Tab Panels */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <PerformanceMetricsCard 
                performanceData={analyticsData.performance_metrics}
                portfolioInfo={analyticsData.portfolio}
              />
            </Grid>
            <Grid item xs={12} lg={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Time Series Analysis
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Trend Strength:</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {(analyticsData.time_series_analysis.trend_strength * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={analyticsData.time_series_analysis.trend_strength * 100}
                      sx={{ mb: 2 }}
                    />
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Trend Direction:</Typography>
                      <Chip 
                        label={analyticsData.time_series_analysis.trend_direction}
                        size="small"
                        color={
                          analyticsData.time_series_analysis.trend_direction === 'upward' 
                            ? 'success' 
                            : analyticsData.time_series_analysis.trend_direction === 'downward'
                            ? 'error'
                            : 'default'
                        }
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <RiskAnalysisCard 
                riskData={analyticsData.risk_analysis}
                portfolioId={portfolioId}
              />
            </Grid>
            <Grid item xs={12} lg={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Risk Summary
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">Portfolio VaR:</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {formatFinancialNumber(analyticsData.risk_analysis.portfolio_var, { percentage: true, decimals: 2 })}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">Diversification Ratio:</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {analyticsData.risk_analysis.diversification_ratio.toFixed(2)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">Concentration Risk:</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {formatFinancialNumber(analyticsData.risk_analysis.concentration_risk, { percentage: true, decimals: 1 })}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <PortfolioAttributionChart portfolioId={portfolioId} />
        </TabPanel>
        
        <TabPanel value={tabValue} index={3}>
          <MarketSentimentWidget />
        </TabPanel>
      </Box>
      
      {/* Download Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={() => setMenuAnchor(null)}
      >
        <MenuItem onClick={() => handleGenerateReport('performance', 'pdf')}>
          Performance Report (PDF)
        </MenuItem>
        <MenuItem onClick={() => handleGenerateReport('risk', 'pdf')}>
          Risk Analysis Report (PDF)
        </MenuItem>
        <MenuItem onClick={() => handleGenerateReport('summary', 'xlsx')}>
          Executive Summary (Excel)
        </MenuItem>
        <MenuItem onClick={() => handleGenerateReport('allocation', 'csv')}>
          Allocation Data (CSV)
        </MenuItem>
      </Menu>
      
      {/* Settings Dialog */}
      <Dialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Analytics Settings</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary">
            Advanced analytics settings will be available in future updates.
          </Typography>
        </DialogContent>
      </Dialog>
      
      {/* Refresh FAB */}
      <Fab
        color="primary"
        size="small"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        onClick={fetchAnalyticsData}
        disabled={loading}
      >
        <Refresh />
      </Fab>
    </Box>
  );
};

export default AdvancedAnalyticsDashboard;
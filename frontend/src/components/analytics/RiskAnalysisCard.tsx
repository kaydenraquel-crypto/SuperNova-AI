import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  CircularProgress,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Warning,
  Security,
  TrendingDown,
  Info,
  Refresh,
  ShowChart
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { formatFinancialNumber } from '@/theme';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ScatterChart,
  Scatter
} from 'recharts';

interface RiskAnalysis {
  portfolio_var: number;
  component_var: Record<string, number>;
  marginal_var: Record<string, number>;
  correlation_matrix: Record<string, Record<string, number>>;
  diversification_ratio: number;
  concentration_risk: number;
}

interface RiskModel {
  value_at_risk_95: number;
  value_at_risk_99: number;
  expected_shortfall_95: number;
  expected_shortfall_99: number;
  volatility_forecast: number;
  correlation_risk: number;
  concentration_risk: number;
}

interface RiskAnalysisCardProps {
  riskData: RiskAnalysis;
  portfolioId: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658'];

const RiskAnalysisCard: React.FC<RiskAnalysisCardProps> = ({ riskData, portfolioId }) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [riskModel, setRiskModel] = useState<RiskModel | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Fetch additional risk model data
  useEffect(() => {
    const fetchRiskModel = async () => {
      try {
        setLoading(true);
        // This would typically fetch from the risk API endpoint
        // For demonstration, we'll create synthetic data
        const syntheticRiskModel: RiskModel = {
          value_at_risk_95: -0.035,
          value_at_risk_99: -0.055,
          expected_shortfall_95: -0.045,
          expected_shortfall_99: -0.070,
          volatility_forecast: 0.18,
          correlation_risk: 0.25,
          concentration_risk: 0.15
        };
        
        setTimeout(() => {
          setRiskModel(syntheticRiskModel);
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error('Error fetching risk model:', error);
        setLoading(false);
      }
    };
    
    fetchRiskModel();
  }, [portfolioId]);
  
  // Prepare component VaR data for pie chart
  const componentVarData = useMemo(() => {
    return Object.entries(riskData.component_var || {}).map(([symbol, var_contribution], index) => ({
      name: symbol,
      value: Math.abs(var_contribution) * 100,
      percentage: (Math.abs(var_contribution) / Math.abs(riskData.portfolio_var)) * 100
    }));
  }, [riskData.component_var, riskData.portfolio_var]);
  
  // Prepare marginal VaR data for bar chart
  const marginalVarData = useMemo(() => {
    return Object.entries(riskData.marginal_var || {}).map(([symbol, marginal_var]) => ({
      symbol,
      marginalVar: Math.abs(marginal_var) * 100,
      componentVar: Math.abs(riskData.component_var?.[symbol] || 0) * 100
    }));
  }, [riskData.marginal_var, riskData.component_var]);
  
  // Risk level assessment
  const getRiskLevel = (diversificationRatio: number, concentrationRisk: number) => {
    const avgRisk = (2 - diversificationRatio) + concentrationRisk;
    
    if (avgRisk > 1.5) return { level: 'High', color: theme.palette.error.main };
    if (avgRisk > 1.0) return { level: 'Medium', color: theme.palette.warning.main };
    return { level: 'Low', color: theme.palette.success.main };
  };
  
  const riskAssessment = getRiskLevel(riskData.diversification_ratio, riskData.concentration_risk);
  
  // Correlation heatmap data preparation
  const correlationData = useMemo(() => {
    const matrix = riskData.correlation_matrix || {};
    const symbols = Object.keys(matrix);
    const data = [];
    
    for (let i = 0; i < symbols.length; i++) {
      for (let j = 0; j < symbols.length; j++) {
        if (i !== j) { // Exclude self-correlations
          data.push({
            x: i,
            y: j,
            symbol1: symbols[i],
            symbol2: symbols[j],
            correlation: matrix[symbols[i]]?.[symbols[j]] || 0
          });
        }
      }
    }
    
    return data;
  }, [riskData.correlation_matrix]);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" component="h3">
            <Security sx={{ mr: 1, verticalAlign: 'middle' }} />
            Risk Analysis
          </Typography>
          
          {/* Risk Level Indicator */}
          <Box sx={{ textAlign: 'center' }}>
            <Chip
              label={`${riskAssessment.level} Risk`}
              sx={{
                backgroundColor: riskAssessment.color,
                color: 'white',
                fontWeight: 'bold'
              }}
            />
          </Box>
        </Box>
        
        {/* Risk Summary Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={4}>
            <Box sx={{ 
              p: 2, 
              bgcolor: theme.palette.background.paper,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1
            }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Portfolio VaR (95%)
              </Typography>
              <Typography variant="h6" sx={{ color: theme.palette.error.main }}>
                {formatFinancialNumber(Math.abs(riskData.portfolio_var), { percentage: true, decimals: 2 })}
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Box sx={{ 
              p: 2, 
              bgcolor: theme.palette.background.paper,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1
            }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Diversification Ratio
              </Typography>
              <Typography variant="h6" sx={{ 
                color: riskData.diversification_ratio > 1.5 
                  ? theme.palette.success.main 
                  : theme.palette.warning.main 
              }}>
                {riskData.diversification_ratio.toFixed(2)}
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Box sx={{ 
              p: 2, 
              bgcolor: theme.palette.background.paper,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1
            }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Concentration Risk
              </Typography>
              <Typography variant="h6" sx={{ 
                color: riskData.concentration_risk < 0.3 
                  ? theme.palette.success.main 
                  : theme.palette.error.main 
              }}>
                {formatFinancialNumber(riskData.concentration_risk, { percentage: true, decimals: 1 })}
              </Typography>
            </Box>
          </Grid>
        </Grid>
        
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs value={activeTab} onChange={handleTabChange} variant="scrollable">
            <Tab label="Risk Decomposition" />
            <Tab label="VaR Analysis" />
            <Tab label="Correlations" />
            <Tab label="Risk Forecast" />
          </Tabs>
        </Box>
        
        {/* Tab Content */}
        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Risk Contribution by Asset
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={componentVarData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name}: ${percentage.toFixed(1)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {componentVarData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip formatter={(value: number) => [`${value.toFixed(2)}%`, 'Risk Contribution']} />
                </PieChart>
              </ResponsiveContainer>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Marginal vs Component VaR
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={marginalVarData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="symbol" />
                  <YAxis label={{ value: 'VaR (%)', angle: -90, position: 'insideLeft' }} />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="marginalVar" fill="#8884d8" name="Marginal VaR" />
                  <Bar dataKey="componentVar" fill="#82ca9d" name="Component VaR" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        )}
        
        {activeTab === 1 && (
          <Box>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : riskModel ? (
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell><Typography variant="subtitle2">Risk Metric</Typography></TableCell>
                          <TableCell align="right"><Typography variant="subtitle2">Value</Typography></TableCell>
                          <TableCell align="center"><Typography variant="subtitle2">Status</Typography></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>Value at Risk (95%)</TableCell>
                          <TableCell align="right">
                            <Typography color="error">
                              {formatFinancialNumber(Math.abs(riskModel.value_at_risk_95), { percentage: true, decimals: 2 })}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip 
                              size="small"
                              label={Math.abs(riskModel.value_at_risk_95) > 0.05 ? "High" : "Normal"}
                              color={Math.abs(riskModel.value_at_risk_95) > 0.05 ? "error" : "success"}
                            />
                          </TableCell>
                        </TableRow>
                        
                        <TableRow>
                          <TableCell>Value at Risk (99%)</TableCell>
                          <TableCell align="right">
                            <Typography color="error">
                              {formatFinancialNumber(Math.abs(riskModel.value_at_risk_99), { percentage: true, decimals: 2 })}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip 
                              size="small"
                              label={Math.abs(riskModel.value_at_risk_99) > 0.08 ? "High" : "Normal"}
                              color={Math.abs(riskModel.value_at_risk_99) > 0.08 ? "error" : "success"}
                            />
                          </TableCell>
                        </TableRow>
                        
                        <TableRow>
                          <TableCell>Expected Shortfall (95%)</TableCell>
                          <TableCell align="right">
                            <Typography color="error">
                              {formatFinancialNumber(Math.abs(riskModel.expected_shortfall_95), { percentage: true, decimals: 2 })}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip 
                              size="small"
                              label="Tail Risk"
                              color="warning"
                            />
                          </TableCell>
                        </TableRow>
                        
                        <TableRow>
                          <TableCell>Volatility Forecast</TableCell>
                          <TableCell align="right">
                            {formatFinancialNumber(riskModel.volatility_forecast, { percentage: true, decimals: 1 })}
                          </TableCell>
                          <TableCell align="center">
                            <Chip 
                              size="small"
                              label={riskModel.volatility_forecast > 0.25 ? "Volatile" : "Stable"}
                              color={riskModel.volatility_forecast > 0.25 ? "warning" : "success"}
                            />
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Risk Interpretation:</strong>
                    </Typography>
                    <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                      • VaR estimates maximum potential loss over 1 day
                    </Typography>
                    <Typography variant="caption" display="block">
                      • Expected Shortfall shows average loss beyond VaR threshold
                    </Typography>
                    <Typography variant="caption" display="block">
                      • Lower values indicate better risk management
                    </Typography>
                  </Alert>
                  
                  {Math.abs(riskModel.value_at_risk_95) > 0.05 && (
                    <Alert severity="warning">
                      <Typography variant="body2">
                        <Warning sx={{ verticalAlign: 'middle', mr: 1 }} />
                        High risk detected. Consider reducing position sizes or increasing diversification.
                      </Typography>
                    </Alert>
                  )}
                </Grid>
              </Grid>
            ) : (
              <Alert severity="error">Failed to load risk model data</Alert>
            )}
          </Box>
        )}
        
        {activeTab === 2 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Asset Correlation Matrix
            </Typography>
            
            {correlationData.length > 0 ? (
              <Box sx={{ height: 300, mb: 2 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" domain={[0, 'dataMax']} />
                    <YAxis type="number" dataKey="y" domain={[0, 'dataMax']} />
                    <RechartsTooltip
                      formatter={(value, name, props) => [
                        `${(props.payload.correlation * 100).toFixed(1)}%`,
                        `${props.payload.symbol1} vs ${props.payload.symbol2}`
                      ]}
                    />
                    <Scatter
                      data={correlationData}
                      fill={(entry) => {
                        const corr = Math.abs(entry.correlation);
                        if (corr > 0.7) return theme.palette.error.main;
                        if (corr > 0.3) return theme.palette.warning.main;
                        return theme.palette.success.main;
                      }}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </Box>
            ) : (
              <Alert severity="info">No correlation data available</Alert>
            )}
            
            <Alert severity="info">
              <Typography variant="body2">
                <strong>Correlation Risk Assessment:</strong>
              </Typography>
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                • Red dots: High correlation (&gt;70%) - Concentration risk
              </Typography>
              <Typography variant="caption" display="block">
                • Yellow dots: Moderate correlation (30-70%) - Some diversification
              </Typography>
              <Typography variant="caption" display="block">
                • Green dots: Low correlation (&lt;30%) - Good diversification
              </Typography>
            </Alert>
          </Box>
        )}
        
        {activeTab === 3 && (
          <Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Risk Forecast Summary
                </Typography>
                
                {riskModel && (
                  <Box>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" gutterBottom>
                        Expected Volatility (Next 30 days)
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={Math.min(riskModel.volatility_forecast * 400, 100)}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {formatFinancialNumber(riskModel.volatility_forecast, { percentage: true, decimals: 1 })}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" gutterBottom>
                        Correlation Risk Level
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={riskModel.correlation_risk * 100}
                        color={riskModel.correlation_risk > 0.5 ? "error" : "success"}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {formatFinancialNumber(riskModel.correlation_risk, { percentage: true, decimals: 1 })}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" gutterBottom>
                        Concentration Risk Level
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={riskModel.concentration_risk * 100}
                        color={riskModel.concentration_risk > 0.3 ? "error" : "success"}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {formatFinancialNumber(riskModel.concentration_risk, { percentage: true, decimals: 1 })}
                      </Typography>
                    </Box>
                  </Box>
                )}
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Alert 
                  severity={riskAssessment.level === 'High' ? 'error' : riskAssessment.level === 'Medium' ? 'warning' : 'success'}
                >
                  <Typography variant="subtitle2" gutterBottom>
                    Risk Management Recommendations
                  </Typography>
                  
                  {riskAssessment.level === 'High' && (
                    <Box>
                      <Typography variant="body2">• Consider reducing position sizes</Typography>
                      <Typography variant="body2">• Increase diversification across asset classes</Typography>
                      <Typography variant="body2">• Implement stop-loss strategies</Typography>
                      <Typography variant="body2">• Review correlation exposure</Typography>
                    </Box>
                  )}
                  
                  {riskAssessment.level === 'Medium' && (
                    <Box>
                      <Typography variant="body2">• Monitor large positions closely</Typography>
                      <Typography variant="body2">• Consider hedging strategies</Typography>
                      <Typography variant="body2">• Maintain diversification discipline</Typography>
                    </Box>
                  )}
                  
                  {riskAssessment.level === 'Low' && (
                    <Box>
                      <Typography variant="body2">• Risk levels are well controlled</Typography>
                      <Typography variant="body2">• Continue current risk management practices</Typography>
                      <Typography variant="body2">• Monitor for changing market conditions</Typography>
                    </Box>
                  )}
                </Alert>
              </Grid>
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RiskAnalysisCard;
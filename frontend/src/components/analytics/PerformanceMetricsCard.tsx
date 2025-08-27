import React, { useState, useMemo } from 'react';
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
  Tooltip,
  IconButton,
  LinearProgress,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Info,
  ShowChart,
  Assessment,
  Timeline
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import { formatFinancialNumber } from '@/theme';
import FinancialChart from '@/components/charts/FinancialChart';

interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  cumulative_return: number;
  excess_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown: number;
  max_drawdown_duration_days: number;
  current_drawdown: number;
  beta: number;
  alpha: number;
  tracking_error: number;
  information_ratio: number;
  skewness: number;
  kurtosis: number;
  var_95: number;
  cvar_95: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
}

interface PortfolioInfo {
  id: number;
  name: string;
  currency: string;
  initial_value: number;
}

interface PerformanceMetricsCardProps {
  performanceData: PerformanceMetrics;
  portfolioInfo: PortfolioInfo;
}

interface MetricRowProps {
  label: string;
  value: number;
  format: 'percentage' | 'ratio' | 'currency' | 'number' | 'days';
  tooltip?: string;
  benchmark?: number;
  isGoodWhenHigh?: boolean;
}

const MetricRow: React.FC<MetricRowProps> = ({ 
  label, 
  value, 
  format, 
  tooltip, 
  benchmark,
  isGoodWhenHigh = true 
}) => {
  const theme = useTheme();
  
  const formatValue = (val: number, fmt: string) => {
    switch (fmt) {
      case 'percentage':
        return formatFinancialNumber(val, { percentage: true, decimals: 2 });
      case 'ratio':
        return val.toFixed(3);
      case 'currency':
        return formatFinancialNumber(val, { currency: true, decimals: 2 });
      case 'days':
        return `${Math.round(val)} days`;
      default:
        return val.toFixed(2);
    }
  };
  
  const getColor = () => {
    if (benchmark !== undefined) {
      const isOutperforming = isGoodWhenHigh ? value > benchmark : value < benchmark;
      return isOutperforming ? theme.palette.success.main : theme.palette.error.main;
    }
    
    if (format === 'percentage' || format === 'ratio') {
      if (isGoodWhenHigh) {
        return value > 0 ? theme.palette.success.main : theme.palette.error.main;
      } else {
        return value < 0 ? theme.palette.success.main : theme.palette.error.main;
      }
    }
    
    return theme.palette.text.primary;
  };
  
  return (
    <TableRow>
      <TableCell>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2">{label}</Typography>
          {tooltip && (
            <Tooltip title={tooltip}>
              <Info sx={{ fontSize: 16, color: 'text.secondary' }} />
            </Tooltip>
          )}
        </Box>
      </TableCell>
      <TableCell align="right">
        <Typography 
          variant="body2" 
          sx={{ 
            color: getColor(),
            fontWeight: 'medium'
          }}
        >
          {formatValue(value, format)}
        </Typography>
      </TableCell>
      {benchmark !== undefined && (
        <TableCell align="right">
          <Typography variant="body2" color="text.secondary">
            {formatValue(benchmark, format)}
          </Typography>
        </TableCell>
      )}
    </TableRow>
  );
};

const PerformanceMetricsCard: React.FC<PerformanceMetricsCardProps> = ({
  performanceData,
  portfolioInfo
}) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  
  // Generate synthetic chart data for demonstration
  const chartData = useMemo(() => {
    const dataPoints = [];
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);
    
    let cumulativeReturn = 1;
    
    for (let i = 0; i < 252; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Simulate daily return based on performance metrics
      const dailyReturn = 1 + (performanceData.annualized_return / 252) + 
                         (Math.random() - 0.5) * (performanceData.volatility / Math.sqrt(252));
      cumulativeReturn *= dailyReturn;
      
      dataPoints.push({
        date: date.toISOString().split('T')[0],
        timestamp: date.getTime(),
        value: portfolioInfo.initial_value * cumulativeReturn,
        close: portfolioInfo.initial_value * cumulativeReturn
      });
    }
    
    return dataPoints;
  }, [performanceData, portfolioInfo]);
  
  // Risk-adjusted performance score
  const performanceScore = useMemo(() => {
    // Weighted score based on multiple factors
    const sharpeWeight = 0.4;
    const returnWeight = 0.3;
    const drawdownWeight = 0.2;
    const consistencyWeight = 0.1;
    
    const sharpeScore = Math.min(Math.max(performanceData.sharpe_ratio / 2, 0), 1);
    const returnScore = Math.min(Math.max(performanceData.annualized_return * 2, 0), 1);
    const drawdownScore = Math.min(Math.max(1 + performanceData.max_drawdown * 2, 0), 1);
    const consistencyScore = Math.min(Math.max(performanceData.win_rate, 0), 1);
    
    const totalScore = (
      sharpeScore * sharpeWeight +
      returnScore * returnWeight +
      drawdownScore * drawdownWeight +
      consistencyScore * consistencyWeight
    ) * 100;
    
    return Math.round(totalScore);
  }, [performanceData]);
  
  const getPerformanceGrade = (score: number) => {
    if (score >= 90) return { grade: 'A+', color: theme.palette.success.main };
    if (score >= 80) return { grade: 'A', color: theme.palette.success.main };
    if (score >= 70) return { grade: 'B', color: theme.palette.info.main };
    if (score >= 60) return { grade: 'C', color: theme.palette.warning.main };
    return { grade: 'D', color: theme.palette.error.main };
  };
  
  const performanceGrade = getPerformanceGrade(performanceScore);
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6" component="h3">
            <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
            Performance Analytics
          </Typography>
          
          {/* Performance Score */}
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: performanceGrade.color, fontWeight: 'bold' }}>
              {performanceGrade.grade}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Score: {performanceScore}/100
            </Typography>
          </Box>
        </Box>
        
        {/* Performance Chart */}
        <Box sx={{ mb: 3 }}>
          <FinancialChart
            data={chartData}
            title="Portfolio Performance"
            type="area"
            height={300}
            showGrid={true}
            timeframe="1Y"
            symbol={portfolioInfo.name}
          />
        </Box>
        
        {/* Tabs for different metric categories */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs value={activeTab} onChange={handleTabChange} variant="scrollable">
            <Tab label="Returns" />
            <Tab label="Risk" />
            <Tab label="Risk-Adjusted" />
            <Tab label="Distribution" />
          </Tabs>
        </Box>
        
        {/* Metrics Tables */}
        <TableContainer>
          {activeTab === 0 && (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><Typography variant="subtitle2">Return Metrics</Typography></TableCell>
                  <TableCell align="right"><Typography variant="subtitle2">Value</Typography></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <MetricRow
                  label="Total Return"
                  value={performanceData.total_return}
                  format="percentage"
                  tooltip="Total return since inception"
                />
                <MetricRow
                  label="Annualized Return"
                  value={performanceData.annualized_return}
                  format="percentage"
                  tooltip="Compound annual growth rate"
                />
                <MetricRow
                  label="Excess Return"
                  value={performanceData.excess_return}
                  format="percentage"
                  tooltip="Return above benchmark"
                />
                <MetricRow
                  label="Average Win"
                  value={performanceData.avg_win}
                  format="percentage"
                  tooltip="Average winning trade return"
                />
                <MetricRow
                  label="Average Loss"
                  value={performanceData.avg_loss}
                  format="percentage"
                  tooltip="Average losing trade return"
                  isGoodWhenHigh={false}
                />
              </TableBody>
            </Table>
          )}
          
          {activeTab === 1 && (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><Typography variant="subtitle2">Risk Metrics</Typography></TableCell>
                  <TableCell align="right"><Typography variant="subtitle2">Value</Typography></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <MetricRow
                  label="Volatility"
                  value={performanceData.volatility}
                  format="percentage"
                  tooltip="Annual volatility (standard deviation)"
                  isGoodWhenHigh={false}
                />
                <MetricRow
                  label="Maximum Drawdown"
                  value={performanceData.max_drawdown}
                  format="percentage"
                  tooltip="Largest peak-to-trough decline"
                  isGoodWhenHigh={false}
                />
                <MetricRow
                  label="Current Drawdown"
                  value={performanceData.current_drawdown}
                  format="percentage"
                  tooltip="Current decline from recent peak"
                  isGoodWhenHigh={false}
                />
                <MetricRow
                  label="Max Drawdown Duration"
                  value={performanceData.max_drawdown_duration_days}
                  format="days"
                  tooltip="Longest recovery period"
                  isGoodWhenHigh={false}
                />
                <MetricRow
                  label="Value at Risk (95%)"
                  value={performanceData.var_95}
                  format="percentage"
                  tooltip="Maximum expected loss (95% confidence)"
                  isGoodWhenHigh={false}
                />
                <MetricRow
                  label="Conditional VaR (95%)"
                  value={performanceData.cvar_95}
                  format="percentage"
                  tooltip="Expected loss beyond VaR threshold"
                  isGoodWhenHigh={false}
                />
              </TableBody>
            </Table>
          )}
          
          {activeTab === 2 && (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><Typography variant="subtitle2">Risk-Adjusted Metrics</Typography></TableCell>
                  <TableCell align="right"><Typography variant="subtitle2">Value</Typography></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <MetricRow
                  label="Sharpe Ratio"
                  value={performanceData.sharpe_ratio}
                  format="ratio"
                  tooltip="Risk-adjusted return (excess return / volatility)"
                />
                <MetricRow
                  label="Sortino Ratio"
                  value={performanceData.sortino_ratio}
                  format="ratio"
                  tooltip="Return relative to downside deviation"
                />
                <MetricRow
                  label="Calmar Ratio"
                  value={performanceData.calmar_ratio}
                  format="ratio"
                  tooltip="Annual return / maximum drawdown"
                />
                <MetricRow
                  label="Information Ratio"
                  value={performanceData.information_ratio}
                  format="ratio"
                  tooltip="Excess return relative to tracking error"
                />
                <MetricRow
                  label="Beta"
                  value={performanceData.beta}
                  format="ratio"
                  tooltip="Sensitivity to market movements"
                />
                <MetricRow
                  label="Alpha"
                  value={performanceData.alpha}
                  format="percentage"
                  tooltip="Excess return vs. expected return from CAPM"
                />
                <MetricRow
                  label="Tracking Error"
                  value={performanceData.tracking_error}
                  format="percentage"
                  tooltip="Standard deviation of excess returns"
                  isGoodWhenHigh={false}
                />
              </TableBody>
            </Table>
          )}
          
          {activeTab === 3 && (
            <Box>
              <Table size="small" sx={{ mb: 2 }}>
                <TableHead>
                  <TableRow>
                    <TableCell><Typography variant="subtitle2">Distribution Metrics</Typography></TableCell>
                    <TableCell align="right"><Typography variant="subtitle2">Value</Typography></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <MetricRow
                    label="Skewness"
                    value={performanceData.skewness}
                    format="ratio"
                    tooltip="Asymmetry of return distribution"
                  />
                  <MetricRow
                    label="Kurtosis"
                    value={performanceData.kurtosis}
                    format="ratio"
                    tooltip="Tail thickness of return distribution"
                  />
                  <MetricRow
                    label="Win Rate"
                    value={performanceData.win_rate}
                    format="percentage"
                    tooltip="Percentage of profitable periods"
                  />
                  <MetricRow
                    label="Profit Factor"
                    value={performanceData.profit_factor}
                    format="ratio"
                    tooltip="Ratio of gross profits to gross losses"
                  />
                </TableBody>
              </Table>
              
              {/* Distribution Interpretation */}
              <Alert severity="info" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  <strong>Distribution Analysis:</strong>
                  {performanceData.skewness > 0.5 && " Positive skew indicates more frequent small losses and occasional large gains."}
                  {performanceData.skewness < -0.5 && " Negative skew indicates more frequent small gains and occasional large losses."}
                  {Math.abs(performanceData.skewness) <= 0.5 && " Returns are approximately symmetrically distributed."}
                  {performanceData.kurtosis > 3 && " High kurtosis suggests fat tails and higher probability of extreme events."}
                </Typography>
              </Alert>
            </Box>
          )}
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default PerformanceMetricsCard;
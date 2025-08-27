import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  Legend,
} from 'recharts';
import { Box, Card, CardContent, Typography, IconButton, Menu, MenuItem, useTheme } from '@mui/material';
import { MoreVert, TrendingUp, TrendingDown } from '@mui/icons-material';
import { useThemeColors } from '@/hooks/useTheme';
import { formatFinancialNumber } from '@/theme';

interface DataPoint {
  date: string;
  timestamp: number;
  value?: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  [key: string]: any;
}

interface FinancialChartProps {
  data: DataPoint[];
  title?: string;
  type?: 'line' | 'area' | 'bar' | 'candlestick' | 'composed';
  height?: number;
  showGrid?: boolean;
  showBrush?: boolean;
  showVolume?: boolean;
  showLegend?: boolean;
  color?: string;
  gradientColors?: [string, string];
  timeframe?: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '5Y';
  symbol?: string;
  loading?: boolean;
  error?: string;
  onTimeframeChange?: (timeframe: string) => void;
}

const FinancialChart: React.FC<FinancialChartProps> = ({
  data,
  title,
  type = 'line',
  height = 400,
  showGrid = true,
  showBrush = false,
  showVolume = false,
  showLegend = false,
  color,
  gradientColors,
  timeframe = '1M',
  symbol,
  loading = false,
  error,
  onTimeframeChange,
}) => {
  const theme = useTheme();
  const { getFinancialColor, getChartColors } = useThemeColors();
  const chartColors = getChartColors();

  const [menuAnchor, setMenuAnchor] = React.useState<null | HTMLElement>(null);

  // Memoized processed data
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      date: new Date(item.timestamp || item.date).toLocaleDateString(),
    }));
  }, [data]);

  // Calculate performance metrics
  const performance = useMemo(() => {
    if (data.length < 2) return null;

    const first = data[0];
    const last = data[data.length - 1];
    
    const startValue = first.close || first.value || 0;
    const endValue = last.close || last.value || 0;
    const change = endValue - startValue;
    const changePercent = (change / startValue) * 100;

    return {
      startValue,
      endValue,
      change,
      changePercent,
    };
  }, [data]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      return (
        <Card sx={{ minWidth: 200 }}>
          <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
            <Typography variant="subtitle2" gutterBottom>
              {label}
            </Typography>
            {payload.map((entry: any, index: number) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    bgcolor: entry.color,
                    borderRadius: '50%',
                  }}
                />
                <Typography variant="body2">
                  {entry.name}: {formatFinancialNumber(entry.value, { currency: true, decimals: 2 })}
                </Typography>
              </Box>
            ))}
            {data.volume && (
              <Typography variant="caption" color="text.secondary">
                Volume: {formatFinancialNumber(data.volume, { compact: true })}
              </Typography>
            )}
          </CardContent>
        </Card>
      );
    }
    return null;
  };

  // Format X-axis labels
  const formatXAxisLabel = (tickItem: string) => {
    const date = new Date(tickItem);
    switch (timeframe) {
      case '1D':
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      case '1W':
      case '1M':
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
      case '3M':
      case '6M':
        return date.toLocaleDateString([], { month: 'short', year: '2-digit' });
      case '1Y':
      case '5Y':
        return date.toLocaleDateString([], { year: '2-digit', month: 'short' });
      default:
        return date.toLocaleDateString();
    }
  };

  const renderChart = () => {
    const commonProps = {
      data: processedData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    const primaryColor = color || chartColors.line1;
    const gradientId = `colorGradient_${type}`;

    switch (type) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={primaryColor} stopOpacity={0.3} />
                <stop offset="95%" stopColor={primaryColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            )}
            <XAxis
              dataKey="date"
              stroke={chartColors.axis}
              tickFormatter={formatXAxisLabel}
              fontSize={12}
            />
            <YAxis
              stroke={chartColors.axis}
              tickFormatter={(value) => formatFinancialNumber(value, { compact: true })}
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            <Area
              type="monotone"
              dataKey="close"
              stroke={primaryColor}
              fillOpacity={1}
              fill={`url(#${gradientId})`}
              strokeWidth={2}
            />
            {showVolume && (
              <Bar dataKey="volume" fill={chartColors.line4} opacity={0.3} />
            )}
            {showBrush && <Brush dataKey="date" height={30} stroke={primaryColor} />}
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            )}
            <XAxis
              dataKey="date"
              stroke={chartColors.axis}
              tickFormatter={formatXAxisLabel}
              fontSize={12}
            />
            <YAxis
              stroke={chartColors.axis}
              tickFormatter={(value) => formatFinancialNumber(value, { compact: true })}
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            <Bar dataKey="volume" fill={primaryColor} />
          </BarChart>
        );

      case 'composed':
        return (
          <ComposedChart {...commonProps}>
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            )}
            <XAxis
              dataKey="date"
              stroke={chartColors.axis}
              tickFormatter={formatXAxisLabel}
              fontSize={12}
            />
            <YAxis
              stroke={chartColors.axis}
              tickFormatter={(value) => formatFinancialNumber(value, { compact: true })}
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            <Line
              type="monotone"
              dataKey="close"
              stroke={primaryColor}
              strokeWidth={2}
              dot={false}
            />
            {showVolume && (
              <Bar dataKey="volume" fill={chartColors.line4} opacity={0.3} />
            )}
          </ComposedChart>
        );

      default: // line
        return (
          <LineChart {...commonProps}>
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            )}
            <XAxis
              dataKey="date"
              stroke={chartColors.axis}
              tickFormatter={formatXAxisLabel}
              fontSize={12}
            />
            <YAxis
              stroke={chartColors.axis}
              tickFormatter={(value) => formatFinancialNumber(value, { compact: true })}
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && <Legend />}
            <Line
              type="monotone"
              dataKey="close"
              stroke={primaryColor}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, stroke: primaryColor, strokeWidth: 2 }}
            />
            {performance && (
              <ReferenceLine
                y={performance.startValue}
                stroke={chartColors.axis}
                strokeDasharray="2 2"
                label="Start"
              />
            )}
            {showBrush && <Brush dataKey="date" height={30} stroke={primaryColor} />}
          </LineChart>
        );
    }
  };

  if (error) {
    return (
      <Card sx={{ height }}>
        <CardContent>
          <Typography color="error">Error loading chart: {error}</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            {title && (
              <Typography variant="h6" component="h3">
                {title}
              </Typography>
            )}
            {symbol && performance && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                <Typography variant="h5" component="span">
                  {formatFinancialNumber(performance.endValue, { currency: true })}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  {performance.change >= 0 ? (
                    <TrendingUp sx={{ color: getFinancialColor(performance.change), fontSize: 20 }} />
                  ) : (
                    <TrendingDown sx={{ color: getFinancialColor(performance.change), fontSize: 20 }} />
                  )}
                  <Typography
                    variant="body2"
                    sx={{ color: getFinancialColor(performance.change), fontWeight: 500 }}
                  >
                    {performance.change >= 0 ? '+' : ''}
                    {formatFinancialNumber(performance.changePercent, { percentage: true, decimals: 2 })}
                  </Typography>
                </Box>
              </Box>
            )}
          </Box>
          <IconButton onClick={(e) => setMenuAnchor(e.currentTarget)}>
            <MoreVert />
          </IconButton>
        </Box>

        {/* Chart */}
        <Box sx={{ width: '100%', height: height - 100 }}>
          <ResponsiveContainer>
            {renderChart()}
          </ResponsiveContainer>
        </Box>

        {/* Options Menu */}
        <Menu
          anchorEl={menuAnchor}
          open={Boolean(menuAnchor)}
          onClose={() => setMenuAnchor(null)}
        >
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Export Chart
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Full Screen
          </MenuItem>
          <MenuItem onClick={() => setMenuAnchor(null)}>
            Settings
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default FinancialChart;
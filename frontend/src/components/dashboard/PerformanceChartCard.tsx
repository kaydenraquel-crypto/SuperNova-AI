import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  ToggleButton,
  ToggleButtonGroup,
  Skeleton,
  Avatar,
  Chip,
} from '@mui/material';
import { ShowChart, Timeline } from '@mui/icons-material';
import FinancialChart from '../charts/FinancialChart';

interface PerformanceData {
  dates: string[];
  values: number[];
  benchmark?: number[];
}

interface PerformanceChartCardProps {
  data?: PerformanceData;
  isLoading?: boolean;
}

const PerformanceChartCard: React.FC<PerformanceChartCardProps> = ({
  data,
  isLoading = false,
}) => {
  const [timeframe, setTimeframe] = useState('1M');
  const [chartType, setChartType] = useState<'line' | 'area'>('area');

  if (isLoading || !data) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
              <ShowChart />
            </Avatar>
            <Box sx={{ flex: 1 }}>
              <Skeleton variant="text" width="40%" height={32} />
              <Skeleton variant="text" width="60%" height={24} />
            </Box>
          </Box>
          <Skeleton variant="rectangular" width="100%" height={350} />
        </CardContent>
      </Card>
    );
  }

  // Transform data for chart
  const chartData = data.dates.map((date, index) => ({
    date,
    timestamp: new Date(date).getTime(),
    close: data.values[index],
    benchmark: data.benchmark?.[index],
  }));

  const handleTimeframeChange = (event: React.MouseEvent<HTMLElement>, newTimeframe: string) => {
    if (newTimeframe !== null) {
      setTimeframe(newTimeframe);
    }
  };

  const handleChartTypeChange = (event: React.MouseEvent<HTMLElement>, newType: 'line' | 'area') => {
    if (newType !== null) {
      setChartType(newType);
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
              <ShowChart />
            </Avatar>
            <Box>
              <Typography variant="h6" component="div">
                Portfolio Performance
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Track your portfolio growth over time
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              icon={<Timeline />}
              label={chartType === 'area' ? 'Area' : 'Line'}
              onClick={() => setChartType(chartType === 'area' ? 'line' : 'area')}
              variant="outlined"
              size="small"
            />
          </Box>
        </Box>

        {/* Timeframe Controls */}
        <Box sx={{ mb: 3 }}>
          <ToggleButtonGroup
            value={timeframe}
            exclusive
            onChange={handleTimeframeChange}
            size="small"
            aria-label="timeframe"
          >
            <ToggleButton value="1D" aria-label="1 day">
              1D
            </ToggleButton>
            <ToggleButton value="1W" aria-label="1 week">
              1W
            </ToggleButton>
            <ToggleButton value="1M" aria-label="1 month">
              1M
            </ToggleButton>
            <ToggleButton value="3M" aria-label="3 months">
              3M
            </ToggleButton>
            <ToggleButton value="6M" aria-label="6 months">
              6M
            </ToggleButton>
            <ToggleButton value="1Y" aria-label="1 year">
              1Y
            </ToggleButton>
            <ToggleButton value="5Y" aria-label="5 years">
              5Y
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* Chart */}
        <FinancialChart
          data={chartData}
          type={chartType}
          height={350}
          timeframe={timeframe as any}
          showGrid
          showLegend={!!data.benchmark}
        />
      </CardContent>
    </Card>
  );
};

export default PerformanceChartCard;
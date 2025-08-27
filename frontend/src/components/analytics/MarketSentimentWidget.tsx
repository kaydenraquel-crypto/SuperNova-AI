import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  SentimentSatisfied,
  SentimentNeutral,
  SentimentDissatisfied,
  Refresh,
  Info
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts';
import { formatFinancialNumber } from '@/theme';

interface SentimentData {
  symbol?: string;
  sector?: string;
  current_sentiment: number;
  confidence: number;
  volume_weighted: number;
  social_sentiment: number;
  news_sentiment: number;
  analyst_sentiment: number;
  total_mentions: number;
  last_updated: string;
  trend: string;
}

interface MarketSentimentWidget {
  refreshInterval?: number;
}

const MarketSentimentWidget: React.FC<MarketSentimentWidget> = ({ 
  refreshInterval = 300000 // 5 minutes
}) => {
  const theme = useTheme();
  const [sentimentData, setSentimentData] = useState<Record<string, SentimentData>>({});
  const [overallMarket, setOverallMarket] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Fetch market sentiment data
  const fetchSentimentData = async () => {
    try {
      setLoading(true);
      setError(null);

      // For demonstration, create synthetic sentiment data
      // In production, this would fetch from /api/analytics/market/sentiment
      const syntheticData = {
        'AAPL': {
          symbol: 'AAPL',
          current_sentiment: 0.65,
          confidence: 0.82,
          volume_weighted: 0.71,
          social_sentiment: 0.58,
          news_sentiment: 0.74,
          analyst_sentiment: 0.69,
          total_mentions: 15420,
          last_updated: new Date().toISOString(),
          trend: 'positive'
        },
        'MSFT': {
          symbol: 'MSFT',
          current_sentiment: 0.42,
          confidence: 0.76,
          volume_weighted: 0.48,
          social_sentiment: 0.38,
          news_sentiment: 0.45,
          analyst_sentiment: 0.52,
          total_mentions: 8930,
          last_updated: new Date().toISOString(),
          trend: 'neutral'
        },
        'TSLA': {
          symbol: 'TSLA',
          current_sentiment: -0.23,
          confidence: 0.91,
          volume_weighted: -0.31,
          social_sentiment: -0.45,
          news_sentiment: -0.12,
          analyst_sentiment: 0.05,
          total_mentions: 23450,
          last_updated: new Date().toISOString(),
          trend: 'negative'
        }
      };

      const syntheticOverall = {
        average_sentiment: 0.28,
        sentiment_distribution: {
          bullish: 12,
          neutral: 18,
          bearish: 8
        },
        market_mood: 'cautiously_optimistic'
      };

      setTimeout(() => {
        setSentimentData(syntheticData);
        setOverallMarket(syntheticOverall);
        setLastUpdate(new Date());
        setLoading(false);
      }, 1000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sentiment data');
      setLoading(false);
    }
  };

  // Initial load and refresh interval
  useEffect(() => {
    fetchSentimentData();
    
    const interval = setInterval(fetchSentimentData, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  // Helper functions
  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.2) return <SentimentSatisfied sx={{ color: theme.palette.success.main }} />;
    if (sentiment < -0.2) return <SentimentDissatisfied sx={{ color: theme.palette.error.main }} />;
    return <SentimentNeutral sx={{ color: theme.palette.warning.main }} />;
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.2) return theme.palette.success.main;
    if (sentiment < -0.2) return theme.palette.error.main;
    return theme.palette.warning.main;
  };

  const getSentimentLabel = (sentiment: number) => {
    if (sentiment > 0.5) return 'Very Bullish';
    if (sentiment > 0.2) return 'Bullish';
    if (sentiment > -0.2) return 'Neutral';
    if (sentiment > -0.5) return 'Bearish';
    return 'Very Bearish';
  };

  // Generate historical sentiment chart data (synthetic)
  const generateHistoricalData = () => {
    const data = [];
    const now = new Date();
    
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      data.push({
        date: date.toISOString().split('T')[0],
        sentiment: Math.random() * 1.4 - 0.7, // Random sentiment between -0.7 and 0.7
        confidence: 0.6 + Math.random() * 0.4,
        mentions: Math.floor(Math.random() * 5000) + 10000
      });
    }
    
    return data;
  };

  const historicalData = generateHistoricalData();

  if (loading && Object.keys(sentimentData).length === 0) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
            <CircularProgress />
            <Typography variant="body2" sx={{ ml: 2 }}>
              Loading market sentiment...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert 
            severity="error"
            action={
              <IconButton size="small" onClick={fetchSentimentData}>
                <Refresh />
              </IconButton>
            }
          >
            {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Overall Market Sentiment */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" component="h3">
                Market Sentiment Analysis
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {lastUpdate && (
                  <Typography variant="caption" color="text.secondary">
                    Updated: {lastUpdate.toLocaleTimeString()}
                  </Typography>
                )}
                <IconButton size="small" onClick={fetchSentimentData} disabled={loading}>
                  <Refresh />
                </IconButton>
              </Box>
            </Box>

            {overallMarket && (
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="h4" sx={{ color: getSentimentColor(overallMarket.average_sentiment) }}>
                      {getSentimentIcon(overallMarket.average_sentiment)}
                    </Typography>
                    <Typography variant="h6" sx={{ color: getSentimentColor(overallMarket.average_sentiment) }}>
                      {(overallMarket.average_sentiment * 100).toFixed(1)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Overall Sentiment
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="h4" color="success.main">
                      {overallMarket.sentiment_distribution.bullish}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Bullish Assets
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="h4" color="warning.main">
                      {overallMarket.sentiment_distribution.neutral}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Neutral Assets
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
                    <Typography variant="h4" color="error.main">
                      {overallMarket.sentiment_distribution.bearish}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Bearish Assets
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            )}
          </CardContent>
        </Card>
      </Grid>

      {/* Historical Sentiment Chart */}
      <Grid item xs={12} lg={8}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Sentiment Trend (30 Days)
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(date) => new Date(date).toLocaleDateString()}
                />
                <YAxis 
                  domain={[-1, 1]}
                  tickFormatter={(value) => (value * 100).toFixed(0)}
                />
                <RechartsTooltip
                  formatter={(value: number, name) => [
                    `${(value * 100).toFixed(1)}%`,
                    name === 'sentiment' ? 'Sentiment Score' : 'Confidence'
                  ]}
                  labelFormatter={(date) => new Date(date).toLocaleDateString()}
                />
                <Area
                  type="monotone"
                  dataKey="sentiment"
                  stroke="#8884d8"
                  fill="url(#sentimentGradient)"
                  fillOpacity={0.3}
                />
                <defs>
                  <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00C49F" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#FF8042" stopOpacity={0.8} />
                  </linearGradient>
                </defs>
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Sentiment Distribution */}
      <Grid item xs={12} lg={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Sentiment Distribution
            </Typography>
            
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={[
                  { name: 'Bullish', value: overallMarket?.sentiment_distribution.bullish || 0, fill: theme.palette.success.main },
                  { name: 'Neutral', value: overallMarket?.sentiment_distribution.neutral || 0, fill: theme.palette.warning.main },
                  { name: 'Bearish', value: overallMarket?.sentiment_distribution.bearish || 0, fill: theme.palette.error.main }
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <RechartsTooltip />
                <Bar dataKey="value" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* Individual Asset Sentiment */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Individual Asset Sentiment
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Sentiment</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Social</TableCell>
                    <TableCell>News</TableCell>
                    <TableCell>Analysts</TableCell>
                    <TableCell align="right">Mentions</TableCell>
                    <TableCell>Trend</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.values(sentimentData).map((data) => (
                    <TableRow key={data.symbol}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {data.symbol}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getSentimentIcon(data.current_sentiment)}
                          <Box>
                            <Typography variant="body2" sx={{ color: getSentimentColor(data.current_sentiment) }}>
                              {(data.current_sentiment * 100).toFixed(1)}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {getSentimentLabel(data.current_sentiment)}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: 100 }}>
                          <LinearProgress
                            variant="determinate"
                            value={data.confidence * 100}
                            sx={{ flex: 1, height: 6 }}
                          />
                          <Typography variant="caption">
                            {(data.confidence * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          size="small"
                          label={(data.social_sentiment * 100).toFixed(1)}
                          sx={{
                            backgroundColor: getSentimentColor(data.social_sentiment),
                            color: 'white',
                            minWidth: 50
                          }}
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          size="small"
                          label={(data.news_sentiment * 100).toFixed(1)}
                          sx={{
                            backgroundColor: getSentimentColor(data.news_sentiment),
                            color: 'white',
                            minWidth: 50
                          }}
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          size="small"
                          label={(data.analyst_sentiment * 100).toFixed(1)}
                          sx={{
                            backgroundColor: getSentimentColor(data.analyst_sentiment),
                            color: 'white',
                            minWidth: 50
                          }}
                        />
                      </TableCell>
                      
                      <TableCell align="right">
                        <Typography variant="body2">
                          {data.total_mentions.toLocaleString()}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {data.trend === 'positive' && <TrendingUp sx={{ color: 'success.main', mr: 0.5 }} />}
                          {data.trend === 'negative' && <TrendingDown sx={{ color: 'error.main', mr: 0.5 }} />}
                          <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                            {data.trend}
                          </Typography>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                <strong>Sentiment Scoring:</strong> Ranges from -100 (very bearish) to +100 (very bullish). 
                Confidence indicates the reliability of the sentiment score based on data quality and volume.
              </Typography>
            </Alert>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default MarketSentimentWidget;
import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardHeader,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Alert,
  AlertTitle,
  Tab,
  Tabs,
  TabPanel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Security as SecurityIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Key as KeyIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Visibility as VisibilityIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  RotateRight as RotateIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface APIKey {
  key_id: string;
  name: string;
  tier: string;
  scopes: string[];
  prefix: string;
  created_at: string;
  expires_at?: string;
  last_used?: string;
  is_active: boolean;
  rate_limits: Record<string, number>;
  usage_stats?: {
    total_requests: number;
    successful_requests: number;
    failed_requests: number;
    avg_response_time: number;
  };
}

interface RealTimeMetrics {
  timestamp: string;
  current_rps: number;
  peak_rps: number;
  total_requests: number;
  error_count: number;
  avg_response_time: number;
  active_api_keys: number;
  unique_users_today: number;
  unique_ips_today: number;
  cache_hit_rate: number;
  cache_size: number;
  system_health_score: number;
  threat_level: boolean;
}

interface DashboardMetrics {
  overview: Record<string, number>;
  traffic_chart: Array<{
    timestamp: string;
    requests: number;
    errors: number;
  }>;
  top_endpoints: Array<{
    endpoint: string;
    count: number;
    avg_response_time: number;
    error_count: number;
  }>;
  recent_errors: Array<{
    error: string;
    count: number;
  }>;
  geographical_data: Record<string, number>;
  performance_trends: Array<{
    timestamp: string;
    response_time: number;
    throughput: number;
  }>;
  security_alerts: Array<{
    type: string;
    severity: string;
    message: string;
    timestamp: string;
  }>;
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
      id={`api-management-tabpanel-${index}`}
      aria-labelledby={`api-management-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const APIManagementDashboard: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [realTimeMetrics, setRealTimeMetrics] = useState<RealTimeMetrics | null>(null);
  const [dashboardMetrics, setDashboardMetrics] = useState<DashboardMetrics | null>(null);
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [createKeyDialog, setCreateKeyDialog] = useState(false);
  const [selectedKey, setSelectedKey] = useState<APIKey | null>(null);
  const [keyDetailsDialog, setKeyDetailsDialog] = useState(false);

  // Auto-refresh interval
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch real-time metrics
        const metricsResponse = await fetch('/api/management/metrics/realtime');
        if (metricsResponse.ok) {
          const metrics = await metricsResponse.json();
          setRealTimeMetrics(metrics);
        }

        // Fetch dashboard metrics
        const dashboardResponse = await fetch('/api/management/dashboard');
        if (dashboardResponse.ok) {
          const dashboard = await dashboardResponse.json();
          setDashboardMetrics(dashboard);
        }

        // Fetch API keys
        const keysResponse = await fetch('/api/management/keys');
        if (keysResponse.ok) {
          const keysData = await keysResponse.json();
          setApiKeys(keysData.keys);
        }

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleRefresh = async () => {
    setLoading(true);
    // Trigger data refresh
    if (typeof window !== 'undefined') {
      window.location.reload(); // Simple refresh for now
    }
  };

  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    } else if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  const getHealthColor = (score: number): string => {
    if (score >= 90) return '#4caf50'; // Green
    if (score >= 70) return '#ff9800'; // Orange
    return '#f44336'; // Red
  };

  const getTierColor = (tier: string): string => {
    switch (tier.toLowerCase()) {
      case 'enterprise': return 'primary';
      case 'pro': return 'secondary';
      case 'admin': return 'error';
      default: return 'default';
    }
  };

  if (loading && !realTimeMetrics) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <LinearProgress sx={{ width: '50%' }} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        <AlertTitle>Error</AlertTitle>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          API Management Dashboard
        </Typography>
        <Box>
          <IconButton onClick={handleRefresh} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* System Health Alert */}
      {realTimeMetrics && realTimeMetrics.threat_level && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>Security Alert</AlertTitle>
          System is under attack or experiencing suspicious activity. Enhanced security measures are active.
        </Alert>
      )}

      {/* Tabs Navigation */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={handleTabChange}>
          <Tab label="Overview" />
          <Tab label="API Keys" />
          <Tab label="Analytics" />
          <Tab label="Security" />
          <Tab label="Settings" />
        </Tabs>
      </Box>

      {/* Overview Tab */}
      <TabPanel value={currentTab} index={0}>
        <Grid container spacing={3}>
          {/* Key Metrics Cards */}
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <SpeedIcon color="primary" />
                  <Box ml={2}>
                    <Typography color="textSecondary" gutterBottom>
                      Current RPS
                    </Typography>
                    <Typography variant="h5">
                      {realTimeMetrics ? realTimeMetrics.current_rps.toFixed(1) : '0'}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <TrendingUpIcon color="primary" />
                  <Box ml={2}>
                    <Typography color="textSecondary" gutterBottom>
                      Total Requests
                    </Typography>
                    <Typography variant="h5">
                      {realTimeMetrics ? formatNumber(realTimeMetrics.total_requests) : '0'}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <SecurityIcon color="primary" />
                  <Box ml={2}>
                    <Typography color="textSecondary" gutterBottom>
                      Active Keys
                    </Typography>
                    <Typography variant="h5">
                      {realTimeMetrics ? realTimeMetrics.active_api_keys : '0'}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <StorageIcon color="primary" />
                  <Box ml={2}>
                    <Typography color="textSecondary" gutterBottom>
                      Cache Hit Rate
                    </Typography>
                    <Typography variant="h5">
                      {realTimeMetrics ? `${realTimeMetrics.cache_hit_rate.toFixed(1)}%` : '0%'}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* System Health */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="System Health" />
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Typography variant="h4" sx={{ mr: 2 }}>
                    {realTimeMetrics ? realTimeMetrics.system_health_score.toFixed(0) : '0'}%
                  </Typography>
                  <Box flexGrow={1}>
                    <LinearProgress
                      variant="determinate"
                      value={realTimeMetrics ? realTimeMetrics.system_health_score : 0}
                      sx={{
                        height: 10,
                        borderRadius: 5,
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: realTimeMetrics 
                            ? getHealthColor(realTimeMetrics.system_health_score)
                            : '#e0e0e0'
                        }
                      }}
                    />
                  </Box>
                </Box>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Avg Response Time
                    </Typography>
                    <Typography variant="h6">
                      {realTimeMetrics ? `${realTimeMetrics.avg_response_time.toFixed(0)}ms` : '0ms'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Error Count
                    </Typography>
                    <Typography variant="h6">
                      {realTimeMetrics ? formatNumber(realTimeMetrics.error_count) : '0'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Traffic Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Traffic Overview" />
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={dashboardMetrics?.traffic_chart || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <ChartTooltip />
                    <Area type="monotone" dataKey="requests" stackId="1" stroke="#8884d8" fill="#8884d8" />
                    <Area type="monotone" dataKey="errors" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Top Endpoints */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Top Endpoints" />
              <CardContent>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Endpoint</TableCell>
                        <TableCell align="right">Requests</TableCell>
                        <TableCell align="right">Avg Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(dashboardMetrics?.top_endpoints || []).slice(0, 5).map((endpoint, index) => (
                        <TableRow key={index}>
                          <TableCell>{endpoint.endpoint}</TableCell>
                          <TableCell align="right">{formatNumber(endpoint.count)}</TableCell>
                          <TableCell align="right">{endpoint.avg_response_time.toFixed(0)}ms</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Errors */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Recent Errors" />
              <CardContent>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Error Type</TableCell>
                        <TableCell align="right">Count</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(dashboardMetrics?.recent_errors || []).slice(0, 5).map((error, index) => (
                        <TableRow key={index}>
                          <TableCell>{error.error}</TableCell>
                          <TableCell align="right">{error.count}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* API Keys Tab */}
      <TabPanel value={currentTab} index={1}>
        <Box mb={3}>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateKeyDialog(true)}
          >
            Create API Key
          </Button>
        </Box>

        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Tier</TableCell>
                <TableCell>Prefix</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Last Used</TableCell>
                <TableCell>Requests</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {apiKeys.map((key) => (
                <TableRow key={key.key_id}>
                  <TableCell>{key.name}</TableCell>
                  <TableCell>
                    <Chip 
                      label={key.tier} 
                      color={getTierColor(key.tier) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" fontFamily="monospace">
                      {key.prefix}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      icon={key.is_active ? <CheckCircleIcon /> : <ErrorIcon />}
                      label={key.is_active ? 'Active' : 'Inactive'}
                      color={key.is_active ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {key.last_used ? new Date(key.last_used).toLocaleDateString() : 'Never'}
                  </TableCell>
                  <TableCell>
                    {key.usage_stats ? formatNumber(key.usage_stats.total_requests) : '0'}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={() => {
                        setSelectedKey(key);
                        setKeyDetailsDialog(true);
                      }}
                    >
                      <VisibilityIcon />
                    </IconButton>
                    <IconButton size="small">
                      <EditIcon />
                    </IconButton>
                    <IconButton size="small">
                      <RotateIcon />
                    </IconButton>
                    <IconButton size="small" color="error">
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Analytics Tab */}
      <TabPanel value={currentTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Performance Trends" />
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={dashboardMetrics?.performance_trends || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <ChartTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="response_time" stroke="#8884d8" name="Response Time (ms)" />
                    <Line type="monotone" dataKey="throughput" stroke="#82ca9d" name="Throughput (rps)" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Security Tab */}
      <TabPanel value={currentTab} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Alert severity={realTimeMetrics?.threat_level ? "error" : "success"}>
              <AlertTitle>Threat Level</AlertTitle>
              {realTimeMetrics?.threat_level 
                ? "System is currently under attack or experiencing suspicious activity"
                : "No current security threats detected"
              }
            </Alert>
          </Grid>

          {dashboardMetrics?.security_alerts && dashboardMetrics.security_alerts.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardHeader title="Security Alerts" />
                <CardContent>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Type</TableCell>
                          <TableCell>Severity</TableCell>
                          <TableCell>Message</TableCell>
                          <TableCell>Time</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {dashboardMetrics.security_alerts.map((alert, index) => (
                          <TableRow key={index}>
                            <TableCell>{alert.type}</TableCell>
                            <TableCell>
                              <Chip
                                label={alert.severity}
                                color={
                                  alert.severity === 'critical' ? 'error' :
                                  alert.severity === 'high' ? 'warning' : 'default'
                                }
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{alert.message}</TableCell>
                            <TableCell>{new Date(alert.timestamp).toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </TabPanel>

      {/* Settings Tab */}
      <TabPanel value={currentTab} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Rate Limiting" />
              <CardContent>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Configure global rate limiting settings
                </Typography>
                {/* Rate limiting configuration form would go here */}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Security Settings" />
              <CardContent>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Configure security policies and monitoring
                </Typography>
                {/* Security settings form would go here */}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Create API Key Dialog */}
      <Dialog open={createKeyDialog} onClose={() => setCreateKeyDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New API Key</DialogTitle>
        <DialogContent>
          <Box component="form" sx={{ mt: 2 }}>
            <TextField
              autoFocus
              margin="dense"
              id="name"
              label="API Key Name"
              type="text"
              fullWidth
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="tier-label">Tier</InputLabel>
              <Select
                labelId="tier-label"
                id="tier"
                label="Tier"
              >
                <MenuItem value="free">Free</MenuItem>
                <MenuItem value="pro">Pro</MenuItem>
                <MenuItem value="enterprise">Enterprise</MenuItem>
              </Select>
            </FormControl>
            <TextField
              margin="dense"
              id="scopes"
              label="Scopes (comma-separated)"
              type="text"
              fullWidth
              variant="outlined"
              placeholder="market:read, portfolio:write"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateKeyDialog(false)}>Cancel</Button>
          <Button variant="contained">Create</Button>
        </DialogActions>
      </Dialog>

      {/* Key Details Dialog */}
      <Dialog 
        open={keyDetailsDialog} 
        onClose={() => setKeyDetailsDialog(false)} 
        maxWidth="md" 
        fullWidth
      >
        <DialogTitle>API Key Details</DialogTitle>
        <DialogContent>
          {selectedKey && (
            <Box>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" color="textSecondary">Name</Typography>
                  <Typography variant="body1">{selectedKey.name}</Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" color="textSecondary">Tier</Typography>
                  <Chip label={selectedKey.tier} color={getTierColor(selectedKey.tier) as any} />
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" color="textSecondary">Created</Typography>
                  <Typography variant="body1">
                    {new Date(selectedKey.created_at).toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" color="textSecondary">Last Used</Typography>
                  <Typography variant="body1">
                    {selectedKey.last_used 
                      ? new Date(selectedKey.last_used).toLocaleString() 
                      : 'Never'
                    }
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="textSecondary">Scopes</Typography>
                  <Box sx={{ mt: 1 }}>
                    {selectedKey.scopes.map((scope, index) => (
                      <Chip key={index} label={scope} size="small" sx={{ mr: 1, mb: 1 }} />
                    ))}
                  </Box>
                </Grid>
                {selectedKey.usage_stats && (
                  <Grid item xs={12}>
                    <Typography variant="subtitle2" color="textSecondary">Usage Statistics</Typography>
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={6}>
                        <Typography variant="body2">Total Requests</Typography>
                        <Typography variant="h6">
                          {formatNumber(selectedKey.usage_stats.total_requests)}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">Success Rate</Typography>
                        <Typography variant="h6">
                          {((selectedKey.usage_stats.successful_requests / 
                             selectedKey.usage_stats.total_requests) * 100).toFixed(1)}%
                        </Typography>
                      </Grid>
                    </Grid>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setKeyDetailsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default APIManagementDashboard;
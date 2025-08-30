import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Box, 
  Card, 
  CardContent, 
  Grid,
  Button
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

const Dashboard = () => (
  <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
    <Typography variant="h3" component="h1" gutterBottom>
      SuperNova AI Dashboard
    </Typography>
    <Grid container spacing={3}>
      <Grid item xs={12} md={6} lg={4}>
        <Card>
          <CardContent>
            <Typography variant="h5" component="h2">
              Portfolio Overview
            </Typography>
            <Typography variant="body2">
              Your portfolio performance and analytics will appear here.
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6} lg={4}>
        <Card>
          <CardContent>
            <Typography variant="h5" component="h2">
              Market Data
            </Typography>
            <Typography variant="body2">
              Real-time market information and trends.
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6} lg={4}>
        <Card>
          <CardContent>
            <Typography variant="h5" component="h2">
              AI Insights
            </Typography>
            <Typography variant="body2">
              AI-powered financial insights and recommendations.
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  </Container>
);

const Chat = () => (
  <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
    <Typography variant="h3" component="h1" gutterBottom>
      AI Assistant
    </Typography>
    <Card>
      <CardContent>
        <Typography variant="body1">
          Chat with your AI financial advisor here.
        </Typography>
      </CardContent>
    </Card>
  </Container>
);

const Portfolio = () => (
  <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
    <Typography variant="h3" component="h1" gutterBottom>
      Portfolio Management
    </Typography>
    <Card>
      <CardContent>
        <Typography variant="body1">
          Manage your investment portfolio here.
        </Typography>
      </CardContent>
    </Card>
  </Container>
);

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              SuperNova AI
            </Typography>
            <Button color="inherit" component={Link} to="/">
              Dashboard
            </Button>
            <Button color="inherit" component={Link} to="/chat">
              Chat
            </Button>
            <Button color="inherit" component={Link} to="/portfolio">
              Portfolio
            </Button>
          </Toolbar>
        </AppBar>
        
        <Box sx={{ flexGrow: 1 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/portfolio" element={<Portfolio />} />
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
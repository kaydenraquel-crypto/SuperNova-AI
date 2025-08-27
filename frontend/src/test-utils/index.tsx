/**
 * Test utilities and custom render functions for testing React components
 */
import React, { ReactElement } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { BrowserRouter } from 'react-router-dom';
import { createTheme } from '@mui/material/styles';

import { NotificationProvider } from '../components/common/NotificationProvider';

// Create theme for testing
const testTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

// Create a test query client
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      cacheTime: 0,
    },
    mutations: {
      retry: false,
    },
  },
});

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialEntries?: string[];
  queryClient?: QueryClient;
  theme?: any;
}

interface AllProvidersProps {
  children: React.ReactNode;
  initialEntries?: string[];
  queryClient?: QueryClient;
  theme?: any;
}

// All providers wrapper
const AllProviders: React.FC<AllProvidersProps> = ({ 
  children, 
  initialEntries = ['/'],
  queryClient = createTestQueryClient(),
  theme = testTheme
}) => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <NotificationProvider>
          <BrowserRouter>
            {children}
          </BrowserRouter>
        </NotificationProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

// Custom render function
const customRender = (
  ui: ReactElement,
  options: CustomRenderOptions = {}
): RenderResult => {
  const {
    initialEntries,
    queryClient,
    theme,
    ...renderOptions
  } = options;

  const Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <AllProviders 
      initialEntries={initialEntries}
      queryClient={queryClient}
      theme={theme}
    >
      {children}
    </AllProviders>
  );

  return render(ui, { wrapper: Wrapper, ...renderOptions });
};

// Mock data generators
export const mockUser = {
  id: 1,
  name: 'Test User',
  email: 'test@example.com',
  profileId: 1,
  riskScore: 60,
};

export const mockOHLCVData = Array.from({ length: 100 }, (_, i) => ({
  timestamp: new Date(Date.now() - (100 - i) * 60000).toISOString(),
  open: 100 + Math.random() * 10,
  high: 105 + Math.random() * 10,
  low: 95 + Math.random() * 10,
  close: 100 + Math.random() * 10,
  volume: Math.floor(Math.random() * 1000000),
}));

export const mockPortfolioData = {
  totalValue: 125000,
  totalReturn: 25000,
  totalReturnPercent: 25.0,
  dayChange: 1250,
  dayChangePercent: 1.01,
  positions: [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      shares: 100,
      currentPrice: 150.00,
      marketValue: 15000,
      costBasis: 12000,
      unrealizedPL: 3000,
      unrealizedPLPercent: 25.0,
    },
    {
      symbol: 'GOOGL', 
      name: 'Alphabet Inc.',
      shares: 50,
      currentPrice: 2500.00,
      marketValue: 125000,
      costBasis: 100000,
      unrealizedPL: 25000,
      unrealizedPLPercent: 25.0,
    },
  ],
};

export const mockChartData = {
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
  datasets: [
    {
      label: 'Portfolio Value',
      data: [100000, 105000, 110000, 108000, 115000, 125000],
      borderColor: '#1976d2',
      backgroundColor: 'rgba(25, 118, 210, 0.1)',
    },
  ],
};

export const mockChatMessages = [
  {
    id: '1',
    type: 'user' as const,
    content: 'What do you think about AAPL?',
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    type: 'assistant' as const,
    content: 'AAPL is showing strong fundamentals with good growth prospects...',
    timestamp: new Date().toISOString(),
  },
];

export const mockMarketData = {
  'AAPL': {
    price: 150.00,
    change: 2.50,
    changePercent: 1.69,
    volume: 50000000,
  },
  'GOOGL': {
    price: 2500.00,
    change: -10.00,
    changePercent: -0.40,
    volume: 1500000,
  },
};

// Custom hooks for testing
export const useTestAuth = () => ({
  user: mockUser,
  isAuthenticated: true,
  login: jest.fn(),
  logout: jest.fn(),
});

export const useTestWebSocket = () => ({
  isConnected: true,
  sendMessage: jest.fn(),
  lastMessage: null,
});

// Test utilities
export const waitForLoadingToFinish = () =>
  new Promise(resolve => setTimeout(resolve, 0));

export const createMockIntersectionObserver = () => {
  const mockIntersectionObserver = jest.fn();
  mockIntersectionObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null,
  });
  window.IntersectionObserver = mockIntersectionObserver;
};

export const createMockResizeObserver = () => {
  const mockResizeObserver = jest.fn();
  mockResizeObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null,
  });
  window.ResizeObserver = mockResizeObserver;
};

// Mock API responses
export const mockApiResponses = {
  getAdvice: {
    symbol: 'AAPL',
    action: 'buy',
    confidence: 0.75,
    rationale: 'Strong upward trend with good volume support',
    keyIndicators: {
      rsi: 65,
      macd: 'bullish',
      volume: 'above_average',
    },
    riskNotes: 'Consider position sizing based on your risk tolerance',
  },
  
  getBacktest: {
    symbol: 'AAPL',
    metrics: {
      finalEquity: 11500,
      totalReturn: 15.0,
      CAGR: 12.5,
      sharpeRatio: 1.25,
      maxDrawdown: -8.5,
      winRate: 65.0,
    },
  },
  
  getChatResponse: {
    response: 'Based on the current market conditions and your risk profile...',
    suggestions: ['Consider diversifying your portfolio', 'Review your stop-loss levels'],
  },
};

// Error boundary for testing
export class TestErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <div data-testid="error-boundary">Something went wrong</div>;
    }
    return this.props.children;
  }
}

// Re-export testing library utilities
export * from '@testing-library/react';
export { customRender as render };
export { userEvent } from '@testing-library/user-event';
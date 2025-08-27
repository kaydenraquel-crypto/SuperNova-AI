/**
 * Frontend Integration Testing Suite
 * =================================
 * 
 * Comprehensive integration tests for SuperNova frontend components
 * covering user journeys, component interactions, and API integration.
 */

import React from 'react';
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from 'react-query';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import '@testing-library/jest-dom';

// Import components
import App from '../../App';
import { theme } from '../../theme';
import { TestProviders } from '../../test-utils';
import { mockApiServer } from '../../test-utils/mocks/server';

// Mock API responses
jest.mock('../../services/api', () => ({
  intakeApi: {
    createProfile: jest.fn(),
    getProfile: jest.fn(),
    updateProfile: jest.fn(),
  },
  portfolioApi: {
    getAdvice: jest.fn(),
    runBacktest: jest.fn(),
    getAnalytics: jest.fn(),
  },
  marketDataApi: {
    getQuote: jest.fn(),
    getHistoricalData: jest.fn(),
  },
  watchlistApi: {
    addSymbol: jest.fn(),
    removeSymbol: jest.fn(),
    getWatchlist: jest.fn(),
  }
}));

describe('Complete User Journey Integration Tests', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    mockApiServer.listen();
  });

  afterEach(() => {
    mockApiServer.resetHandlers();
  });

  afterAll(() => {
    mockApiServer.close();
  });

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <TestProviders>
        <BrowserRouter>
          <ThemeProvider theme={theme}>
            <QueryClientProvider client={queryClient}>
              {component}
            </QueryClientProvider>
          </ThemeProvider>
        </BrowserRouter>
      </TestProviders>
    );
  };

  describe('New User Onboarding Flow', () => {
    test('completes full onboarding journey', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Step 1: Landing page
      expect(screen.getByText(/Welcome to SuperNova/i)).toBeInTheDocument();
      
      // Click "Get Started" button
      const getStartedBtn = screen.getByRole('button', { name: /get started/i });
      await user.click(getStartedBtn);

      // Step 2: Risk Assessment
      await waitFor(() => {
        expect(screen.getByText(/Risk Assessment/i)).toBeInTheDocument();
      });

      // Complete risk questionnaire
      const riskQuestions = screen.getAllByRole('radiogroup');
      expect(riskQuestions).toHaveLength(5);

      for (let i = 0; i < riskQuestions.length; i++) {
        const radioButtons = within(riskQuestions[i]).getAllByRole('radio');
        await user.click(radioButtons[2]); // Select moderate option
      }

      const nextBtn = screen.getByRole('button', { name: /next/i });
      await user.click(nextBtn);

      // Step 3: Personal Information
      await waitFor(() => {
        expect(screen.getByLabelText(/Full Name/i)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Full Name/i), 'Test User');
      await user.type(screen.getByLabelText(/Email/i), 'test@example.com');
      await user.type(screen.getByLabelText(/Annual Income/i), '75000');

      // Select investment goals
      const goalsDropdown = screen.getByLabelText(/Investment Goals/i);
      await user.click(goalsDropdown);
      
      const retirementOption = screen.getByRole('option', { name: /retirement/i });
      await user.click(retirementOption);

      // Set time horizon
      const timeHorizonSlider = screen.getByLabelText(/Time Horizon/i);
      fireEvent.change(timeHorizonSlider, { target: { value: 15 } });

      const generatePortfolioBtn = screen.getByRole('button', { name: /generate portfolio/i });
      await user.click(generatePortfolioBtn);

      // Step 4: Portfolio Recommendation
      await waitFor(() => {
        expect(screen.getByText(/Recommended Portfolio/i)).toBeInTheDocument();
      }, { timeout: 10000 });

      // Verify portfolio components
      expect(screen.getByTestId('portfolio-allocation-chart')).toBeInTheDocument();
      expect(screen.getByTestId('risk-metrics-summary')).toBeInTheDocument();
      expect(screen.getByTestId('expected-returns')).toBeInTheDocument();

      // Accept portfolio
      const acceptBtn = screen.getByRole('button', { name: /accept portfolio/i });
      await user.click(acceptBtn);

      // Step 5: Account Creation
      await waitFor(() => {
        expect(screen.getByText(/Create Your Account/i)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Password/i), 'SecurePassword123!');
      await user.type(screen.getByLabelText(/Confirm Password/i), 'SecurePassword123!');

      const createAccountBtn = screen.getByRole('button', { name: /create account/i });
      await user.click(createAccountBtn);

      // Step 6: Dashboard
      await waitFor(() => {
        expect(screen.getByText(/Dashboard/i)).toBeInTheDocument();
      }, { timeout: 10000 });

      // Verify dashboard components
      expect(screen.getByTestId('portfolio-overview-card')).toBeInTheDocument();
      expect(screen.getByTestId('performance-chart')).toBeInTheDocument();
      expect(screen.getByTestId('asset-allocation-pie')).toBeInTheDocument();
      expect(screen.getByTestId('watchlist-widget')).toBeInTheDocument();
    });

    test('handles validation errors in onboarding', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Navigate to personal info
      await user.click(screen.getByRole('button', { name: /get started/i }));
      
      // Skip risk assessment (mock already completed)
      const skipBtn = screen.queryByRole('button', { name: /skip for now/i });
      if (skipBtn) await user.click(skipBtn);

      await waitFor(() => {
        expect(screen.getByLabelText(/Full Name/i)).toBeInTheDocument();
      });

      // Try to submit without required fields
      const generateBtn = screen.getByRole('button', { name: /generate portfolio/i });
      await user.click(generateBtn);

      // Should show validation errors
      await waitFor(() => {
        expect(screen.getByText(/Name is required/i)).toBeInTheDocument();
        expect(screen.getByText(/Email is required/i)).toBeInTheDocument();
      });

      // Fill invalid email
      await user.type(screen.getByLabelText(/Email/i), 'invalid-email');
      await user.click(generateBtn);

      await waitFor(() => {
        expect(screen.getByText(/Please enter a valid email/i)).toBeInTheDocument();
      });
    });
  });

  describe('Portfolio Management Workflow', () => {
    beforeEach(async () => {
      // Mock user is already logged in
      const user = userEvent.setup();
      renderWithProviders(<App />);
      
      // Navigate to portfolio page
      const portfolioLink = screen.getByRole('link', { name: /portfolio/i });
      await user.click(portfolioLink);
    });

    test('views and analyzes current portfolio', async () => {
      const user = userEvent.setup();

      await waitFor(() => {
        expect(screen.getByText(/Portfolio Overview/i)).toBeInTheDocument();
      });

      // Verify portfolio components
      expect(screen.getByTestId('current-allocation')).toBeInTheDocument();
      expect(screen.getByTestId('performance-metrics')).toBeInTheDocument();
      expect(screen.getByTestId('asset-breakdown')).toBeInTheDocument();

      // Test allocation pie chart interaction
      const pieChart = screen.getByTestId('allocation-pie-chart');
      await user.hover(pieChart);

      // Should show tooltip with allocation details
      await waitFor(() => {
        expect(screen.getByRole('tooltip')).toBeInTheDocument();
      });

      // Test performance time period selector
      const periodSelector = screen.getByTestId('period-selector');
      const oneYearBtn = within(periodSelector).getByRole('button', { name: /1y/i });
      await user.click(oneYearBtn);

      // Chart should update
      await waitFor(() => {
        expect(screen.getByTestId('performance-chart')).toHaveAttribute('data-period', '1y');
      });
    });

    test('rebalances portfolio', async () => {
      const user = userEvent.setup();

      await waitFor(() => {
        expect(screen.getByTestId('rebalance-button')).toBeInTheDocument();
      });

      const rebalanceBtn = screen.getByTestId('rebalance-button');
      await user.click(rebalanceBtn);

      // Rebalancing modal should open
      await waitFor(() => {
        expect(screen.getByRole('dialog', { name: /rebalance portfolio/i })).toBeInTheDocument();
      });

      // Should show current vs recommended allocations
      expect(screen.getByTestId('current-allocations')).toBeInTheDocument();
      expect(screen.getByTestId('recommended-allocations')).toBeInTheDocument();

      // Should show rebalancing impact
      expect(screen.getByTestId('rebalancing-impact')).toBeInTheDocument();
      expect(screen.getByTestId('estimated-costs')).toBeInTheDocument();

      // Accept rebalancing
      const confirmBtn = screen.getByRole('button', { name: /confirm rebalancing/i });
      await user.click(confirmBtn);

      // Should show success message
      await waitFor(() => {
        expect(screen.getByText(/Portfolio rebalanced successfully/i)).toBeInTheDocument();
      });

      // Modal should close and portfolio should update
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      });
    });

    test('adds and removes watchlist items', async () => {
      const user = userEvent.setup();

      // Find watchlist section
      const watchlistSection = screen.getByTestId('watchlist-section');
      expect(watchlistSection).toBeInTheDocument();

      // Add new symbol
      const addSymbolBtn = within(watchlistSection).getByRole('button', { name: /add symbol/i });
      await user.click(addSymbolBtn);

      // Symbol search modal should open
      await waitFor(() => {
        expect(screen.getByRole('dialog', { name: /add to watchlist/i })).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText(/search symbols/i);
      await user.type(searchInput, 'AAPL');

      // Should show search results
      await waitFor(() => {
        expect(screen.getByText(/Apple Inc/i)).toBeInTheDocument();
      });

      const appleResult = screen.getByRole('button', { name: /Apple Inc/i });
      await user.click(appleResult);

      // Should close modal and add to watchlist
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      });

      // Verify AAPL appears in watchlist
      const watchlistItems = within(watchlistSection).getAllByTestId('watchlist-item');
      const appleItem = watchlistItems.find(item => 
        within(item).queryByText('AAPL')
      );
      expect(appleItem).toBeInTheDocument();

      // Remove item from watchlist
      if (appleItem) {
        const removeBtn = within(appleItem).getByRole('button', { name: /remove/i });
        await user.click(removeBtn);

        // Should confirm removal
        await waitFor(() => {
          expect(within(watchlistSection).queryByText('AAPL')).not.toBeInTheDocument();
        });
      }
    });
  });

  describe('Analytics and Backtesting Workflow', () => {
    test('runs comprehensive backtest', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Navigate to backtesting
      const analyticsLink = screen.getByRole('link', { name: /analytics/i });
      await user.click(analyticsLink);

      const backtestingTab = screen.getByRole('tab', { name: /backtesting/i });
      await user.click(backtestingTab);

      await waitFor(() => {
        expect(screen.getByText(/Portfolio Backtesting/i)).toBeInTheDocument();
      });

      // Set backtest parameters
      const startDateInput = screen.getByLabelText(/start date/i);
      await user.clear(startDateInput);
      await user.type(startDateInput, '01/01/2020');

      const endDateInput = screen.getByLabelText(/end date/i);
      await user.clear(endDateInput);
      await user.type(endDateInput, '12/31/2023');

      const initialCapitalInput = screen.getByLabelText(/initial capital/i);
      await user.clear(initialCapitalInput);
      await user.type(initialCapitalInput, '100000');

      // Select assets
      const assetSelector = screen.getByTestId('asset-selector');
      const vtiCheckbox = within(assetSelector).getByRole('checkbox', { name: /VTI/i });
      const bndCheckbox = within(assetSelector).getByRole('checkbox', { name: /BND/i });
      
      await user.click(vtiCheckbox);
      await user.click(bndCheckbox);

      // Run backtest
      const runBacktestBtn = screen.getByRole('button', { name: /run backtest/i });
      await user.click(runBacktestBtn);

      // Should show loading state
      expect(screen.getByTestId('backtest-loading')).toBeInTheDocument();

      // Wait for results
      await waitFor(() => {
        expect(screen.getByTestId('backtest-results')).toBeInTheDocument();
      }, { timeout: 30000 });

      // Verify results components
      expect(screen.getByTestId('performance-summary')).toBeInTheDocument();
      expect(screen.getByTestId('risk-metrics')).toBeInTheDocument();
      expect(screen.getByTestId('drawdown-chart')).toBeInTheDocument();
      expect(screen.getByTestId('returns-distribution')).toBeInTheDocument();

      // Test interactive charts
      const performanceChart = screen.getByTestId('cumulative-returns-chart');
      await user.hover(performanceChart);

      // Should show tooltip with specific values
      await waitFor(() => {
        expect(screen.getByRole('tooltip')).toBeInTheDocument();
      });

      // Export results
      const exportBtn = screen.getByRole('button', { name: /export results/i });
      await user.click(exportBtn);

      const exportMenu = screen.getByRole('menu');
      const exportPdfBtn = within(exportMenu).getByRole('menuitem', { name: /pdf/i });
      await user.click(exportPdfBtn);

      // Should trigger download
      // Note: In real tests, you'd mock the download functionality
    });

    test('views advanced analytics insights', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Navigate to analytics
      const analyticsLink = screen.getByRole('link', { name: /analytics/i });
      await user.click(analyticsLink);

      const insightsTab = screen.getByRole('tab', { name: /insights/i });
      await user.click(insightsTab);

      await waitFor(() => {
        expect(screen.getByText(/Portfolio Insights/i)).toBeInTheDocument();
      });

      // Verify insights components
      expect(screen.getByTestId('performance-attribution')).toBeInTheDocument();
      expect(screen.getByTestId('risk-breakdown')).toBeInTheDocument();
      expect(screen.getByTestId('sector-analysis')).toBeInTheDocument();
      expect(screen.getByTestId('correlation-matrix')).toBeInTheDocument();

      // Test time period selector
      const periodSelector = screen.getByTestId('insights-period-selector');
      const quarterlyBtn = within(periodSelector).getByRole('button', { name: /quarterly/i });
      await user.click(quarterlyBtn);

      // Charts should update
      await waitFor(() => {
        expect(screen.getByTestId('performance-attribution')).toHaveAttribute('data-period', 'quarterly');
      });

      // Test sector drill-down
      const sectorChart = screen.getByTestId('sector-analysis');
      const technologySector = within(sectorChart).getByText(/technology/i);
      await user.click(technologySector);

      // Should show detailed sector breakdown
      await waitFor(() => {
        expect(screen.getByTestId('sector-detail-modal')).toBeInTheDocument();
      });

      // Verify sector details
      expect(screen.getByText(/Technology Sector Analysis/i)).toBeInTheDocument();
      expect(screen.getByTestId('sector-holdings')).toBeInTheDocument();
      expect(screen.getByTestId('sector-performance')).toBeInTheDocument();

      // Close modal
      const closeBtn = screen.getByRole('button', { name: /close/i });
      await user.click(closeBtn);

      await waitFor(() => {
        expect(screen.queryByTestId('sector-detail-modal')).not.toBeInTheDocument();
      });
    });
  });

  describe('Responsive Design and Accessibility', () => {
    test('adapts to mobile viewport', async () => {
      // Set mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 667,
      });

      window.dispatchEvent(new Event('resize'));

      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Mobile navigation should be present
      expect(screen.getByTestId('mobile-menu-button')).toBeInTheDocument();

      // Desktop navigation should be hidden
      expect(screen.queryByTestId('desktop-navigation')).not.toBeInTheDocument();

      // Test mobile menu
      const mobileMenuBtn = screen.getByTestId('mobile-menu-button');
      await user.click(mobileMenuBtn);

      await waitFor(() => {
        expect(screen.getByTestId('mobile-menu-drawer')).toBeInTheDocument();
      });

      // Mobile menu items should be present
      expect(screen.getByText(/Dashboard/i)).toBeInTheDocument();
      expect(screen.getByText(/Portfolio/i)).toBeInTheDocument();
      expect(screen.getByText(/Analytics/i)).toBeInTheDocument();
    });

    test('meets accessibility requirements', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Test keyboard navigation
      await user.tab();
      expect(document.activeElement).toHaveAttribute('role', 'button');

      await user.keyboard('{Enter}');
      
      // Should navigate or trigger action
      await waitFor(() => {
        expect(document.activeElement).not.toBe(document.body);
      });

      // Test screen reader labels
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName();
      });

      // Test form labels
      const inputs = screen.getAllByRole('textbox');
      inputs.forEach(input => {
        expect(input).toHaveAccessibleName();
      });

      // Test color contrast (would need additional tools in real implementation)
      // Test focus indicators
      const firstButton = buttons[0];
      firstButton.focus();
      expect(firstButton).toHaveFocus();

      // Test ARIA landmarks
      expect(screen.getByRole('main')).toBeInTheDocument();
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });

    test('handles loading states and error boundaries', async () => {
      const user = userEvent.setup();
      
      // Mock API to throw error
      const mockApi = require('../../services/api');
      mockApi.portfolioApi.getAdvice.mockRejectedValue(new Error('API Error'));

      renderWithProviders(<App />);

      // Navigate to portfolio
      const portfolioLink = screen.getByRole('link', { name: /portfolio/i });
      await user.click(portfolioLink);

      // Should show loading spinner
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

      // Should show error boundary after API failure
      await waitFor(() => {
        expect(screen.getByTestId('error-boundary')).toBeInTheDocument();
      });

      expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

      // Test retry functionality
      const retryBtn = screen.getByRole('button', { name: /retry/i });
      
      // Mock successful retry
      mockApi.portfolioApi.getAdvice.mockResolvedValue({
        allocations: [],
        risk_metrics: {}
      });

      await user.click(retryBtn);

      // Should recover from error
      await waitFor(() => {
        expect(screen.queryByTestId('error-boundary')).not.toBeInTheDocument();
      });
    });
  });

  describe('Real-time Data and WebSocket Integration', () => {
    test('receives real-time portfolio updates', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Navigate to portfolio
      const portfolioLink = screen.getByRole('link', { name: /portfolio/i });
      await user.click(portfolioLink);

      await waitFor(() => {
        expect(screen.getByTestId('portfolio-overview')).toBeInTheDocument();
      });

      // Verify WebSocket connection indicator
      expect(screen.getByTestId('connection-status')).toHaveTextContent(/connected/i);

      // Simulate real-time price update
      const mockWebSocket = require('../../services/websocket');
      mockWebSocket.emit('portfolio_update', {
        totalValue: 105000,
        dayChange: 2.5,
        holdings: [
          { symbol: 'VTI', value: 63000, change: 1.8 },
          { symbol: 'BND', value: 42000, change: 0.3 }
        ]
      });

      // Portfolio should update in real-time
      await waitFor(() => {
        expect(screen.getByTestId('portfolio-value')).toHaveTextContent('$105,000');
      });

      expect(screen.getByTestId('day-change')).toHaveTextContent('+2.5%');

      // Individual holdings should update
      const vtiHolding = screen.getByTestId('holding-VTI');
      expect(within(vtiHolding).getByTestId('holding-value')).toHaveTextContent('$63,000');
      expect(within(vtiHolding).getByTestId('holding-change')).toHaveTextContent('+1.8%');
    });

    test('handles WebSocket connection issues', async () => {
      const user = userEvent.setup();
      renderWithProviders(<App />);

      // Navigate to live tracking
      const portfolioLink = screen.getByRole('link', { name: /portfolio/i });
      await user.click(portfolioLink);

      const liveTrackingTab = screen.getByRole('tab', { name: /live tracking/i });
      await user.click(liveTrackingTab);

      await waitFor(() => {
        expect(screen.getByTestId('live-portfolio')).toBeInTheDocument();
      });

      // Simulate connection loss
      const mockWebSocket = require('../../services/websocket');
      mockWebSocket.emit('disconnect');

      // Should show disconnection warning
      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent(/disconnected/i);
      });

      expect(screen.getByText(/Connection lost/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /reconnect/i })).toBeInTheDocument();

      // Test manual reconnection
      const reconnectBtn = screen.getByRole('button', { name: /reconnect/i });
      await user.click(reconnectBtn);

      // Simulate successful reconnection
      mockWebSocket.emit('connect');

      await waitFor(() => {
        expect(screen.getByTestId('connection-status')).toHaveTextContent(/connected/i);
      });

      expect(screen.queryByText(/Connection lost/i)).not.toBeInTheDocument();
    });
  });
});

describe('Performance and Optimization Tests', () => {
  test('renders large datasets efficiently', async () => {
    const user = userEvent.setup();
    
    // Mock large dataset
    const largeWatchlist = Array.from({ length: 1000 }, (_, i) => ({
      symbol: `STOCK${i}`,
      name: `Stock Company ${i}`,
      price: 100 + Math.random() * 50,
      change: (Math.random() - 0.5) * 10
    }));

    const mockApi = require('../../services/api');
    mockApi.watchlistApi.getWatchlist.mockResolvedValue(largeWatchlist);

    const { container } = renderWithProviders(<App />);

    // Navigate to watchlist
    const watchlistLink = screen.getByRole('link', { name: /watchlist/i });
    await user.click(watchlistLink);

    const startTime = performance.now();

    await waitFor(() => {
      expect(screen.getByTestId('watchlist-virtualized')).toBeInTheDocument();
    });

    const endTime = performance.now();
    const renderTime = endTime - startTime;

    // Should render large list quickly (under 2 seconds)
    expect(renderTime).toBeLessThan(2000);

    // Should use virtualization for performance
    expect(screen.getByTestId('watchlist-virtualized')).toBeInTheDocument();

    // Only visible items should be rendered
    const renderedItems = container.querySelectorAll('[data-testid^="watchlist-item-"]');
    expect(renderedItems.length).toBeLessThan(50); // Should not render all 1000 items
  });

  test('implements efficient chart rendering', async () => {
    const user = userEvent.setup();
    
    // Mock large historical data
    const largeDataset = Array.from({ length: 5000 }, (_, i) => ({
      timestamp: new Date(2020, 0, 1 + i).toISOString(),
      value: 100 + Math.sin(i / 100) * 20 + Math.random() * 5
    }));

    const mockApi = require('../../services/api');
    mockApi.portfolioApi.getAnalytics.mockResolvedValue({
      performanceData: largeDataset
    });

    renderWithProviders(<App />);

    // Navigate to analytics
    const analyticsLink = screen.getByRole('link', { name: /analytics/i });
    await user.click(analyticsLink);

    const startTime = performance.now();

    await waitFor(() => {
      expect(screen.getByTestId('performance-chart')).toBeInTheDocument();
    });

    const endTime = performance.now();
    const chartRenderTime = endTime - startTime;

    // Chart should render efficiently even with large dataset
    expect(chartRenderTime).toBeLessThan(3000);

    // Should implement data downsampling or other optimizations
    const chart = screen.getByTestId('performance-chart');
    expect(chart).toHaveAttribute('data-optimized', 'true');
  });
});

// Helper functions for common test patterns
const expectPortfolioComponentsToBeVisible = () => {
  expect(screen.getByTestId('portfolio-overview')).toBeInTheDocument();
  expect(screen.getByTestId('asset-allocation')).toBeInTheDocument();
  expect(screen.getByTestId('performance-metrics')).toBeInTheDocument();
};

const expectLoadingStateToResolve = async () => {
  await waitFor(() => {
    expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
  });
};

const expectNoErrors = () => {
  expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
  expect(screen.queryByTestId('error-boundary')).not.toBeInTheDocument();
};
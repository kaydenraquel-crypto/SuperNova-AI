/**
 * Tests for PortfolioOverviewCard component
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PortfolioOverviewCard } from '../PortfolioOverviewCard';
import { mockPortfolioData } from '../../../test-utils';

describe('PortfolioOverviewCard', () => {
  test('renders portfolio overview with correct data', () => {
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
    expect(screen.getByText('$125,000.00')).toBeInTheDocument(); // Total value
    expect(screen.getByText('+$25,000.00')).toBeInTheDocument(); // Total return
    expect(screen.getByText('25.0%')).toBeInTheDocument(); // Return percentage
  });

  test('displays daily change with correct formatting', () => {
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    expect(screen.getByText('+$1,250.00')).toBeInTheDocument(); // Day change
    expect(screen.getByText('+1.01%')).toBeInTheDocument(); // Day change percentage
  });

  test('shows negative changes in red', () => {
    const negativeData = {
      ...mockPortfolioData,
      dayChange: -500,
      dayChangePercent: -0.5,
      totalReturn: -2000,
      totalReturnPercent: -2.0,
    };

    render(<PortfolioOverviewCard data={negativeData} />);

    const negativeElements = screen.getAllByText(/-\$|-%/);
    negativeElements.forEach(element => {
      expect(element).toHaveStyle({ color: expect.stringContaining('error') });
    });
  });

  test('shows positive changes in green', () => {
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    const positiveElements = screen.getAllByText(/\+\$|\+%/);
    positiveElements.forEach(element => {
      expect(element).toHaveStyle({ color: expect.stringContaining('success') });
    });
  });

  test('renders loading state when no data provided', () => {
    render(<PortfolioOverviewCard data={undefined} />);

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test('displays positions count', () => {
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    expect(screen.getByText('2 Positions')).toBeInTheDocument();
  });

  test('expands to show detailed positions', async () => {
    const user = userEvent.setup();
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    const expandButton = screen.getByRole('button', { name: /expand/i });
    await user.click(expandButton);

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
      expect(screen.getByText('GOOGL')).toBeInTheDocument();
      expect(screen.getByText('Alphabet Inc.')).toBeInTheDocument();
    });
  });

  test('shows individual position performance', async () => {
    const user = userEvent.setup();
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    const expandButton = screen.getByRole('button', { name: /expand/i });
    await user.click(expandButton);

    await waitFor(() => {
      // Check AAPL position
      expect(screen.getByText('$15,000.00')).toBeInTheDocument(); // Market value
      expect(screen.getByText('+$3,000.00')).toBeInTheDocument(); // P&L
      
      // Check GOOGL position
      expect(screen.getByText('$125,000.00')).toBeInTheDocument(); // Market value
      expect(screen.getByText('+$25,000.00')).toBeInTheDocument(); // P&L
    });
  });

  test('handles empty positions array', () => {
    const emptyData = {
      ...mockPortfolioData,
      positions: [],
    };

    render(<PortfolioOverviewCard data={emptyData} />);

    expect(screen.getByText('0 Positions')).toBeInTheDocument();
    expect(screen.getByText('$125,000.00')).toBeInTheDocument(); // Still shows total
  });

  test('formats large numbers correctly', () => {
    const largeData = {
      totalValue: 1250000,
      totalReturn: 250000,
      totalReturnPercent: 25.0,
      dayChange: 12500,
      dayChangePercent: 1.01,
      positions: [],
    };

    render(<PortfolioOverviewCard data={largeData} />);

    expect(screen.getByText('$1,250,000.00')).toBeInTheDocument();
    expect(screen.getByText('+$250,000.00')).toBeInTheDocument();
  });

  test('handles zero values correctly', () => {
    const zeroData = {
      totalValue: 0,
      totalReturn: 0,
      totalReturnPercent: 0,
      dayChange: 0,
      dayChangePercent: 0,
      positions: [],
    };

    render(<PortfolioOverviewCard data={zeroData} />);

    expect(screen.getByText('$0.00')).toBeInTheDocument();
    expect(screen.getByText('0.0%')).toBeInTheDocument();
  });

  test('includes refresh functionality', async () => {
    const mockRefresh = jest.fn();
    const user = userEvent.setup();

    render(
      <PortfolioOverviewCard 
        data={mockPortfolioData} 
        onRefresh={mockRefresh}
      />
    );

    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    await user.click(refreshButton);

    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  test('shows last updated timestamp', () => {
    const dataWithTimestamp = {
      ...mockPortfolioData,
      lastUpdated: '2024-01-01T10:30:00Z',
    };

    render(<PortfolioOverviewCard data={dataWithTimestamp} />);

    expect(screen.getByText(/last updated/i)).toBeInTheDocument();
    expect(screen.getByText(/10:30/)).toBeInTheDocument();
  });

  test('has proper accessibility attributes', () => {
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    const card = screen.getByRole('region');
    expect(card).toHaveAttribute('aria-label', expect.stringContaining('Portfolio'));

    const totalValue = screen.getByText('$125,000.00');
    expect(totalValue).toHaveAttribute('aria-label', 'Total portfolio value 125000 dollars');
  });

  test('responds to keyboard navigation', async () => {
    const user = userEvent.setup();
    render(<PortfolioOverviewCard data={mockPortfolioData} />);

    const expandButton = screen.getByRole('button', { name: /expand/i });
    
    // Focus the button
    expandButton.focus();
    expect(expandButton).toHaveFocus();

    // Press Enter to expand
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument();
    });
  });
});
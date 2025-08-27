/**
 * Tests for LoadingScreen component
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import { LoadingScreen } from '../LoadingScreen';

describe('LoadingScreen', () => {
  test('renders loading spinner by default', () => {
    render(<LoadingScreen />);
    
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test('renders custom message when provided', () => {
    const customMessage = 'Loading your portfolio data...';
    
    render(<LoadingScreen message={customMessage} />);
    
    expect(screen.getByText(customMessage)).toBeInTheDocument();
  });

  test('renders with backdrop when specified', () => {
    render(<LoadingScreen backdrop={true} />);
    
    const backdrop = screen.getByTestId('loading-backdrop');
    expect(backdrop).toBeInTheDocument();
    expect(backdrop).toHaveStyle({ position: 'fixed' });
  });

  test('renders without backdrop by default', () => {
    render(<LoadingScreen />);
    
    expect(screen.queryByTestId('loading-backdrop')).not.toBeInTheDocument();
  });

  test('renders with different sizes', () => {
    const { rerender } = render(<LoadingScreen size="small" />);
    
    let spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveAttribute('aria-label', expect.stringContaining('small'));

    rerender(<LoadingScreen size="large" />);
    
    spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveAttribute('aria-label', expect.stringContaining('large'));
  });

  test('has proper accessibility attributes', () => {
    render(<LoadingScreen message="Loading data" />);
    
    const spinner = screen.getByRole('progressbar');
    expect(spinner).toHaveAttribute('aria-label');
    
    const message = screen.getByText('Loading data');
    expect(message).toHaveAttribute('aria-live', 'polite');
  });

  test('supports custom className', () => {
    render(<LoadingScreen className="custom-loading" />);
    
    const container = screen.getByTestId('loading-container');
    expect(container).toHaveClass('custom-loading');
  });

  test('renders with progress percentage when provided', () => {
    render(<LoadingScreen progress={65} message="Processing..." />);
    
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
    
    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toHaveAttribute('aria-valuenow', '65');
  });

  test('handles zero progress', () => {
    render(<LoadingScreen progress={0} />);
    
    expect(screen.getByText('0%')).toBeInTheDocument();
    
    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toHaveAttribute('aria-valuenow', '0');
  });

  test('handles complete progress', () => {
    render(<LoadingScreen progress={100} />);
    
    expect(screen.getByText('100%')).toBeInTheDocument();
    
    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toHaveAttribute('aria-valuenow', '100');
  });

  test('renders with subtitle when provided', () => {
    render(
      <LoadingScreen 
        message="Loading Portfolio" 
        subtitle="Fetching latest market data..."
      />
    );
    
    expect(screen.getByText('Loading Portfolio')).toBeInTheDocument();
    expect(screen.getByText('Fetching latest market data...')).toBeInTheDocument();
  });
});
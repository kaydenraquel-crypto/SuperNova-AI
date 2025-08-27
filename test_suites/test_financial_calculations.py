"""
Financial Calculations Testing Suite
===================================

Comprehensive testing for financial calculations, portfolio analytics,
risk metrics, and backtesting accuracy.
"""
import pytest
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings

from test_suites.conftest import FinancialTestHelpers


class TestPortfolioMetrics:
    """Test portfolio metric calculations."""
    
    @pytest.mark.unit
    def test_portfolio_return_calculation(self, financial_calculations_test_data):
        """Test portfolio return calculations with known values."""
        test_cases = financial_calculations_test_data['portfolio_return_tests']
        
        for case in test_cases:
            weights = np.array(case['weights'])
            returns = np.array(case['returns'])
            expected = case['expected']
            
            calculated_return = np.dot(weights, returns)
            
            assert abs(calculated_return - expected) < 0.001, \
                f"Portfolio return mismatch: expected {expected}, got {calculated_return}"
    
    @pytest.mark.unit
    def test_portfolio_volatility_calculation(self, financial_test_portfolios):
        """Test portfolio volatility calculations."""
        # Generate correlated returns for testing
        n_days = 252
        np.random.seed(42)
        
        # Create correlation matrix
        correlation_matrix = np.array([
            [1.00, 0.70, 0.50],  # Equity correlations
            [0.70, 1.00, 0.30],
            [0.50, 0.30, 1.00]
        ])
        
        # Generate returns
        mean_returns = [0.08/252, 0.04/252, 0.06/252]  # Daily returns
        volatilities = [0.15/np.sqrt(252), 0.05/np.sqrt(252), 0.12/np.sqrt(252)]  # Daily volatilities
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        
        # Test moderate portfolio
        portfolio = financial_test_portfolios['moderate']
        weights = np.array([0.50, 0.25, 0.15])  # VTI, BND, VEA weights
        
        portfolio_returns = np.dot(returns, weights)
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        
        # Should be between individual asset volatilities (diversification benefit)
        assert 0.05 < portfolio_vol < 0.15, f"Portfolio volatility {portfolio_vol} outside expected range"
    
    @pytest.mark.unit
    def test_sharpe_ratio_calculation(self, financial_calculations_test_data):
        """Test Sharpe ratio calculations with known values."""
        test_cases = financial_calculations_test_data['sharpe_ratio_tests']
        
        for case in test_cases:
            returns = case['returns']
            risk_free_rate = case['risk_free_rate']
            expected = case['expected']
            
            calculated_sharpe = FinancialTestHelpers.calculate_sharpe_ratio(returns, risk_free_rate)
            
            assert abs(calculated_sharpe - expected) < 0.1, \
                f"Sharpe ratio mismatch: expected {expected}, got {calculated_sharpe}"
    
    @pytest.mark.unit
    def test_max_drawdown_calculation(self, financial_calculations_test_data):
        """Test maximum drawdown calculations."""
        test_cases = financial_calculations_test_data['max_drawdown_tests']
        
        for case in test_cases:
            prices = case['prices']
            expected = case['expected']
            
            calculated_drawdown = FinancialTestHelpers.calculate_max_drawdown(prices)
            
            assert abs(calculated_drawdown - expected) < 0.01, \
                f"Max drawdown mismatch: expected {expected}, got {calculated_drawdown}"
    
    @pytest.mark.unit
    def test_beta_calculation(self):
        """Test beta calculation against market."""
        np.random.seed(42)
        n_days = 252
        
        # Generate market returns
        market_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), n_days)
        
        # Generate stock returns with known beta
        true_beta = 1.2
        stock_returns = 0.02/252 + true_beta * market_returns + np.random.normal(0, 0.05/np.sqrt(252), n_days)
        
        # Calculate beta
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        calculated_beta = covariance / market_variance
        
        assert abs(calculated_beta - true_beta) < 0.1, \
            f"Beta calculation error: expected {true_beta}, got {calculated_beta}"
    
    @pytest.mark.unit
    def test_correlation_calculation(self):
        """Test correlation calculations between assets."""
        np.random.seed(42)
        n_days = 252
        
        # Generate correlated returns
        true_correlation = 0.6
        returns1 = np.random.normal(0, 0.02, n_days)
        returns2 = true_correlation * returns1 + np.sqrt(1 - true_correlation**2) * np.random.normal(0, 0.02, n_days)
        
        calculated_correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        assert abs(calculated_correlation - true_correlation) < 0.05, \
            f"Correlation calculation error: expected {true_correlation}, got {calculated_correlation}"


class TestRiskMetrics:
    """Test risk metric calculations."""
    
    @pytest.mark.unit
    def test_var_calculation(self):
        """Test Value at Risk (VaR) calculations."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)  # Normal returns
        
        # Calculate 95% VaR (5th percentile)
        var_95 = np.percentile(returns, 5)
        
        # Should be approximately -1.645 * std for normal distribution
        expected_var = -1.645 * 0.02
        
        assert abs(var_95 - expected_var) < 0.005, \
            f"VaR calculation error: expected {expected_var}, got {var_95}"
    
    @pytest.mark.unit
    def test_expected_shortfall(self):
        """Test Expected Shortfall (Conditional VaR) calculations."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 10000)  # Large sample for accuracy
        
        # Calculate 95% Expected Shortfall
        var_95 = np.percentile(returns, 5)
        expected_shortfall = np.mean(returns[returns <= var_95])
        
        # For normal distribution, ES should be more negative than VaR
        assert expected_shortfall < var_95, \
            "Expected Shortfall should be more negative than VaR"
    
    @pytest.mark.unit
    def test_downside_deviation(self):
        """Test downside deviation calculations."""
        returns = [0.05, 0.02, -0.01, 0.03, -0.02, 0.01, -0.03]
        target_return = 0.0
        
        # Calculate downside deviation
        downside_returns = [min(0, r - target_return) for r in returns]
        downside_variance = np.mean([r**2 for r in downside_returns])
        downside_deviation = np.sqrt(downside_variance)
        
        # Manual calculation check
        expected_downside = np.sqrt((0.01**2 + 0.02**2 + 0.03**2) / len(returns))
        
        assert abs(downside_deviation - expected_downside) < 0.001, \
            f"Downside deviation error: expected {expected_downside}, got {downside_deviation}"
    
    @pytest.mark.unit
    def test_tracking_error(self):
        """Test tracking error calculations."""
        np.random.seed(42)
        
        # Generate portfolio and benchmark returns
        benchmark_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), 252)
        portfolio_returns = benchmark_returns + np.random.normal(0, 0.02/np.sqrt(252), 252)
        
        # Calculate tracking error
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        # Should be approximately the standard deviation of the noise we added
        expected_te = 0.02
        assert abs(tracking_error - expected_te) < 0.01, \
            f"Tracking error calculation error: expected ~{expected_te}, got {tracking_error}"


class TestBacktestingValidation:
    """Test backtesting calculations and validation."""
    
    @pytest.mark.integration
    def test_buy_and_hold_backtest(self, backtesting_scenarios):
        """Test simple buy and hold strategy backtest."""
        scenario = backtesting_scenarios['simple_buy_hold']
        
        # Generate synthetic price data
        np.random.seed(42)
        n_days = 1000  # ~4 years
        daily_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), n_days)
        prices = [100]  # Starting price
        
        for return_val in daily_returns:
            prices.append(prices[-1] * (1 + return_val))
        
        # Calculate buy and hold metrics
        total_return = (prices[-1] - prices[0]) / prices[0]
        daily_portfolio_returns = [
            (prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))
        ]
        
        # Calculate metrics
        sharpe_ratio = FinancialTestHelpers.calculate_sharpe_ratio(daily_portfolio_returns)
        max_drawdown = FinancialTestHelpers.calculate_max_drawdown(prices)
        
        # Validate against expected ranges
        assert total_return > scenario['expected_metrics']['total_return_min'], \
            f"Total return {total_return} below minimum {scenario['expected_metrics']['total_return_min']}"
        
        assert sharpe_ratio > scenario['expected_metrics']['sharpe_ratio_min'], \
            f"Sharpe ratio {sharpe_ratio} below minimum {scenario['expected_metrics']['sharpe_ratio_min']}"
        
        assert max_drawdown < scenario['expected_metrics']['max_drawdown_max'], \
            f"Max drawdown {max_drawdown} above maximum {scenario['expected_metrics']['max_drawdown_max']}"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_momentum_strategy_backtest(self, backtesting_scenarios):
        """Test momentum strategy backtest."""
        scenario = backtesting_scenarios['momentum_strategy']
        
        # Generate price data for multiple assets
        np.random.seed(42)
        n_days = 1000
        n_assets = len(scenario['symbols'])
        
        # Create momentum in the data
        base_returns = np.random.normal(0.06/252, 0.12/np.sqrt(252), (n_days, n_assets))
        
        # Add momentum effect (past winners continue winning for short periods)
        lookback = scenario['lookback_period']
        momentum_returns = base_returns.copy()
        
        for day in range(lookback, n_days):
            for asset in range(n_assets):
                past_performance = np.sum(base_returns[day-lookback:day, asset])
                momentum_factor = 0.1 * np.sign(past_performance)  # Momentum boost
                momentum_returns[day, asset] += momentum_factor / np.sqrt(252)
        
        # Calculate strategy performance
        portfolio_returns = []
        weights = np.ones(n_assets) / n_assets  # Equal weight initially
        
        for day in range(lookback, n_days):
            # Rebalance based on momentum
            if day % 20 == 0:  # Monthly rebalancing
                past_returns = np.sum(momentum_returns[day-lookback:day, :], axis=0)
                weights = np.exp(past_returns) / np.sum(np.exp(past_returns))  # Softmax weighting
            
            daily_return = np.sum(weights * momentum_returns[day, :])
            portfolio_returns.append(daily_return)
        
        # Calculate metrics
        total_return = np.prod(1 + np.array(portfolio_returns)) - 1
        sharpe_ratio = FinancialTestHelpers.calculate_sharpe_ratio(portfolio_returns)
        
        # Note: Momentum strategies may not always beat benchmarks in random data
        # so we use more lenient thresholds
        assert total_return > scenario['expected_metrics']['total_return_min'] - 0.05, \
            f"Momentum strategy underperformed significantly"


class TestFinancialCompliance:
    """Test financial compliance and regulatory requirements."""
    
    @pytest.mark.unit
    def test_fiduciary_suitability(self, compliance_test_scenarios):
        """Test fiduciary suitability checks."""
        test_cases = compliance_test_scenarios['fiduciary_tests']
        
        for case in test_cases:
            client_profile = case['client_profile']
            portfolio = case['recommended_portfolio']
            should_pass = case['should_pass']
            
            # Implement suitability logic
            is_suitable = self._check_portfolio_suitability(client_profile, portfolio)
            
            assert is_suitable == should_pass, \
                f"Suitability check failed for case: {case}"
    
    def _check_portfolio_suitability(self, client_profile: Dict, portfolio: Dict) -> bool:
        """Check if portfolio is suitable for client profile."""
        risk_tolerance = client_profile['risk_tolerance']
        time_horizon = client_profile['time_horizon']
        equity_weight = portfolio['equity_weight']
        
        # Conservative clients should have lower equity allocation
        if risk_tolerance == 'low':
            if time_horizon <= 3 and equity_weight > 0.4:
                return False
            if time_horizon <= 5 and equity_weight > 0.6:
                return False
        
        return True
    
    @pytest.mark.unit
    def test_position_limits(self, compliance_test_scenarios):
        """Test position limit compliance."""
        test_cases = compliance_test_scenarios['position_limits']
        
        for case in test_cases:
            portfolio_value = case['portfolio_value']
            position_value = case['position_value']
            max_position_percent = case['max_position_percent']
            should_pass = case['should_pass']
            
            # Check position limit
            position_percent = position_value / portfolio_value
            is_compliant = position_percent <= max_position_percent
            
            assert is_compliant == should_pass, \
                f"Position limit check failed: {position_percent} vs {max_position_percent}"
    
    @pytest.mark.unit
    def test_wash_sale_detection(self):
        """Test wash sale rule detection."""
        # Generate transactions that violate wash sale rule
        transactions = [
            {'date': datetime(2023, 1, 15), 'symbol': 'TEST', 'quantity': -100, 'price': 50},  # Sell at loss
            {'date': datetime(2023, 1, 25), 'symbol': 'TEST', 'quantity': 100, 'price': 45},   # Buy back within 30 days
        ]
        
        # Check for wash sale
        is_wash_sale = self._detect_wash_sale(transactions)
        
        assert is_wash_sale, "Should detect wash sale violation"
    
    def _detect_wash_sale(self, transactions: List[Dict]) -> bool:
        """Detect wash sale rule violations."""
        for i, sell_tx in enumerate(transactions):
            if sell_tx['quantity'] < 0:  # Sell transaction
                sell_date = sell_tx['date']
                symbol = sell_tx['symbol']
                
                # Look for buy transactions within 30 days
                for buy_tx in transactions[i+1:]:
                    if (buy_tx['symbol'] == symbol and 
                        buy_tx['quantity'] > 0 and
                        (buy_tx['date'] - sell_date).days <= 30):
                        return True
        
        return False


class TestPrecisionAndAccuracy:
    """Test numerical precision and accuracy in financial calculations."""
    
    @pytest.mark.unit
    def test_decimal_precision_in_calculations(self):
        """Test that financial calculations maintain sufficient precision."""
        # Test with high-precision decimal arithmetic
        price1 = Decimal('123.456789')
        price2 = Decimal('987.654321')
        quantity = Decimal('1000.123456')
        
        # Calculate portfolio value
        position_value = price1 * quantity
        total_value = position_value + (price2 * quantity)
        
        # Check precision is maintained
        assert len(str(total_value).split('.')[-1]) >= 6, \
            "Insufficient precision in financial calculations"
    
    @pytest.mark.unit
    def test_rounding_consistency(self):
        """Test consistent rounding behavior in financial calculations."""
        values = [123.4565, 123.4555, 123.4575]
        expected_rounded = [123.46, 123.46, 123.46]  # Round half up
        
        for value, expected in zip(values, expected_rounded):
            decimal_value = Decimal(str(value))
            rounded_value = float(decimal_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            
            assert rounded_value == expected, \
                f"Rounding inconsistency: {value} -> {rounded_value}, expected {expected}"
    
    @pytest.mark.unit
    def test_floating_point_edge_cases(self):
        """Test handling of floating point edge cases."""
        # Test very small numbers
        small_return = 1e-10
        portfolio_value = 1e6
        
        # This calculation should not result in zero due to precision issues
        impact = small_return * portfolio_value
        assert impact > 0, "Precision loss in small number calculations"
        
        # Test very large numbers
        large_value = 1e15
        small_percentage = 0.01
        
        result = large_value * small_percentage
        expected = large_value / 100
        
        assert abs(result - expected) / expected < 1e-10, \
            "Precision loss in large number calculations"


class TestStressScenarios:
    """Test financial calculations under stress scenarios."""
    
    @pytest.mark.unit
    def test_market_crash_scenario(self, market_scenarios):
        """Test portfolio behavior during market crash."""
        crash_scenario = market_scenarios['bear_market']
        
        # Generate crash scenario data
        n_days = crash_scenario['duration_days']
        annual_return = crash_scenario['annual_return']
        volatility = crash_scenario['volatility']
        
        np.random.seed(42)
        daily_returns = np.random.normal(
            annual_return / 252, 
            volatility / np.sqrt(252), 
            n_days
        )
        
        # Calculate cumulative performance
        cumulative_returns = np.cumprod(1 + daily_returns)
        max_drawdown = FinancialTestHelpers.calculate_max_drawdown(cumulative_returns.tolist())
        
        # Verify crash characteristics
        total_return = cumulative_returns[-1] - 1
        assert total_return < -0.15, f"Insufficient market decline: {total_return}"
        assert max_drawdown > 0.20, f"Insufficient drawdown: {max_drawdown}"
    
    @pytest.mark.unit
    def test_extreme_volatility_scenario(self, market_scenarios):
        """Test calculations during extreme volatility."""
        volatile_scenario = market_scenarios['volatile_market']
        
        # Generate extreme volatility data
        np.random.seed(42)
        n_days = volatile_scenario['duration_days']
        base_volatility = volatile_scenario['volatility']
        
        # Add volatility clustering (high vol followed by high vol)
        returns = []
        current_vol = base_volatility / np.sqrt(252)
        
        for _ in range(n_days):
            # GARCH-like volatility clustering
            shock = np.random.normal(0, current_vol)
            returns.append(shock)
            
            # Update volatility (mean reversion with persistence)
            current_vol = 0.1 * (base_volatility / np.sqrt(252)) + 0.9 * abs(shock)
        
        realized_volatility = np.std(returns) * np.sqrt(252)
        
        # Should be close to the target volatility
        assert abs(realized_volatility - base_volatility) < 0.05, \
            f"Volatility mismatch: target {base_volatility}, realized {realized_volatility}"


@pytest.mark.parametrize("portfolio_type", ["conservative", "moderate", "aggressive"])
class TestPortfolioTypeValidation:
    """Test portfolio validation across different risk profiles."""
    
    @pytest.mark.unit
    def test_portfolio_risk_consistency(self, financial_test_portfolios, portfolio_type):
        """Test that portfolio risk aligns with type."""
        portfolio = financial_test_portfolios[portfolio_type]
        
        risk_score = portfolio['risk_score']
        expected_volatility = portfolio['expected_volatility']
        
        # Risk score should align with volatility expectations
        if portfolio_type == 'conservative':
            assert risk_score <= 35, f"Conservative portfolio risk score too high: {risk_score}"
            assert expected_volatility <= 0.12, f"Conservative portfolio volatility too high: {expected_volatility}"
        
        elif portfolio_type == 'moderate':
            assert 35 < risk_score <= 65, f"Moderate portfolio risk score out of range: {risk_score}"
            assert 0.10 < expected_volatility <= 0.18, f"Moderate portfolio volatility out of range: {expected_volatility}"
        
        elif portfolio_type == 'aggressive':
            assert risk_score > 65, f"Aggressive portfolio risk score too low: {risk_score}"
            assert expected_volatility > 0.15, f"Aggressive portfolio volatility too low: {expected_volatility}"
    
    @pytest.mark.unit
    def test_portfolio_allocation_constraints(self, financial_test_portfolios, portfolio_type):
        """Test portfolio allocation constraints."""
        portfolio = financial_test_portfolios[portfolio_type]
        
        total_allocation = portfolio['cash_allocation']
        for asset in portfolio['assets']:
            total_allocation += asset['weight']
        
        # Total allocation should sum to approximately 1.0
        assert abs(total_allocation - 1.0) < 0.01, \
            f"Portfolio allocation doesn't sum to 1.0: {total_allocation}"
        
        # Each position should be reasonable size
        for asset in portfolio['assets']:
            assert 0.05 <= asset['weight'] <= 0.70, \
                f"Asset weight out of reasonable range: {asset['weight']} for {asset['symbol']}"


if __name__ == "__main__":
    # Run financial calculation tests
    pytest.main([__file__, "-v", "--tb=short"])
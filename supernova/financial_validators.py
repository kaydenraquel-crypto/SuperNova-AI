"""
SuperNova AI Financial Data Validators
Advanced business logic validation for financial data and transactions
"""

from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import re
from enum import Enum
from pydantic import validator, ValidationError

from .input_validation import input_validator, ValidationError as InputValidationError


class MarketRegime(str, Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class AssetClass(str, Enum):
    """Supported asset classes"""
    STOCK = "stock"
    BOND = "bond"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"
    REIT = "reit"


class RiskLevel(str, Enum):
    """Risk level classifications"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"


class FinancialDataValidator:
    """
    Comprehensive financial data validation with business logic
    """
    
    def __init__(self):
        self.market_hours = {
            'NYSE': {'open': 9.5, 'close': 16.0, 'timezone': 'US/Eastern'},
            'NASDAQ': {'open': 9.5, 'close': 16.0, 'timezone': 'US/Eastern'},
            'LSE': {'open': 8.0, 'close': 16.5, 'timezone': 'Europe/London'},
            'TSE': {'open': 9.0, 'close': 15.0, 'timezone': 'Asia/Tokyo'},
        }
        
        self.currency_pairs = {
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
        }
        
        self.major_exchanges = {
            'NYSE', 'NASDAQ', 'LSE', 'TSE', 'HKEX', 'SSE', 'BSE', 'TSX'
        }
        
        # Risk limits by asset class
        self.asset_class_limits = {
            AssetClass.STOCK: {'max_position': 0.10, 'max_sector': 0.25},
            AssetClass.CRYPTO: {'max_position': 0.05, 'max_total': 0.15},
            AssetClass.OPTION: {'max_position': 0.02, 'max_total': 0.10},
            AssetClass.COMMODITY: {'max_position': 0.08, 'max_total': 0.20},
        }
    
    def validate_stock_symbol(self, symbol: str) -> str:
        """Validate stock symbol with comprehensive business rules"""
        # Basic format validation
        symbol = input_validator.validate_financial_symbol(symbol)
        
        # Check symbol length and format patterns
        if len(symbol) < 1 or len(symbol) > 6:
            raise ValidationError("Stock symbol must be 1-6 characters")
        
        # Special handling for different markets
        if '.' in symbol:
            # Handle symbols with suffixes (e.g., BRK.A)
            parts = symbol.split('.')
            if len(parts) != 2 or len(parts[1]) > 2:
                raise ValidationError("Invalid symbol suffix format")
        
        # Check against known delisted or suspended symbols
        forbidden_symbols = {'ENRON', 'LEHMAN', 'WCOM'}  # Example blacklist
        if symbol in forbidden_symbols:
            raise ValidationError(f"Symbol {symbol} is not tradeable")
        
        return symbol
    
    def validate_price_data(
        self, 
        open_price: Decimal, 
        high: Decimal, 
        low: Decimal, 
        close: Decimal,
        volume: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """Validate OHLC price relationships with business logic"""
        
        # Convert to Decimal for precision
        prices = {
            'open': Decimal(str(open_price)),
            'high': Decimal(str(high)),
            'low': Decimal(str(low)),
            'close': Decimal(str(close))
        }
        
        # Basic OHLC relationship validation
        if not (prices['low'] <= prices['open'] <= prices['high']):
            raise ValidationError("Open price must be between low and high")
        
        if not (prices['low'] <= prices['close'] <= prices['high']):
            raise ValidationError("Close price must be between low and high")
        
        if prices['high'] < prices['low']:
            raise ValidationError("High price cannot be less than low price")
        
        # Business logic validations
        price_range = prices['high'] - prices['low']
        mid_price = (prices['high'] + prices['low']) / 2
        
        # Check for unrealistic price gaps (> 50% from mid-price)
        for price_type, price in prices.items():
            if price_type in ['open', 'close']:
                deviation = abs(price - mid_price) / mid_price
                if deviation > Decimal('0.5'):
                    raise ValidationError(f"{price_type} price shows unrealistic gap: {deviation:.1%}")
        
        # Volume validation if provided
        if volume is not None:
            volume_decimal = Decimal(str(volume))
            if volume_decimal < 0:
                raise ValidationError("Volume cannot be negative")
            
            # Check for unrealistic volume (basic sanity check)
            if volume_decimal > Decimal('1000000000'):  # 1B shares
                raise ValidationError("Volume appears unrealistically high")
        
        return prices
    
    def validate_portfolio_allocation(
        self, 
        positions: List[Dict[str, Any]],
        risk_level: RiskLevel = RiskLevel.MODERATE
    ) -> Dict[str, Any]:
        """Validate portfolio allocation with risk management rules"""
        
        total_weight = Decimal('0')
        asset_class_totals = {}
        sector_totals = {}
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'risk_metrics': {}
        }
        
        for position in positions:
            try:
                # Validate individual position
                symbol = self.validate_stock_symbol(position['symbol'])
                weight = Decimal(str(position.get('weight', 0)))
                asset_class = AssetClass(position.get('asset_class', 'stock'))
                sector = position.get('sector', 'unknown')
                
                # Accumulate totals
                total_weight += weight
                asset_class_totals[asset_class] = asset_class_totals.get(asset_class, Decimal('0')) + weight
                sector_totals[sector] = sector_totals.get(sector, Decimal('0')) + weight
                
                # Individual position risk checks
                if asset_class in self.asset_class_limits:
                    max_position = Decimal(str(self.asset_class_limits[asset_class]['max_position']))
                    if weight > max_position:
                        validation_results['errors'].append(
                            f"Position {symbol} exceeds maximum weight for {asset_class}: {weight} > {max_position}"
                        )
                        validation_results['valid'] = False
                
            except (ValidationError, ValueError, KeyError) as e:
                validation_results['errors'].append(f"Invalid position data: {str(e)}")
                validation_results['valid'] = False
        
        # Portfolio-level validations
        if total_weight > Decimal('1.05'):  # Allow for small rounding errors
            validation_results['errors'].append(f"Total portfolio weight exceeds 100%: {total_weight}")
            validation_results['valid'] = False
        elif total_weight < Decimal('0.95'):
            validation_results['warnings'].append(f"Portfolio appears under-allocated: {total_weight}")
        
        # Asset class concentration checks
        for asset_class, total in asset_class_totals.items():
            if asset_class in self.asset_class_limits:
                max_total = Decimal(str(self.asset_class_limits[asset_class].get('max_total', 1.0)))
                if total > max_total:
                    validation_results['errors'].append(
                        f"Asset class {asset_class} exceeds maximum allocation: {total} > {max_total}"
                    )
                    validation_results['valid'] = False
        
        # Sector concentration checks
        for sector, total in sector_totals.items():
            if total > Decimal('0.30'):  # No sector > 30%
                validation_results['warnings'].append(
                    f"High sector concentration in {sector}: {total:.1%}"
                )
        
        # Risk-adjusted validations based on risk level
        if risk_level == RiskLevel.CONSERVATIVE:
            crypto_allocation = asset_class_totals.get(AssetClass.CRYPTO, Decimal('0'))
            if crypto_allocation > Decimal('0.05'):
                validation_results['errors'].append(
                    f"Crypto allocation too high for conservative profile: {crypto_allocation:.1%}"
                )
                validation_results['valid'] = False
        
        elif risk_level == RiskLevel.SPECULATIVE:
            safe_assets = asset_class_totals.get(AssetClass.BOND, Decimal('0'))
            if safe_assets < Decimal('0.10'):
                validation_results['warnings'].append(
                    "Consider some safe assets even with speculative risk tolerance"
                )
        
        # Calculate risk metrics
        validation_results['risk_metrics'] = {
            'total_allocation': float(total_weight),
            'asset_class_distribution': {k.value: float(v) for k, v in asset_class_totals.items()},
            'sector_distribution': {k: float(v) for k, v in sector_totals.items()},
            'concentration_risk': max([float(v) for v in sector_totals.values()] + [0]),
            'diversification_score': self._calculate_diversification_score(sector_totals)
        }
        
        return validation_results
    
    def validate_transaction(
        self,
        transaction_type: str,
        symbol: Optional[str],
        quantity: Optional[Decimal],
        price: Optional[Decimal],
        amount: Decimal,
        timestamp: datetime,
        market_regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        """Validate financial transaction with business rules"""
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'adjusted_values': {}
        }
        
        # Validate transaction type
        valid_types = ['buy', 'sell', 'deposit', 'withdrawal', 'dividend', 'fee', 'split', 'merger']
        if transaction_type not in valid_types:
            validation_result['errors'].append(f"Invalid transaction type: {transaction_type}")
            validation_result['valid'] = False
            return validation_result
        
        # Trade-specific validations
        if transaction_type in ['buy', 'sell']:
            if not symbol:
                validation_result['errors'].append("Symbol required for buy/sell transactions")
                validation_result['valid'] = False
            else:
                try:
                    validated_symbol = self.validate_stock_symbol(symbol)
                    validation_result['adjusted_values']['symbol'] = validated_symbol
                except ValidationError as e:
                    validation_result['errors'].append(f"Invalid symbol: {str(e)}")
                    validation_result['valid'] = False
            
            if not quantity or quantity <= 0:
                validation_result['errors'].append("Positive quantity required for trades")
                validation_result['valid'] = False
            
            if not price or price <= 0:
                validation_result['errors'].append("Positive price required for trades")
                validation_result['valid'] = False
            
            # Market regime considerations
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                if quantity and price:
                    trade_value = quantity * price
                    if trade_value > amount * Decimal('1.1'):  # Allow 10% variance
                        validation_result['warnings'].append(
                            "Large trade during high volatility - consider staging"
                        )
        
        # Amount validations
        if amount <= 0:
            validation_result['errors'].append("Transaction amount must be positive")
            validation_result['valid'] = False
        
        # Check for reasonable amount limits
        if amount > Decimal('10000000'):  # $10M
            validation_result['warnings'].append("Large transaction amount - verify legitimacy")
        
        # Timestamp validations
        now = datetime.now()
        if timestamp > now + timedelta(minutes=5):
            validation_result['errors'].append("Transaction timestamp cannot be in future")
            validation_result['valid'] = False
        
        # Market hours validation for trades
        if transaction_type in ['buy', 'sell'] and symbol:
            if not self._is_market_hours(timestamp, self._get_exchange_for_symbol(symbol)):
                validation_result['warnings'].append("Trade outside normal market hours")
        
        return validation_result
    
    def validate_risk_parameters(
        self,
        risk_score: int,
        max_drawdown: Decimal,
        position_size_limit: Decimal,
        stop_loss_threshold: Decimal,
        user_age: Optional[int] = None,
        investment_horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate risk management parameters"""
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Risk score validation
        if not 1 <= risk_score <= 100:
            validation_result['errors'].append("Risk score must be between 1 and 100")
            validation_result['valid'] = False
        
        # Drawdown validation
        if not Decimal('0.01') <= max_drawdown <= Decimal('0.50'):
            validation_result['errors'].append("Max drawdown must be between 1% and 50%")
            validation_result['valid'] = False
        
        # Position size validation
        if not Decimal('0.01') <= position_size_limit <= Decimal('0.25'):
            validation_result['errors'].append("Position size limit must be between 1% and 25%")
            validation_result['valid'] = False
        
        # Stop loss validation
        if not Decimal('0.01') <= stop_loss_threshold <= Decimal('0.30'):
            validation_result['errors'].append("Stop loss threshold must be between 1% and 30%")
            validation_result['valid'] = False
        
        # Age-based recommendations
        if user_age:
            if user_age < 25:
                if risk_score < 60:
                    validation_result['recommendations'].append(
                        "Young investors can typically handle higher risk for long-term growth"
                    )
            elif user_age > 60:
                if risk_score > 70:
                    validation_result['warnings'].append(
                        "High risk tolerance unusual for pre-retirement age"
                    )
                if max_drawdown > Decimal('0.20'):
                    validation_result['warnings'].append(
                        "Consider lower drawdown limit approaching retirement"
                    )
        
        # Investment horizon considerations
        if investment_horizon:
            if investment_horizon < 2 and risk_score > 80:
                validation_result['warnings'].append(
                    "High risk inappropriate for short investment horizon"
                )
            elif investment_horizon > 10 and risk_score < 40:
                validation_result['recommendations'].append(
                    "Long investment horizon allows for more growth-oriented approach"
                )
        
        return validation_result
    
    def validate_market_data_quality(
        self,
        data_points: List[Dict[str, Any]],
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Validate market data quality and consistency"""
        
        validation_result = {
            'valid': True,
            'quality_score': 0.0,
            'issues': [],
            'statistics': {}
        }
        
        if not data_points:
            validation_result['valid'] = False
            validation_result['issues'].append("No data points provided")
            return validation_result
        
        # Sort by timestamp
        try:
            sorted_data = sorted(data_points, key=lambda x: datetime.fromisoformat(x['timestamp']))
        except (KeyError, ValueError) as e:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Invalid timestamp data: {str(e)}")
            return validation_result
        
        quality_points = 0
        total_points = 0
        price_gaps = []
        volume_anomalies = []
        
        for i, data_point in enumerate(sorted_data):
            try:
                # Validate individual data point
                ohlc = self.validate_price_data(
                    data_point['open'],
                    data_point['high'],
                    data_point['low'],
                    data_point['close'],
                    data_point.get('volume')
                )
                quality_points += 1
                
                # Check for price gaps between consecutive periods
                if i > 0:
                    prev_close = Decimal(str(sorted_data[i-1]['close']))
                    current_open = ohlc['open']
                    gap = abs(current_open - prev_close) / prev_close
                    
                    if gap > Decimal('0.1'):  # 10% gap
                        price_gaps.append({
                            'index': i,
                            'gap_percentage': float(gap),
                            'timestamp': data_point['timestamp']
                        })
                
                # Volume consistency checks
                if 'volume' in data_point:
                    volume = Decimal(str(data_point['volume']))
                    if i >= 5:  # Need some history
                        avg_volume = sum(
                            Decimal(str(sorted_data[j].get('volume', 0)))
                            for j in range(max(0, i-5), i)
                        ) / 5
                        
                        if avg_volume > 0:
                            volume_ratio = volume / avg_volume
                            if volume_ratio > 5 or volume_ratio < 0.2:
                                volume_anomalies.append({
                                    'index': i,
                                    'volume_ratio': float(volume_ratio),
                                    'timestamp': data_point['timestamp']
                                })
                
            except ValidationError:
                validation_result['issues'].append(f"Invalid OHLC data at index {i}")
            
            total_points += 1
        
        # Calculate quality score
        if total_points > 0:
            validation_result['quality_score'] = quality_points / total_points
        
        # Add quality issues
        if price_gaps:
            validation_result['issues'].append(f"Found {len(price_gaps)} significant price gaps")
        
        if volume_anomalies:
            validation_result['issues'].append(f"Found {len(volume_anomalies)} volume anomalies")
        
        # Data completeness check
        expected_periods = self._calculate_expected_periods(
            sorted_data[0]['timestamp'],
            sorted_data[-1]['timestamp'],
            timeframe
        )
        
        completeness = len(sorted_data) / expected_periods if expected_periods > 0 else 0
        
        validation_result['statistics'] = {
            'data_points': len(sorted_data),
            'quality_score': validation_result['quality_score'],
            'completeness': completeness,
            'price_gaps': len(price_gaps),
            'volume_anomalies': len(volume_anomalies),
            'time_span_hours': (
                datetime.fromisoformat(sorted_data[-1]['timestamp']) - 
                datetime.fromisoformat(sorted_data[0]['timestamp'])
            ).total_seconds() / 3600
        }
        
        # Overall validation
        if validation_result['quality_score'] < 0.8:
            validation_result['valid'] = False
            validation_result['issues'].append("Data quality below acceptable threshold")
        
        return validation_result
    
    def _calculate_diversification_score(self, sector_totals: Dict[str, Decimal]) -> float:
        """Calculate portfolio diversification score using Herfindahl index"""
        if not sector_totals:
            return 0.0
        
        # Herfindahl-Hirschman Index (lower is more diversified)
        hhi = sum(float(weight) ** 2 for weight in sector_totals.values())
        
        # Convert to diversification score (higher is better)
        max_hhi = 1.0  # Completely concentrated
        min_hhi = 1.0 / len(sector_totals)  # Perfectly diversified
        
        if max_hhi == min_hhi:
            return 1.0
        
        diversification_score = (max_hhi - hhi) / (max_hhi - min_hhi)
        return max(0.0, min(1.0, diversification_score))
    
    def _is_market_hours(self, timestamp: datetime, exchange: str) -> bool:
        """Check if timestamp falls within market hours"""
        if exchange not in self.market_hours:
            return True  # Unknown exchange, assume always open
        
        hours = self.market_hours[exchange]
        # Simplified check - in practice, would need proper timezone handling
        hour = timestamp.hour + timestamp.minute / 60.0
        
        return hours['open'] <= hour <= hours['close']
    
    def _get_exchange_for_symbol(self, symbol: str) -> str:
        """Determine exchange for a symbol (simplified logic)"""
        # This is a simplified implementation
        # In practice, would use a symbol-to-exchange mapping
        return 'NYSE'  # Default
    
    def _calculate_expected_periods(
        self,
        start_time: str,
        end_time: str,
        timeframe: str
    ) -> int:
        """Calculate expected number of periods for given timeframe"""
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration = end_dt - start_dt
        
        # Convert timeframe to minutes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }.get(timeframe, 60)
        
        return int(duration.total_seconds() / (timeframe_minutes * 60))


# Global instance
financial_validator = FinancialDataValidator()
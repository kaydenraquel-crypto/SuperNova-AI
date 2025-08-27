"""
Global test configuration and fixtures for comprehensive testing suite.
"""
import pytest
import asyncio
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis
import json
import pandas as pd
import numpy as np

# Import application components
from supernova.api import app
from supernova.db import Base, User, Profile, Asset, WatchlistItem
from supernova.schemas import OHLCVBar, SentimentDataPoint
from supernova.config import settings

# Test database configuration
TEST_DATABASE_URL = "sqlite:///./test_supernova_comprehensive.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=test_engine)
    
    # Override the SessionLocal in the api module
    import supernova.api
    original_session = supernova.api.SessionLocal
    supernova.api.SessionLocal = TestSessionLocal
    
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Restore original session and clean up
        supernova.api.SessionLocal = original_session
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db_session):
    """Create test client with database override."""
    return TestClient(app)


@pytest.fixture
def test_user(db_session):
    """Create test user."""
    user = User(name="Test User", email="test@example.com")
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture
def test_profile(db_session, test_user):
    """Create test user profile."""
    profile = Profile(
        user_id=test_user.id,
        risk_score=60,
        time_horizon_yrs=5,
        objectives="growth",
        constraints="no crypto"
    )
    db_session.add(profile)
    db_session.commit()
    return profile


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch('redis.Redis') as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_ohlcv_data():
    """Generate comprehensive sample OHLCV data."""
    np.random.seed(42)  # Reproducible data
    
    bars = []
    base_price = 100.0
    base_volume = 1000000
    
    for i in range(1000):  # Large dataset for comprehensive testing
        timestamp = (datetime.now() - timedelta(hours=1000-i)).isoformat() + "Z"
        
        # Add realistic price movements
        price_change = np.random.normal(0, 0.02) * base_price
        base_price = max(1.0, base_price + price_change)
        
        # Realistic OHLC relationships
        open_price = base_price
        high_price = open_price + abs(np.random.normal(0, 0.01) * open_price)
        low_price = open_price - abs(np.random.normal(0, 0.01) * open_price)
        close_price = low_price + (high_price - low_price) * np.random.random()
        
        # Volume with some correlation to price movement
        volume_multiplier = 1 + abs(price_change / base_price)
        volume = int(base_volume * volume_multiplier * (0.5 + np.random.random()))
        
        bars.append(OHLCVBar(
            timestamp=timestamp,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume
        ))
    
    return bars


@pytest.fixture
def sample_sentiment_data():
    """Generate sample sentiment data."""
    data_points = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(1000):
        timestamp = base_time + timedelta(hours=i)
        sentiment = np.random.normal(0, 0.3)  # Centered around neutral
        sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
        
        data_points.append(SentimentDataPoint(
            symbol="TEST",
            timestamp=timestamp.isoformat(),
            sentiment_score=round(sentiment, 3),
            confidence=np.random.uniform(0.1, 0.9),
            source="test_source",
            text_sample="Test sentiment text"
        ))
    
    return data_points


@pytest.fixture
def performance_test_data():
    """Generate large dataset for performance testing."""
    return {
        'large_ohlcv': [
            OHLCVBar(
                timestamp=(datetime.now() - timedelta(minutes=i)).isoformat() + "Z",
                open=100 + np.random.normal(0, 5),
                high=105 + np.random.normal(0, 5),
                low=95 + np.random.normal(0, 5),
                close=100 + np.random.normal(0, 5),
                volume=int(1000000 * np.random.uniform(0.5, 2.0))
            ) for i in range(10000)
        ],
        'symbols': [f"PERF_TEST_{i}" for i in range(100)],
        'user_profiles': [
            {
                'name': f"User_{i}",
                'email': f"user_{i}@test.com",
                'risk_questions': [np.random.randint(1, 5) for _ in range(5)]
            } for i in range(1000)
        ]
    }


@pytest.fixture
def security_test_payloads():
    """Generate security test payloads."""
    return {
        'sql_injection': [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM profiles--",
            "'; DELETE FROM profiles WHERE '1'='1'; --"
        ],
        'xss_payloads': [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
        ],
        'path_traversal': [
            "../../../etc/passwd",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
    }


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_llm_responses():
    """Mock LLM API responses."""
    responses = {
        'openai': {
            'chat_completion': {
                'choices': [{
                    'message': {
                        'content': 'Test response from OpenAI',
                        'role': 'assistant'
                    }
                }],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15
                }
            }
        },
        'anthropic': {
            'completion': 'Test response from Anthropic'
        }
    }
    return responses


@pytest.fixture
def mock_external_apis():
    """Mock external API calls."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'success', 'data': 'test'}
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        yield mock_client


@pytest.fixture
def websocket_test_client():
    """Create WebSocket test client."""
    with TestClient(app).websocket_connect("/ws") as websocket:
        yield websocket


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        'TESTING': 'true',
        'DATABASE_URL': TEST_DATABASE_URL,
        'REDIS_URL': 'redis://localhost:6379/1',
        'SECRET_KEY': 'test_secret_key_for_testing_only',
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'LOG_LEVEL': 'DEBUG'
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Cleanup
    for key in test_env:
        os.environ.pop(key, None)


@pytest.fixture
def benchmark_results_tracker():
    """Track benchmark results across tests."""
    results = {
        'api_response_times': [],
        'database_query_times': [],
        'calculation_times': [],
        'memory_usage': [],
        'cpu_usage': []
    }
    yield results


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=['sqlite', 'postgresql'])
def database_type(request):
    """Test with different database types."""
    return request.param


@pytest.fixture(params=['sync', 'async'])
def execution_mode(request):
    """Test with different execution modes."""
    return request.param


@pytest.fixture(params=[1, 10, 100, 1000])
def data_sizes(request):
    """Test with different data sizes."""
    return request.param


@pytest.fixture(params=['conservative', 'moderate', 'aggressive'])
def risk_profiles(request):
    """Test with different risk profiles."""
    risk_scores = {
        'conservative': 25,
        'moderate': 50,
        'aggressive': 85
    }
    return risk_scores[request.param]


# Helper functions for tests
def create_test_bars(count: int, symbol: str = "TEST", trending: bool = False) -> List[OHLCVBar]:
    """Helper to create test OHLCV bars."""
    bars = []
    base_price = 100.0
    
    for i in range(count):
        timestamp = (datetime.now() - timedelta(hours=count-i)).isoformat() + "Z"
        
        if trending:
            price_change = (i - count/2) * 0.1
        else:
            price_change = np.random.normal(0, 0.5)
        
        base_price += price_change
        base_price = max(1.0, base_price)
        
        bars.append(OHLCVBar(
            timestamp=timestamp,
            open=base_price,
            high=base_price + abs(np.random.normal(0, 1)),
            low=base_price - abs(np.random.normal(0, 1)),
            close=base_price + np.random.normal(0, 0.5),
            volume=int(10000 * (0.5 + np.random.random()))
        ))
    
    return bars


def assert_response_structure(response_data: dict, expected_keys: List[str]):
    """Helper to assert response structure."""
    for key in expected_keys:
        assert key in response_data, f"Missing key: {key}"


def assert_performance_metrics(execution_time: float, max_time: float, memory_usage: Optional[float] = None):
    """Helper to assert performance metrics."""
    assert execution_time <= max_time, f"Execution time {execution_time}s exceeded maximum {max_time}s"
    if memory_usage is not None:
        assert memory_usage <= 500 * 1024 * 1024, f"Memory usage {memory_usage} bytes exceeded 500MB"


# Financial-specific fixtures
@pytest.fixture
def financial_test_portfolios():
    """Generate test portfolios with various compositions."""
    portfolios = {
        'conservative': {
            'name': 'Conservative Portfolio',
            'cash_allocation': 0.20,
            'assets': [
                {'symbol': 'VTI', 'weight': 0.30, 'type': 'equity'},
                {'symbol': 'BND', 'weight': 0.40, 'type': 'bond'},
                {'symbol': 'VEA', 'weight': 0.10, 'type': 'international_equity'}
            ],
            'expected_return': 0.06,
            'expected_volatility': 0.10,
            'risk_score': 25
        },
        'moderate': {
            'name': 'Moderate Portfolio',
            'cash_allocation': 0.10,
            'assets': [
                {'symbol': 'VTI', 'weight': 0.50, 'type': 'equity'},
                {'symbol': 'BND', 'weight': 0.25, 'type': 'bond'},
                {'symbol': 'VEA', 'weight': 0.15, 'type': 'international_equity'}
            ],
            'expected_return': 0.08,
            'expected_volatility': 0.15,
            'risk_score': 50
        },
        'aggressive': {
            'name': 'Aggressive Portfolio',
            'cash_allocation': 0.05,
            'assets': [
                {'symbol': 'VTI', 'weight': 0.60, 'type': 'equity'},
                {'symbol': 'VEA', 'weight': 0.20, 'type': 'international_equity'},
                {'symbol': 'VWO', 'weight': 0.15, 'type': 'emerging_markets'}
            ],
            'expected_return': 0.10,
            'expected_volatility': 0.20,
            'risk_score': 85
        }
    }
    return portfolios


@pytest.fixture
def market_scenarios():
    """Generate various market scenario test data."""
    scenarios = {
        'bull_market': {
            'description': 'Strong upward trend',
            'duration_days': 252,  # 1 year
            'annual_return': 0.20,
            'volatility': 0.12,
            'max_drawdown': 0.05
        },
        'bear_market': {
            'description': 'Significant downward trend',
            'duration_days': 180,  # 6 months
            'annual_return': -0.30,
            'volatility': 0.25,
            'max_drawdown': 0.40
        },
        'sideways_market': {
            'description': 'Range-bound market',
            'duration_days': 365,
            'annual_return': 0.02,
            'volatility': 0.15,
            'max_drawdown': 0.15
        },
        'volatile_market': {
            'description': 'High volatility period',
            'duration_days': 90,
            'annual_return': 0.05,
            'volatility': 0.35,
            'max_drawdown': 0.25
        }
    }
    return scenarios


@pytest.fixture
def financial_calculations_test_data():
    """Financial calculation validation data."""
    return {
        'sharpe_ratio_tests': [
            {'returns': [0.01, 0.02, -0.01, 0.03, 0.01], 'risk_free_rate': 0.02, 'expected': 0.447},
            {'returns': [0.05, 0.03, 0.07, 0.02, 0.04], 'risk_free_rate': 0.01, 'expected': 1.789},
        ],
        'max_drawdown_tests': [
            {'prices': [100, 105, 110, 95, 100, 102], 'expected': 0.136},
            {'prices': [100, 120, 90, 110, 85], 'expected': 0.292},
        ],
        'volatility_tests': [
            {'returns': [0.01, -0.02, 0.03, -0.01, 0.02], 'expected': 0.0187},
            {'returns': [0.05, -0.03, 0.07, -0.02, 0.04], 'expected': 0.0424},
        ],
        'portfolio_return_tests': [
            {
                'weights': [0.6, 0.4],
                'returns': [0.10, 0.05],
                'expected': 0.08
            },
            {
                'weights': [0.3, 0.3, 0.4],
                'returns': [0.12, 0.08, 0.06],
                'expected': 0.084
            }
        ]
    }


@pytest.fixture
def backtesting_scenarios():
    """Backtesting scenario test data."""
    return {
        'simple_buy_hold': {
            'strategy': 'buy_and_hold',
            'initial_capital': 100000,
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'symbols': ['VTI'],
            'expected_metrics': {
                'total_return_min': 0.15,
                'sharpe_ratio_min': 0.8,
                'max_drawdown_max': 0.25
            }
        },
        'momentum_strategy': {
            'strategy': 'momentum',
            'initial_capital': 100000,
            'lookback_period': 20,
            'symbols': ['VTI', 'VEA', 'VWO'],
            'expected_metrics': {
                'total_return_min': 0.10,
                'sharpe_ratio_min': 0.6,
                'max_drawdown_max': 0.30
            }
        }
    }


@pytest.fixture
def compliance_test_scenarios():
    """Financial compliance test scenarios."""
    return {
        'fiduciary_tests': [
            {
                'client_profile': {'risk_tolerance': 'low', 'time_horizon': 5},
                'recommended_portfolio': {'equity_weight': 0.3, 'bond_weight': 0.7},
                'should_pass': True
            },
            {
                'client_profile': {'risk_tolerance': 'low', 'time_horizon': 2},
                'recommended_portfolio': {'equity_weight': 0.8, 'bond_weight': 0.2},
                'should_pass': False
            }
        ],
        'position_limits': [
            {
                'portfolio_value': 100000,
                'position_value': 5000,
                'max_position_percent': 0.10,
                'should_pass': True
            },
            {
                'portfolio_value': 100000,
                'position_value': 15000,
                'max_position_percent': 0.10,
                'should_pass': False
            }
        ]
    }


# Test data generators
class TestDataGenerator:
    """Generate various test data patterns."""
    
    @staticmethod
    def generate_trend_data(length: int, trend_direction: str = "up") -> List[float]:
        """Generate trending price data."""
        base = 100.0
        trend_factor = 0.1 if trend_direction == "up" else -0.1
        
        prices = []
        for i in range(length):
            base += trend_factor + np.random.normal(0, 0.05)
            prices.append(max(1.0, base))
        
        return prices
    
    @staticmethod
    def generate_volatile_data(length: int, volatility: float = 0.02) -> List[float]:
        """Generate volatile price data."""
        base = 100.0
        prices = []
        
        for i in range(length):
            change = np.random.normal(0, volatility * base)
            base = max(1.0, base + change)
            prices.append(base)
        
        return prices
    
    @staticmethod
    def generate_sideways_data(length: int, range_pct: float = 0.05) -> List[float]:
        """Generate sideways/ranging price data."""
        base = 100.0
        range_value = base * range_pct
        
        prices = []
        for i in range(length):
            price = base + np.random.uniform(-range_value, range_value)
            prices.append(max(1.0, price))
        
        return prices
    
    @staticmethod
    def generate_market_crash_scenario(length: int, crash_day: int, severity: float = 0.3) -> List[float]:
        """Generate market crash scenario."""
        prices = []
        base = 100.0
        
        for i in range(length):
            if i == crash_day:
                base *= (1 - severity)  # Crash
            elif i > crash_day and i < crash_day + 20:
                base *= (1 + np.random.uniform(0.01, 0.03))  # Recovery
            else:
                base += np.random.normal(0, 0.01) * base
            
            prices.append(max(1.0, base))
        
        return prices
    
    @staticmethod
    def generate_earnings_event_data(base_price: float, event_day: int, impact: float) -> List[Dict]:
        """Generate price data around earnings events."""
        data = []
        for i in range(30):  # 30 days around event
            if i == event_day:
                price_change = impact
            else:
                price_change = np.random.normal(0, 0.02)
            
            price = base_price * (1 + price_change)
            volume_multiplier = 3.0 if abs(i - event_day) <= 1 else 1.0
            
            data.append({
                'day': i,
                'price': price,
                'volume': int(1000000 * volume_multiplier * (0.5 + np.random.random())),
                'is_earnings_day': i == event_day
            })
        
        return data
    
    @staticmethod
    def generate_correlation_matrix_data(symbols: List[str], correlations: Dict[str, Dict[str, float]]):
        """Generate correlated returns for multiple assets."""
        n_days = 252
        returns_data = {}
        
        # Generate base random returns
        base_returns = np.random.multivariate_normal(
            mean=[0] * len(symbols),
            cov=[[correlations.get(s1, {}).get(s2, 0.0) if s1 != s2 else 0.02**2 
                  for s2 in symbols] for s1 in symbols],
            size=n_days
        )
        
        for i, symbol in enumerate(symbols):
            returns_data[symbol] = base_returns[:, i].tolist()
        
        return returns_data


# Financial calculation helpers
class FinancialTestHelpers:
    """Helper functions for financial calculations in tests."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for testing."""
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> float:
        """Calculate maximum drawdown for testing."""
        prices_array = np.array(prices)
        peak = np.maximum.accumulate(prices_array)
        drawdown = (prices_array - peak) / peak
        return abs(np.min(drawdown))
    
    @staticmethod
    def calculate_portfolio_metrics(weights: List[float], returns: List[List[float]]) -> Dict:
        """Calculate portfolio metrics for testing."""
        weights_array = np.array(weights)
        returns_array = np.array(returns).T
        
        portfolio_returns = np.dot(returns_array, weights_array)
        
        return {
            'total_return': np.prod(1 + portfolio_returns) - 1,
            'annualized_return': np.mean(portfolio_returns) * 252,
            'volatility': np.std(portfolio_returns) * np.sqrt(252),
            'sharpe_ratio': FinancialTestHelpers.calculate_sharpe_ratio(portfolio_returns.tolist())
        }
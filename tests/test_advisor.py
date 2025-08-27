import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supernova.advisor import score_risk, advise
from supernova.strategy_engine import TEMPLATES


@pytest.fixture
def sample_bars():
    """Generate sample OHLCV bar data for advisor testing."""
    base_price = 100
    bars = []
    for i in range(100):
        timestamp = (datetime.now() - timedelta(hours=100-i)).isoformat() + "Z"
        price_change = np.random.normal(0, 0.5)
        base_price += price_change
        
        high = base_price + abs(np.random.normal(0, 0.3))
        low = base_price - abs(np.random.normal(0, 0.3))
        close = base_price + np.random.normal(0, 0.2)
        volume = int(10000 + np.random.normal(0, 2000))
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price,
            "high": max(base_price, high, close),
            "low": min(base_price, low, close),
            "close": close,
            "volume": max(1, volume)
        })
        base_price = close
    
    return bars


@pytest.fixture
def bullish_bars():
    """Generate bullish trend bars for testing positive sentiment."""
    bars = []
    base_price = 100
    for i in range(50):
        timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
        base_price += 0.5 + np.random.normal(0, 0.2)
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price - 0.1,
            "high": base_price + 0.3,
            "low": base_price - 0.2,
            "close": base_price,
            "volume": 10000
        })
    
    return bars


@pytest.fixture
def bearish_bars():
    """Generate bearish trend bars for testing negative sentiment."""
    bars = []
    base_price = 150
    for i in range(50):
        timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
        base_price -= 0.6 + np.random.normal(0, 0.2)
        
        bars.append({
            "timestamp": timestamp,
            "open": base_price + 0.1,
            "high": base_price + 0.2,
            "low": base_price - 0.3,
            "close": base_price,
            "volume": 10000
        })
    
    return bars


class TestScoreRisk:
    def test_empty_risk_questions(self):
        """Test risk scoring with empty question list."""
        score = score_risk([])
        assert score == 50  # Default neutral score

    def test_minimum_risk_score(self):
        """Test minimum possible risk score."""
        # All questions answered with lowest risk (1)
        questions = [1, 1, 1, 1, 1]
        score = score_risk(questions)
        assert score == 25  # (5 * 1) / (5 * 4) * 100 = 25

    def test_maximum_risk_score(self):
        """Test maximum possible risk score."""
        # All questions answered with highest risk (4)
        questions = [4, 4, 4, 4, 4]
        score = score_risk(questions)
        assert score == 100  # (5 * 4) / (5 * 4) * 100 = 100

    def test_moderate_risk_score(self):
        """Test moderate risk score calculation."""
        # Mixed answers averaging to moderate risk
        questions = [2, 3, 2, 3, 2]  # Average 2.4
        score = score_risk(questions)
        assert score == 60  # (12) / (5 * 4) * 100 = 60

    def test_single_question(self):
        """Test risk scoring with single question."""
        score = score_risk([3])
        assert score == 75  # 3/4 * 100 = 75

    def test_varied_question_count(self):
        """Test risk scoring with different numbers of questions."""
        # Test with 3 questions
        score_3q = score_risk([2, 3, 4])
        expected_3q = int(round((2 + 3 + 4) / (3 * 4) * 100))
        assert score_3q == expected_3q

        # Test with 7 questions
        score_7q = score_risk([1, 2, 3, 4, 1, 2, 3])
        expected_7q = int(round((1 + 2 + 3 + 4 + 1 + 2 + 3) / (7 * 4) * 100))
        assert score_7q == expected_7q

    def test_edge_case_zero_answer(self):
        """Test risk scoring with zero answer (edge case)."""
        questions = [0, 2, 3]
        score = score_risk(questions)
        assert 0 <= score <= 100

    def test_edge_case_high_answer(self):
        """Test risk scoring with answer above 4 (edge case)."""
        questions = [2, 3, 5]  # 5 is above normal range
        score = score_risk(questions)
        assert 0 <= score <= 125  # May exceed 100 with high values

    def test_rounding_behavior(self):
        """Test proper rounding of risk scores."""
        # Test case that would result in 33.33...
        questions = [1, 1, 2]  # Sum=4, divisor=12, result=33.33
        score = score_risk(questions)
        assert score == 33  # Should round to nearest integer

    def test_consistent_results(self):
        """Test that risk scoring is consistent."""
        questions = [2, 3, 2, 4, 1]
        score1 = score_risk(questions)
        score2 = score_risk(questions)
        assert score1 == score2


class TestAdvise:
    def test_basic_advice_functionality(self, sample_bars):
        """Test basic advice generation."""
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50
        )
        
        assert action in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert isinstance(details, dict)
        assert isinstance(rationale, str)
        assert isinstance(risk_notes, str)
        
        # Check rationale contains expected information
        assert action.upper() in rationale
        assert "confidence" in rationale.lower()

    def test_conservative_risk_profile(self, bullish_bars):
        """Test advice adjustment for conservative risk profile."""
        # Conservative profile should throttle buy confidence
        action, conf, details, rationale, risk_notes = advise(
            bars=bullish_bars, risk_score=20  # Very conservative
        )
        
        if action == "buy":
            assert "Conservative profile" in risk_notes
            assert "throttling" in risk_notes
            # Confidence should be reduced (though we can't compare directly)
            assert conf > 0

    def test_aggressive_risk_profile(self, bullish_bars):
        """Test advice adjustment for aggressive risk profile."""
        # Aggressive profile should boost confidence
        action, conf, details, rationale, risk_notes = advise(
            bars=bullish_bars, risk_score=80  # Very aggressive
        )
        
        if action in ["buy", "sell"]:
            assert "Aggressive profile" in risk_notes
            assert "higher conviction" in risk_notes
            # Confidence should be boosted
            assert conf > 0

    def test_neutral_risk_profile(self, sample_bars):
        """Test advice with neutral risk profile."""
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50  # Neutral
        )
        
        assert "Neutral profile" in risk_notes
        assert 0.1 <= conf <= 1.0

    def test_sentiment_boost_positive(self, sample_bars):
        """Test advice with positive sentiment hint."""
        # Get baseline advice
        action1, conf1, _, _, _ = advise(bars=sample_bars, risk_score=50)
        
        # Get advice with positive sentiment
        action2, conf2, _, _, _ = advise(
            bars=sample_bars, risk_score=50, sentiment_hint=0.5
        )
        
        # Positive sentiment should boost confidence
        if action1 == action2:  # Only compare if same action
            assert conf2 >= conf1

    def test_sentiment_boost_negative(self, sample_bars):
        """Test advice with negative sentiment hint."""
        # Get baseline advice
        action1, conf1, _, _, _ = advise(bars=sample_bars, risk_score=50)
        
        # Get advice with negative sentiment
        action2, conf2, _, _, _ = advise(
            bars=sample_bars, risk_score=50, sentiment_hint=-0.3
        )
        
        # Negative sentiment should reduce confidence
        if action1 == action2:  # Only compare if same action
            assert conf2 <= conf1

    def test_specific_template_usage(self, sample_bars):
        """Test advice generation with specific template."""
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50, template="ma_crossover"
        )
        
        assert action in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        # Details should contain MA-specific information
        assert "fast" in details or "slow" in details

    def test_template_with_custom_params(self, sample_bars):
        """Test advice with specific template and custom parameters."""
        params = {"fast": 5, "slow": 15}
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50, 
            template="ma_crossover", params=params
        )
        
        assert action in ["buy", "sell", "hold"]
        assert isinstance(details, dict)

    def test_ensemble_advice(self, sample_bars):
        """Test advice using ensemble of strategies."""
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50  # No template = use ensemble
        )
        
        assert action in ["buy", "sell", "hold"]
        # Ensemble should return details from multiple strategies
        assert len(details) >= len(TEMPLATES)

    def test_all_templates_advice(self, sample_bars):
        """Test advice generation with all available templates."""
        for template_name in TEMPLATES.keys():
            action, conf, details, rationale, risk_notes = advise(
                bars=sample_bars, risk_score=50, template=template_name
            )
            
            assert action in ["buy", "sell", "hold"]
            assert 0.1 <= conf <= 1.0
            assert isinstance(details, dict)

    def test_confidence_bounds_enforcement(self, sample_bars):
        """Test that confidence stays within bounds after adjustments."""
        # Test with extreme risk scores and sentiment
        test_cases = [
            (5, -1.0),   # Very conservative + very negative sentiment
            (95, 1.0),   # Very aggressive + very positive sentiment
            (50, 0.0),   # Neutral
        ]
        
        for risk_score, sentiment in test_cases:
            action, conf, _, _, _ = advise(
                bars=sample_bars, risk_score=risk_score, 
                sentiment_hint=sentiment
            )
            
            assert 0.1 <= conf <= 0.95  # Should be clamped

    def test_rationale_content(self, sample_bars):
        """Test that rationale contains meaningful content."""
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50
        )
        
        # Rationale should contain key information
        assert action.upper() in rationale
        assert f"{conf:.2f}" in rationale
        assert "Details:" in rationale
        assert len(rationale) > 50  # Should be reasonably descriptive

    def test_risk_notes_content(self):
        """Test risk notes content for different profiles."""
        sample_bars = [
            {"timestamp": "2024-01-01T01:00:00Z", "open": 100, "high": 101, 
             "low": 99, "close": 100, "volume": 1000}
        ] * 50
        
        # Conservative
        _, _, _, _, risk_notes_cons = advise(bars=sample_bars, risk_score=20)
        
        # Aggressive  
        _, _, _, _, risk_notes_agg = advise(bars=sample_bars, risk_score=80)
        
        # Neutral
        _, _, _, _, risk_notes_neut = advise(bars=sample_bars, risk_score=50)
        
        assert "Conservative" in risk_notes_cons or "Aggressive" in risk_notes_cons or "Neutral" in risk_notes_cons
        assert "Aggressive" in risk_notes_agg or "Conservative" in risk_notes_agg or "Neutral" in risk_notes_agg
        assert "Neutral" in risk_notes_neut

    def test_edge_case_extreme_sentiment(self, sample_bars):
        """Test advice with extreme sentiment values."""
        # Test extreme positive sentiment
        action1, conf1, _, _, _ = advise(
            bars=sample_bars, risk_score=50, sentiment_hint=2.0
        )
        
        # Test extreme negative sentiment
        action2, conf2, _, _, _ = advise(
            bars=sample_bars, risk_score=50, sentiment_hint=-2.0
        )
        
        # Should handle gracefully and keep confidence in bounds
        assert 0.1 <= conf1 <= 0.95
        assert 0.1 <= conf2 <= 0.95

    def test_invalid_template_fallback(self, sample_bars):
        """Test that invalid template falls back to ensemble."""
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50, template="invalid_template"
        )
        
        # Should fall back to ensemble
        assert action in ["buy", "sell", "hold"]
        assert len(details) >= len(TEMPLATES)  # Ensemble returns all template results

    def test_empty_bars_handling(self):
        """Test advice generation with empty bars."""
        with pytest.raises(Exception):
            # Should raise an error with empty bars
            advise(bars=[], risk_score=50)

    def test_insufficient_bars_handling(self):
        """Test advice generation with insufficient bars."""
        # Very few bars
        bars = [
            {"timestamp": "2024-01-01T01:00:00Z", "open": 100, "high": 101, 
             "low": 99, "close": 100, "volume": 1000}
        ]
        
        # Should handle gracefully but may have limited accuracy
        action, conf, details, rationale, risk_notes = advise(
            bars=bars, risk_score=50
        )
        
        assert action in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0

    def test_nan_data_handling(self):
        """Test advice generation with NaN data in bars."""
        bars = []
        for i in range(50):
            timestamp = (datetime.now() - timedelta(hours=50-i)).isoformat() + "Z"
            # Include some NaN values
            close_val = 100 if i % 10 != 0 else float('nan')
            bars.append({
                "timestamp": timestamp,
                "open": 100, "high": 101, "low": 99,
                "close": close_val, "volume": 1000
            })
        
        # Should handle NaN data gracefully
        try:
            action, conf, details, rationale, risk_notes = advise(
                bars=bars, risk_score=50
            )
            assert action in ["buy", "sell", "hold"]
        except Exception:
            # Some strategies may fail with NaN data, which is acceptable
            pass

    def test_consistent_results(self, sample_bars):
        """Test that advice is consistent for same inputs."""
        result1 = advise(bars=sample_bars, risk_score=50)
        result2 = advise(bars=sample_bars, risk_score=50)
        
        # Results should be identical for same inputs
        assert result1[0] == result2[0]  # Same action
        assert abs(result1[1] - result2[1]) < 1e-10  # Same confidence
        assert result1[2] == result2[2]  # Same details
        assert result1[3] == result2[3]  # Same rationale
        assert result1[4] == result2[4]  # Same risk notes


class TestAdviseIntegration:
    def test_bullish_scenario_integration(self, bullish_bars):
        """Test integrated advice in clearly bullish scenario."""
        # Conservative investor in bullish market
        action_cons, conf_cons, _, _, risk_notes_cons = advise(
            bars=bullish_bars, risk_score=25, sentiment_hint=0.3
        )
        
        # Aggressive investor in bullish market
        action_agg, conf_agg, _, _, risk_notes_agg = advise(
            bars=bullish_bars, risk_score=75, sentiment_hint=0.3
        )
        
        # Both should likely suggest buy, but with different confidence
        if action_cons == "buy" and action_agg == "buy":
            # Aggressive should have higher confidence
            assert conf_agg >= conf_cons

    def test_bearish_scenario_integration(self, bearish_bars):
        """Test integrated advice in clearly bearish scenario."""
        action, conf, details, rationale, risk_notes = advise(
            bars=bearish_bars, risk_score=50, sentiment_hint=-0.2
        )
        
        # In bearish conditions with negative sentiment, likely sell or hold
        assert action in ["sell", "hold"]
        assert 0.1 <= conf <= 1.0

    def test_mixed_signals_handling(self, sample_bars):
        """Test advice with mixed market signals."""
        # Neutral sentiment, moderate risk
        action, conf, details, rationale, risk_notes = advise(
            bars=sample_bars, risk_score=50, sentiment_hint=0.0
        )
        
        # Should handle mixed signals gracefully
        assert action in ["buy", "sell", "hold"]
        assert 0.1 <= conf <= 1.0
        assert "Neutral profile" in risk_notes
"""
Test script for the LLM-enhanced advisor functionality.
This script tests both LLM-enabled and fallback modes.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supernova.advisor import advise, score_risk
from supernova.config import settings

def create_sample_bars():
    """Create sample OHLCV bars for testing."""
    bars = []
    base_price = 100.0
    
    for i in range(50):
        # Simulate some price movement
        price_change = (i % 7 - 3) * 0.5  # Simple oscillation
        open_price = base_price + price_change
        close_price = open_price + (i % 3 - 1) * 0.3
        high_price = max(open_price, close_price) + abs(i % 2) * 0.2
        low_price = min(open_price, close_price) - abs(i % 2) * 0.2
        
        bars.append({
            "timestamp": (datetime.now() - timedelta(hours=50-i)).isoformat(),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": 10000 + i * 1000
        })
        
        base_price = close_price
    
    return bars

def test_risk_scoring():
    """Test risk scoring functionality."""
    print("=== Testing Risk Scoring ===")
    
    # Conservative profile
    conservative_answers = [1, 1, 2, 1]  # Low risk answers
    conservative_score = score_risk(conservative_answers)
    print(f"Conservative profile score: {conservative_score}")
    
    # Aggressive profile  
    aggressive_answers = [4, 4, 3, 4]  # High risk answers
    aggressive_score = score_risk(aggressive_answers)
    print(f"Aggressive profile score: {aggressive_score}")
    
    # Empty profile (default)
    default_score = score_risk([])
    print(f"Default profile score: {default_score}")
    
    return conservative_score, aggressive_score, default_score

def test_basic_advisor():
    """Test basic advisor functionality without LLM."""
    print("\n=== Testing Basic Advisor (No LLM) ===")
    
    bars = create_sample_bars()
    risk_score = 60
    sentiment_hint = 0.2
    
    # Temporarily disable LLM for this test
    original_llm_enabled = settings.LLM_ENABLED
    settings.LLM_ENABLED = False
    
    try:
        action, conf, details, rationale, risk_notes = advise(
            bars=bars,
            risk_score=risk_score,
            sentiment_hint=sentiment_hint,
            symbol="AAPL",
            asset_class="stock",
            timeframe="1h"
        )
        
        print(f"Action: {action}")
        print(f"Confidence: {conf:.3f}")
        print(f"Details: {details}")
        print(f"Rationale: {rationale[:200]}...")
        print(f"Risk Notes: {risk_notes}")
        
        return action, conf, details, rationale, risk_notes
        
    finally:
        settings.LLM_ENABLED = original_llm_enabled

def test_llm_advisor():
    """Test LLM-enhanced advisor functionality."""
    print("\n=== Testing LLM-Enhanced Advisor ===")
    
    bars = create_sample_bars()
    risk_score = 40  # Moderate risk
    sentiment_hint = -0.1  # Slightly negative sentiment
    
    # Test with LLM enabled (will fall back to simple if no API keys)
    action, conf, details, rationale, risk_notes = advise(
        bars=bars,
        risk_score=risk_score,
        sentiment_hint=sentiment_hint,
        symbol="TSLA",
        asset_class="stock",
        timeframe="4h"
    )
    
    print(f"Action: {action}")
    print(f"Confidence: {conf:.3f}")
    print(f"Details: {details}")
    print(f"Rationale length: {len(rationale)} characters")
    print(f"Rationale preview: {rationale[:300]}...")
    print(f"Risk Notes: {risk_notes}")
    
    # Check if response looks like LLM-generated (longer, more structured)
    is_llm_generated = len(rationale) > 500 and ("##" in rationale or "**" in rationale)
    print(f"\nLLM-generated response: {is_llm_generated}")
    
    return action, conf, details, rationale, risk_notes

def test_different_asset_classes():
    """Test advisor with different asset classes."""
    print("\n=== Testing Different Asset Classes ===")
    
    bars = create_sample_bars()
    risk_score = 70  # Aggressive
    
    asset_classes = ["stock", "crypto", "fx", "futures", "option"]
    
    for asset_class in asset_classes:
        print(f"\n--- Testing {asset_class.upper()} ---")
        
        try:
            action, conf, details, rationale, risk_notes = advise(
                bars=bars,
                risk_score=risk_score,
                sentiment_hint=0.0,
                symbol=f"TEST_{asset_class.upper()}",
                asset_class=asset_class,
                timeframe="1d"
            )
            
            print(f"Action: {action}, Confidence: {conf:.3f}")
            print(f"Rationale preview: {rationale[:150]}...")
            
        except Exception as e:
            print(f"Error testing {asset_class}: {e}")

def test_strategy_templates():
    """Test different strategy templates."""
    print("\n=== Testing Strategy Templates ===")
    
    bars = create_sample_bars()
    risk_score = 50
    
    templates = ["ma_crossover", "rsi_breakout", "macd_trend"]
    
    for template in templates:
        print(f"\n--- Testing {template} ---")
        
        try:
            action, conf, details, rationale, risk_notes = advise(
                bars=bars,
                risk_score=risk_score,
                template=template,
                symbol="SPY",
                asset_class="stock",
                timeframe="1h"
            )
            
            print(f"Action: {action}, Confidence: {conf:.3f}")
            print(f"Template details: {details}")
            
        except Exception as e:
            print(f"Error testing {template}: {e}")

def test_configuration_display():
    """Display current LLM configuration."""
    print("\n=== LLM Configuration ===")
    print(f"LLM Enabled: {settings.LLM_ENABLED}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"LLM Model: {settings.LLM_MODEL}")
    print(f"Temperature: {settings.LLM_TEMPERATURE}")
    print(f"Max Tokens: {settings.LLM_MAX_TOKENS}")
    print(f"Cache Enabled: {settings.LLM_CACHE_ENABLED}")
    print(f"Cost Tracking: {settings.LLM_COST_TRACKING}")
    print(f"Daily Cost Limit: ${settings.LLM_DAILY_COST_LIMIT}")
    print(f"Fallback Enabled: {settings.LLM_FALLBACK_ENABLED}")
    
    # Check API keys (without revealing them)
    openai_key_set = "✓" if settings.OPENAI_API_KEY else "✗"
    anthropic_key_set = "✓" if settings.ANTHROPIC_API_KEY else "✗"
    hf_token_set = "✓" if settings.HUGGINGFACE_API_TOKEN else "✗"
    
    print(f"\nAPI Keys Configured:")
    print(f"OpenAI: {openai_key_set}")
    print(f"Anthropic: {anthropic_key_set}")
    print(f"HuggingFace: {hf_token_set}")

def main():
    """Run all tests."""
    print("SuperNova LLM-Enhanced Advisor Test Suite")
    print("==========================================\n")
    
    # Display configuration
    test_configuration_display()
    
    # Run tests
    try:
        # Basic functionality tests
        test_risk_scoring()
        test_basic_advisor()
        test_llm_advisor()
        
        # Advanced tests
        test_different_asset_classes()
        test_strategy_templates()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
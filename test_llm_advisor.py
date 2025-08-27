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
    print("\\n=== Testing Basic Advisor (No LLM) ===")
    
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
        
        return action, conf, details, rationale, risk_notes\n        \n    finally:\n        settings.LLM_ENABLED = original_llm_enabled\n\ndef test_llm_advisor():\n    \"\"\"Test LLM-enhanced advisor functionality.\"\"\"\n    print(\"\\n=== Testing LLM-Enhanced Advisor ===\")\n    \n    bars = create_sample_bars()\n    risk_score = 40  # Moderate risk\n    sentiment_hint = -0.1  # Slightly negative sentiment\n    \n    # Test with LLM enabled (will fall back to simple if no API keys)\n    action, conf, details, rationale, risk_notes = advise(\n        bars=bars,\n        risk_score=risk_score,\n        sentiment_hint=sentiment_hint,\n        symbol=\"TSLA\",\n        asset_class=\"stock\",\n        timeframe=\"4h\"\n    )\n    \n    print(f\"Action: {action}\")\n    print(f\"Confidence: {conf:.3f}\")\n    print(f\"Details: {details}\")\n    print(f\"Rationale length: {len(rationale)} characters\")\n    print(f\"Rationale preview: {rationale[:300]}...\")\n    print(f\"Risk Notes: {risk_notes}\")\n    \n    # Check if response looks like LLM-generated (longer, more structured)\n    is_llm_generated = len(rationale) > 500 and (\"##\" in rationale or \"**\" in rationale)\n    print(f\"\\nLLM-generated response: {is_llm_generated}\")\n    \n    return action, conf, details, rationale, risk_notes\n\ndef test_different_asset_classes():\n    \"\"\"Test advisor with different asset classes.\"\"\"\n    print(\"\\n=== Testing Different Asset Classes ===\")\n    \n    bars = create_sample_bars()\n    risk_score = 70  # Aggressive\n    \n    asset_classes = [\"stock\", \"crypto\", \"fx\", \"futures\", \"option\"]\n    \n    for asset_class in asset_classes:\n        print(f\"\\n--- Testing {asset_class.upper()} ---\")\n        \n        try:\n            action, conf, details, rationale, risk_notes = advise(\n                bars=bars,\n                risk_score=risk_score,\n                sentiment_hint=0.0,\n                symbol=f\"TEST_{asset_class.upper()}\",\n                asset_class=asset_class,\n                timeframe=\"1d\"\n            )\n            \n            print(f\"Action: {action}, Confidence: {conf:.3f}\")\n            print(f\"Rationale preview: {rationale[:150]}...\")\n            \n        except Exception as e:\n            print(f\"Error testing {asset_class}: {e}\")\n\ndef test_strategy_templates():\n    \"\"\"Test different strategy templates.\"\"\"\n    print(\"\\n=== Testing Strategy Templates ===\")\n    \n    bars = create_sample_bars()\n    risk_score = 50\n    \n    templates = [\"ma_crossover\", \"rsi_breakout\", \"macd_trend\"]\n    \n    for template in templates:\n        print(f\"\\n--- Testing {template} ---\")\n        \n        try:\n            action, conf, details, rationale, risk_notes = advise(\n                bars=bars,\n                risk_score=risk_score,\n                template=template,\n                symbol=\"SPY\",\n                asset_class=\"stock\",\n                timeframe=\"1h\"\n            )\n            \n            print(f\"Action: {action}, Confidence: {conf:.3f}\")\n            print(f\"Template details: {details}\")\n            \n        except Exception as e:\n            print(f\"Error testing {template}: {e}\")\n\ndef test_configuration_display():\n    \"\"\"Display current LLM configuration.\"\"\"\n    print(\"\\n=== LLM Configuration ===\")\n    print(f\"LLM Enabled: {settings.LLM_ENABLED}\")\n    print(f\"LLM Provider: {settings.LLM_PROVIDER}\")\n    print(f\"LLM Model: {settings.LLM_MODEL}\")\n    print(f\"Temperature: {settings.LLM_TEMPERATURE}\")\n    print(f\"Max Tokens: {settings.LLM_MAX_TOKENS}\")\n    print(f\"Cache Enabled: {settings.LLM_CACHE_ENABLED}\")\n    print(f\"Cost Tracking: {settings.LLM_COST_TRACKING}\")\n    print(f\"Daily Cost Limit: ${settings.LLM_DAILY_COST_LIMIT}\")\n    print(f\"Fallback Enabled: {settings.LLM_FALLBACK_ENABLED}\")\n    \n    # Check API keys (without revealing them)\n    openai_key_set = \"✓\" if settings.OPENAI_API_KEY else \"✗\"\n    anthropic_key_set = \"✓\" if settings.ANTHROPIC_API_KEY else \"✗\"\n    hf_token_set = \"✓\" if settings.HUGGINGFACE_API_TOKEN else \"✗\"\n    \n    print(f\"\\nAPI Keys Configured:\")\n    print(f\"OpenAI: {openai_key_set}\")\n    print(f\"Anthropic: {anthropic_key_set}\")\n    print(f\"HuggingFace: {hf_token_set}\")\n\ndef main():\n    \"\"\"Run all tests.\"\"\"\n    print(\"SuperNova LLM-Enhanced Advisor Test Suite\")\n    print(\"==========================================\\n\")\n    \n    # Display configuration\n    test_configuration_display()\n    \n    # Run tests\n    try:\n        # Basic functionality tests\n        test_risk_scoring()\n        test_basic_advisor()\n        test_llm_advisor()\n        \n        # Advanced tests\n        test_different_asset_classes()\n        test_strategy_templates()\n        \n        print(\"\\n=== All Tests Completed Successfully! ===\")\n        \n    except Exception as e:\n        print(f\"\\n❌ Test failed with error: {e}\")\n        import traceback\n        traceback.print_exc()\n        return 1\n    \n    return 0\n\nif __name__ == \"__main__\":\n    exit_code = main()\n    sys.exit(exit_code)"
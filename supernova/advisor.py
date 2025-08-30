from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import asyncio
import time
import hashlib
import json
import logging
from functools import lru_cache
from datetime import datetime, timedelta

# Core imports
from .strategy_engine import make_df, ensemble, TEMPLATES
from .config import settings
from .prompts import (
    get_prompt_for_action, format_technical_indicators, 
    format_risk_factors, RISK_DISCLAIMER_TEMPLATE
)

# New LLM service imports
try:
    from .llm_service import get_llm_service, LLMRequest, ResponseQuality, TaskComplexity, RequestPriority
    from langchain_core.messages import SystemMessage, HumanMessage
    LLM_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LLM service not available: {e}. Using simple rationale generation.")
    LLM_SERVICE_AVAILABLE = False

# Legacy LangChain imports for fallback
try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    # Provider-specific imports
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI as LLMModel
    elif settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic as LLMModel
    elif settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM as LLMModel
    elif settings.LLM_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFacePipeline as LLMModel
    else:
        from langchain_openai import ChatOpenAI as LLMModel  # Default fallback
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}. Using fallback rationale generation.")
    LANGCHAIN_AVAILABLE = False
    BaseLanguageModel = None
    LLMModel = None

# Setup logging
logger = logging.getLogger(__name__)

def score_risk(risk_questions: list[int]) -> int:
    """Score risk profile from questionnaire responses (0-4 scale)."""
    if not risk_questions:
        return 50
    raw = sum(risk_questions) / (len(risk_questions)*4)
    return int(round(raw * 100))

def _get_risk_profile_name(risk_score: int) -> str:
    """Convert numeric risk score to descriptive profile name."""
    if risk_score <= 25:
        return "conservative"
    elif risk_score >= 75:
        return "aggressive"
    else:
        return "moderate"

def _estimate_token_cost(text: str, model: str) -> float:
    """Estimate cost based on text length and model pricing."""
    # Rough token estimation (4 characters per token average)
    tokens = len(text) / 4
    
    # Model pricing (approximate, per 1K tokens)
    pricing = {
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.002,
        "claude-3-opus": 0.075,
        "claude-3-sonnet": 0.015,
        "claude-3-haiku": 0.0025,
        "ollama": 0.0,  # Local models
        "huggingface": 0.0  # Free tier
    }
    
    base_price = pricing.get(model.lower(), 0.01)  # Default pricing
    return (tokens / 1000) * base_price

def _update_cost_tracker(cost: float) -> bool:
    """Update daily cost tracker and check if within limits."""
    global _cost_tracker
    
    # Reset daily cost if new day
    today = datetime.now().date()
    if _cost_tracker["last_reset"] != today:
        _cost_tracker["daily_cost"] = 0.0
        _cost_tracker["last_reset"] = today
    
    # Check if adding this cost would exceed limit
    if settings.LLM_COST_TRACKING:
        if _cost_tracker["daily_cost"] + cost > settings.LLM_DAILY_COST_LIMIT:
            logger.warning(f"Daily LLM cost limit reached: ${_cost_tracker['daily_cost']:.2f}")
            return False
    
    _cost_tracker["daily_cost"] += cost
    return True

@lru_cache(maxsize=128)
def _get_cache_key(bars_hash: str, risk_score: int, sentiment: float, template: str, params_hash: str) -> str:
    """Generate cache key for LLM responses."""
    return hashlib.md5(
        f"{bars_hash}_{risk_score}_{sentiment}_{template}_{params_hash}".encode()
    ).hexdigest()

# Simple in-memory cache with size limit (use Redis in production)
from collections import OrderedDict
_llm_cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()  # LRU cache with size limit
MAX_CACHE_SIZE = 1000  # Limit cache to prevent memory issues

def _get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached LLM response if still valid."""
    if not settings.LLM_CACHE_ENABLED:
        return None
        
    if cache_key in _llm_cache:
        response, timestamp = _llm_cache[cache_key]
        if time.time() - timestamp < settings.LLM_CACHE_TTL:
            return response
        else:
            del _llm_cache[cache_key]  # Remove expired entry
    return None

def _cache_response(cache_key: str, response: str):
    """Cache LLM response with proper LRU eviction."""
    if settings.LLM_CACHE_ENABLED:
        _llm_cache[cache_key] = (response, time.time())
        _llm_cache.move_to_end(cache_key)  # Mark as most recently used
        
        # Enforce cache size limit with LRU eviction
        while len(_llm_cache) > MAX_CACHE_SIZE:
            # Remove least recently used entry
            _llm_cache.popitem(last=False)

def _get_llm_model() -> Optional[BaseLanguageModel]:
    """Initialize and return the configured LLM model."""
    if not LANGCHAIN_AVAILABLE or not settings.LLM_ENABLED:
        return None
    
    try:
        if settings.LLM_PROVIDER == "openai":
            if not settings.OPENAI_API_KEY:
                logger.error("OpenAI API key not configured")
                return None
            return LLMModel(
                model=settings.LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
                organization=settings.OPENAI_ORG_ID,
                base_url=settings.OPENAI_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES
            )
            
        elif settings.LLM_PROVIDER == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                logger.error("Anthropic API key not configured")
                return None
            return LLMModel(
                model=settings.LLM_MODEL,
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT,
                max_retries=settings.LLM_MAX_RETRIES
            )
            
        elif settings.LLM_PROVIDER == "ollama":
            return LLMModel(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                timeout=settings.LLM_TIMEOUT
            )
            
        elif settings.LLM_PROVIDER == "huggingface":
            if not settings.HUGGINGFACE_API_TOKEN:
                logger.error("Hugging Face API token not configured")
                return None
            return LLMModel.from_model_id(
                model_id=settings.HUGGINGFACE_MODEL,
                huggingfacehub_api_token=settings.HUGGINGFACE_API_TOKEN,
                model_kwargs={"temperature": settings.LLM_TEMPERATURE, "max_length": settings.LLM_MAX_TOKENS}
            )
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM model: {e}")
        return None
    
    return None

async def _generate_rationale_with_service(bars: list[dict], action: str, conf: float, details: dict, 
                                         risk_score: int, sentiment_hint: Optional[float], 
                                         symbol: str, asset_class: str, timeframe: str) -> Optional[str]:
    """Generate rationale using the new integrated LLM service"""
    try:
        llm_service = get_llm_service()
        
        # Prepare context for LLM
        context = _prepare_llm_context(
            bars, action, conf, details, risk_score, sentiment_hint,
            symbol, asset_class, timeframe
        )
        
        # Create system message for financial analysis
        system_content = f"""You are a professional financial advisor providing detailed analysis and recommendations for {asset_class} trading.

Your task is to generate clear, insightful, and actionable trading advice explanations based on technical analysis.

Current Action: {action.upper()}
Confidence: {conf:.2f}
Risk Profile: {_get_risk_profile_name(risk_score)}
Asset: {symbol}
Timeframe: {timeframe}

Please provide a comprehensive analysis with:
1. Market Analysis - Current market conditions and technical indicators
2. Rationale for {action.upper()} Recommendation - Detailed reasoning with specific indicators
3. Risk Assessment - Risks considering the {_get_risk_profile_name(risk_score)} risk profile
4. Key Considerations - Important factors that could affect this trade
5. Confidence Level - Why the confidence is {conf:.2f} and factors that could change it

Keep the explanation professional, educational, and actionable. Include appropriate risk disclaimers."""

        # Create analysis request
        system_message = SystemMessage(content=system_content)
        analysis_prompt = f"""
TECHNICAL ANALYSIS:
{format_technical_indicators(details)}

MARKET CONDITIONS:
Sentiment Score: {_format_sentiment(sentiment_hint)}
Market Regime: {context.get('market_regime', 'normal')}

KEY SIGNALS:
{context.get('key_signals', 'Standard technical analysis signals')}

RISK FACTORS:
{format_risk_factors(_get_risk_profile_name(risk_score), {'regime': context.get('market_regime', 'normal')})}

Please provide your comprehensive analysis for this {action} recommendation on {symbol}.
"""
        
        user_message = HumanMessage(content=analysis_prompt)
        
        # Create LLM request with appropriate settings
        request = LLMRequest(
            messages=[system_message, user_message],
            task_type="analysis",
            priority=RequestPriority.HIGH,
            quality=ResponseQuality.PREMIUM,
            complexity=TaskComplexity.COMPLEX,
            max_tokens=2000,
            temperature=0.2
        )
        
        # Generate response
        response = await llm_service.generate_response(request)
        
        if response.success and response.content:
            # Add risk disclaimer if not present
            content = response.content
            if "RISK DISCLOSURES" not in content:
                content += "\n\n" + RISK_DISCLAIMER_TEMPLATE
            
            logger.info(f"Generated LLM rationale using {response.provider} (${response.cost:.4f})")
            return content
        else:
            logger.warning(f"LLM service failed: {response.error_message}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating rationale with LLM service: {e}")
        return None

async def _generate_rationale_legacy(bars: list[dict], action: str, conf: float, details: dict,
                                   risk_score: int, sentiment_hint: Optional[float],
                                   symbol: str, asset_class: str, timeframe: str) -> Optional[str]:
    """Legacy rationale generation using direct LangChain"""
    try:
        llm = _get_llm_model()
        if not llm:
            return None
            
        context = _prepare_llm_context(
            bars, action, conf, details, risk_score, sentiment_hint,
            symbol, asset_class, timeframe
        )
        
        rationale = await _generate_llm_rationale(llm, context, asset_class)
        
        if rationale and "RISK DISCLOSURES" not in rationale:
            rationale += "\n\n" + RISK_DISCLAIMER_TEMPLATE
            
        return rationale
        
    except Exception as e:
        logger.error(f"Legacy rationale generation failed: {e}")
        return None

def _format_sentiment(sentiment_hint: Optional[float]) -> str:
    """Format sentiment score for display"""
    if sentiment_hint is None:
        return "neutral"
    elif sentiment_hint > 0.3:
        return f"strongly positive ({sentiment_hint:.2f})"
    elif sentiment_hint > 0.1:
        return f"moderately positive ({sentiment_hint:.2f})"
    elif sentiment_hint < -0.3:
        return f"strongly negative ({sentiment_hint:.2f})"
    elif sentiment_hint < -0.1:
        return f"moderately negative ({sentiment_hint:.2f})"
    else:
        return f"neutral ({sentiment_hint:.2f})"

def _generate_simple_rationale(action: str, conf: float, details: dict, risk_score: int, sentiment_hint: Optional[float]) -> str:
    """Generate simple rationale when LLM is not available."""
    risk_profile = _get_risk_profile_name(risk_score)
    sentiment_text = ""
    if sentiment_hint is not None:
        if sentiment_hint > 0.1:
            sentiment_text = " Positive market sentiment supports this signal."
        elif sentiment_hint < -0.1:
            sentiment_text = " Negative market sentiment adds caution to this signal."
    
    return f"Technical analysis suggests {action.upper()} with {conf:.2f} confidence. Risk profile: {risk_profile}.{sentiment_text} Key indicators: {details}."

def _prepare_llm_context(bars: list[dict], action: str, conf: float, details: dict, 
                        risk_score: int, sentiment_hint: Optional[float], 
                        symbol: str = "SYMBOL", asset_class: str = "stock", timeframe: str = "1h") -> dict:
    """Prepare context for LLM prompt."""
    
    risk_profile = _get_risk_profile_name(risk_score)
    
    # Format technical indicators for better LLM understanding
    formatted_indicators = format_technical_indicators(details)
    
    # Determine market regime from recent bars
    df = make_df(bars)
    returns = df["close"].pct_change().dropna()
    
    if len(returns) >= 10:
        recent_vol = returns.tail(10).std() * (252 ** 0.5)
        avg_vol = returns.std() * (252 ** 0.5)
        
        if recent_vol > 1.5 * avg_vol:
            market_regime = "high_vol"
        elif recent_vol < 0.7 * avg_vol:
            market_regime = "low_vol"
        else:
            market_regime = "normal"
    else:
        market_regime = "normal"
    
    # Enhanced sentiment interpretation
    sentiment_score = sentiment_hint or 0.0
    if sentiment_score > 0.3:
        sentiment_desc = "strongly positive"
    elif sentiment_score > 0.1:
        sentiment_desc = "moderately positive"
    elif sentiment_score < -0.3:
        sentiment_desc = "strongly negative"
    elif sentiment_score < -0.1:
        sentiment_desc = "moderately negative"
    else:
        sentiment_desc = "neutral"
    
    # Prepare key signals summary
    key_signals = []
    for indicator, value in details.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    key_signals.append(f"{indicator} {sub_key}: {sub_value:.4f}")
        elif isinstance(value, (int, float)):
            key_signals.append(f"{indicator}: {value:.4f}")
    
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "action": action,
        "confidence": conf,
        "risk_profile": risk_profile,
        "technical_indicators": formatted_indicators,
        "sentiment_score": sentiment_desc,
        "market_regime": market_regime,
        "key_signals": "\n".join(key_signals),
        "risk_factors": format_risk_factors(risk_profile, {"regime": market_regime})
    }

async def _generate_llm_rationale(llm: BaseLanguageModel, context: dict, asset_class: str) -> Optional[str]:
    """Generate sophisticated rationale using LLM."""
    try:
        # Select appropriate prompt template
        prompt_template = get_prompt_for_action(context["action"], asset_class)
        
        # Create the chain using LCEL (LangChain Expression Language)
        chain = (
            RunnablePassthrough()
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Generate response
        if settings.LLM_STREAMING:
            # For streaming (not implemented in this basic version)
            response = await chain.ainvoke(context)
        else:
            response = await chain.ainvoke(context)
        
        # Add disclaimer
        if response and not "RISK DISCLOSURES" in response:
            response += "\n\n" + RISK_DISCLAIMER_TEMPLATE
        
        return response
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return None

def advise(bars: list[dict], risk_score: int, sentiment_hint: float | None = None,
           template: str | None = None, params: dict | None = None, 
           symbol: str = "SYMBOL", asset_class: str = "stock", timeframe: str = "1h") -> Tuple[str, float, dict, str, str]:
    """Enhanced advisor function with LLM-powered rationale generation."""
    
    # 1. Generate core trading signal using existing logic
    df = make_df(bars)
    if template and template in TEMPLATES:
        action, conf, details = TEMPLATES[template](df, **(params or {}))
    else:
        action, conf, details = ensemble(df, params or {})

    # 2. Apply risk score adjustments
    original_conf = conf
    if risk_score <= 25 and action == "buy":
        conf *= 0.8
        risk_notes = "Conservative profile: reduced position size recommended."
    elif risk_score >= 75 and action in ("buy","sell"):
        conf *= 1.05
        risk_notes = "Aggressive profile: higher conviction permitted."
    else:
        risk_notes = "Moderate risk profile: balanced approach."

    # 3. Apply sentiment adjustments
    if sentiment_hint is not None:
        conf = float(max(0.1, min(0.95, conf * (1 + 0.2*sentiment_hint))))
    
    # 4. Generate sophisticated rationale using new LLM service
    rationale = None
    
    if settings.LLM_ENABLED:
        try:
            if LLM_SERVICE_AVAILABLE:
                # Use new integrated LLM service
                rationale = asyncio.run(_generate_rationale_with_service(
                    bars, action, conf, details, risk_score, sentiment_hint,
                    symbol, asset_class, timeframe
                ))
            elif LANGCHAIN_AVAILABLE:
                # Fallback to legacy LangChain implementation
                rationale = asyncio.run(_generate_rationale_legacy(
                    bars, action, conf, details, risk_score, sentiment_hint,
                    symbol, asset_class, timeframe
                ))
        except Exception as e:
            logger.error(f"LLM rationale generation error: {e}")
            rationale = None
    
    # 5. Fallback to simple rationale if LLM unavailable or failed
    if not rationale:
        if settings.LLM_FALLBACK_ENABLED and settings.LLM_FALLBACK_SIMPLE_RATIONALE:
            rationale = _generate_simple_rationale(action, conf, details, risk_score, sentiment_hint)
            logger.info("Using simple rationale fallback")
        else:
            rationale = f"Signals suggest {action.upper()} with confidence {conf:.2f}. Details: {details}."
    
    # 6. Enhance risk notes with LLM insights if available
    if settings.LLM_CONTEXT_ENHANCEMENT and rationale and len(rationale) > 200:
        # Extract risk insights from LLM response
        if "risk" in rationale.lower() and original_conf != conf:
            risk_adjustment = "(confidence adjusted for risk profile)" if conf != original_conf else ""
            risk_notes = f"{risk_notes} {risk_adjustment}"
    
    return action, conf, details, rationale, risk_notes

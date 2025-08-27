"""
LangChain prompt templates for generating sophisticated financial advice explanations.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

# Base financial advice prompt template
FINANCIAL_ADVICE_TEMPLATE = """You are a professional financial advisor AI providing detailed analysis and recommendations. 
Your task is to generate clear, insightful, and actionable trading advice explanations based on technical analysis.

CONTEXT:
Symbol: {symbol}
Asset Class: {asset_class}
Timeframe: {timeframe}
Current Action: {action}
Base Confidence: {confidence:.2f}
Risk Profile: {risk_profile}

TECHNICAL ANALYSIS:
{technical_indicators}

MARKET CONDITIONS:
Sentiment Score: {sentiment_score}
Market Regime: {market_regime}
Key Signals: {key_signals}

RISK FACTORS:
{risk_factors}

Please provide a comprehensive analysis with the following structure:

## Market Analysis
Explain the current market conditions and what the technical indicators are telling us about {symbol}.

## Rationale for {action.upper()} Recommendation  
Provide detailed reasoning for why this {action} signal was generated, referencing specific technical indicators and their values.

## Risk Assessment
Discuss the risks associated with this position, considering the investor's {risk_profile} risk profile and current market conditions.

## Key Considerations
Highlight important factors that could affect this trade, including potential catalysts or headwinds.

## Confidence Level
Explain why the confidence level is {confidence:.2f} and what factors could increase or decrease conviction in this trade.

Keep the explanation professional, educational, and actionable. Use specific technical values where relevant. 
Avoid making guarantees about future performance and always include appropriate disclaimers about market risks.

IMPORTANT DISCLAIMERS TO INCLUDE:
- Past performance does not guarantee future results
- All investments carry risk of loss
- Consider consulting with a qualified financial advisor
- This analysis is for informational purposes only and not personalized advice
"""

BUY_SIGNAL_TEMPLATE = """You are analyzing a BUY signal for {symbol}. Focus on:

BULLISH INDICATORS:
{technical_indicators}

Current Market Setup:
- Action: BUY
- Confidence: {confidence:.2f}
- Risk Profile: {risk_profile}
- Sentiment: {sentiment_score}

Provide a detailed explanation of why this BUY signal was generated, emphasizing:
1. Technical momentum and trend confirmation
2. Entry timing and price levels
3. Upside potential and price targets
4. Stop-loss considerations
5. Position sizing recommendations based on {risk_profile} risk tolerance

Risk Considerations for Long Position:
- Market volatility impact
- Sector/asset class specific risks
- Macro-economic factors
- Liquidity considerations

Remember to maintain objectivity and include appropriate risk warnings.

MARKET DATA:
{key_signals}

Generate a comprehensive BUY recommendation analysis that is educational and actionable.
"""

SELL_SIGNAL_TEMPLATE = """You are analyzing a SELL signal for {symbol}. Focus on:

BEARISH INDICATORS:
{technical_indicators}

Current Market Setup:
- Action: SELL
- Confidence: {confidence:.2f}
- Risk Profile: {risk_profile}
- Sentiment: {sentiment_score}

Provide a detailed explanation of why this SELL signal was generated, emphasizing:
1. Technical breakdown and trend reversal signals
2. Exit timing and price levels
3. Downside risks and protection strategies
4. Short-selling considerations (if applicable)
5. Risk management for bearish positions

Risk Considerations for Short Position:
- Unlimited loss potential (for short positions)
- Market volatility and squeeze risks
- Sector rotation possibilities
- Regulatory and borrowing costs

Be particularly cautious with sell recommendations and emphasize risk management.

MARKET DATA:
{key_signals}

Generate a comprehensive SELL recommendation analysis focusing on risk mitigation.
"""

HOLD_SIGNAL_TEMPLATE = """You are analyzing a HOLD/NEUTRAL signal for {symbol}. Focus on:

MIXED OR NEUTRAL INDICATORS:
{technical_indicators}

Current Market Setup:
- Action: HOLD
- Confidence: {confidence:.2f}
- Risk Profile: {risk_profile}
- Sentiment: {sentiment_score}

Provide a detailed explanation of why a HOLD position is recommended, emphasizing:
1. Conflicting technical signals or range-bound conditions
2. Wait-and-see market conditions
3. Consolidation patterns and breakout potential
4. Risk-reward not favorable for active trading
5. Portfolio rebalancing considerations

When to Reconsider HOLD Position:
- Key technical levels to watch (support/resistance)
- Catalysts that could trigger action signals
- Market regime changes
- Volume and volatility shifts

MARKET DATA:
{key_signals}

Generate a comprehensive HOLD analysis that helps investors understand why patience is currently the best strategy.
"""

OPTIONS_ANALYSIS_TEMPLATE = """You are analyzing an OPTIONS strategy signal for {symbol}. Focus on:

OPTIONS SPECIFIC DATA:
{technical_indicators}

Greeks Analysis:
- Delta: {delta}
- Gamma: {gamma}  
- Theta: {theta}
- Vega: {vega}

Strategy Setup:
- Action: {action}
- Strategy Type: {strategy_type}
- Confidence: {confidence:.2f}
- Implied Volatility: {implied_vol}

Provide detailed analysis covering:
1. Options strategy rationale (straddle, strangle, spread, etc.)
2. Greeks interpretation and risk management
3. Volatility analysis and expectations
4. Time decay considerations
5. Profit/loss scenarios and breakeven points
6. Position sizing and capital requirements

Options-Specific Risks:
- Time decay (theta)
- Volatility changes (vega)
- Pin risk at expiration
- Liquidity and bid-ask spreads
- Assignment risk (for written options)

MARKET CONDITIONS:
{market_regime}
Sentiment: {sentiment_score}

Generate sophisticated options analysis suitable for experienced traders.
"""

FX_ANALYSIS_TEMPLATE = """You are analyzing a FOREX signal for {symbol}. Focus on:

CURRENCY PAIR ANALYSIS:
{technical_indicators}

Fundamental Factors:
- Interest Rate Differential: {interest_rate_diff}
- Economic Data: {economic_indicators}
- Central Bank Policy: {central_bank_stance}

Technical Setup:
- Action: {action}
- Confidence: {confidence:.2f}
- Currency Pair Type: {pair_type}

Provide comprehensive FX analysis including:
1. Technical analysis specific to currency markets
2. Fundamental drivers (interest rates, economic data)
3. Central bank policy implications
4. Carry trade opportunities
5. Risk-on/risk-off sentiment impact
6. Key support/resistance levels

FX-Specific Considerations:
- Leverage and margin requirements
- Rollover/swap rates
- Economic calendar events
- Cross-currency correlations
- Geopolitical risk factors

MARKET SENTIMENT: {sentiment_score}
MARKET REGIME: {market_regime}

Generate professional forex analysis suitable for currency traders.
"""

FUTURES_ANALYSIS_TEMPLATE = """You are analyzing a FUTURES signal for {symbol}. Focus on:

FUTURES CONTRACT ANALYSIS:
{technical_indicators}

Contract Specifications:
- Contract Type: {contract_type}
- Days to Expiry: {days_to_expiry}
- Roll Risk Assessment: {roll_risk}

Market Structure:
- Contango/Backwardation: {curve_signal}
- Seasonal Patterns: {seasonal_component}
- Storage/Carry Costs: {carry_costs}

Technical Setup:
- Action: {action}
- Confidence: {confidence:.2f}

Provide comprehensive futures analysis including:
1. Contract-specific technical analysis
2. Curve structure and roll implications
3. Seasonal and cyclical patterns
4. Storage and carry cost factors
5. Supply/demand fundamentals
6. Expiry and roll strategy

Futures-Specific Risks:
- Roll risk near expiry
- Margin calls and leverage
- Delivery obligations
- Basis risk
- Commodity-specific factors

MARKET CONDITIONS:
Sentiment: {sentiment_score}
Market Regime: {market_regime}

Generate professional futures analysis for institutional-grade trading.
"""

# Prompt template instances
financial_advice_prompt = PromptTemplate(
    input_variables=[
        "symbol", "asset_class", "timeframe", "action", "confidence", 
        "risk_profile", "technical_indicators", "sentiment_score", 
        "market_regime", "key_signals", "risk_factors"
    ],
    template=FINANCIAL_ADVICE_TEMPLATE
)

buy_signal_prompt = PromptTemplate(
    input_variables=[
        "symbol", "confidence", "risk_profile", "sentiment_score",
        "technical_indicators", "key_signals"
    ],
    template=BUY_SIGNAL_TEMPLATE
)

sell_signal_prompt = PromptTemplate(
    input_variables=[
        "symbol", "confidence", "risk_profile", "sentiment_score",
        "technical_indicators", "key_signals"  
    ],
    template=SELL_SIGNAL_TEMPLATE
)

hold_signal_prompt = PromptTemplate(
    input_variables=[
        "symbol", "confidence", "risk_profile", "sentiment_score",
        "technical_indicators", "key_signals"
    ],
    template=HOLD_SIGNAL_TEMPLATE
)

options_analysis_prompt = PromptTemplate(
    input_variables=[
        "symbol", "action", "confidence", "strategy_type", "technical_indicators",
        "delta", "gamma", "theta", "vega", "implied_vol", "market_regime", "sentiment_score"
    ],
    template=OPTIONS_ANALYSIS_TEMPLATE
)

fx_analysis_prompt = PromptTemplate(
    input_variables=[
        "symbol", "action", "confidence", "pair_type", "technical_indicators",
        "interest_rate_diff", "economic_indicators", "central_bank_stance", "sentiment_score", "market_regime"
    ],
    template=FX_ANALYSIS_TEMPLATE
)

futures_analysis_prompt = PromptTemplate(
    input_variables=[
        "symbol", "action", "confidence", "contract_type", "days_to_expiry", 
        "roll_risk", "curve_signal", "seasonal_component", "carry_costs",
        "technical_indicators", "sentiment_score", "market_regime"
    ],
    template=FUTURES_ANALYSIS_TEMPLATE
)

# Risk disclaimer template
RISK_DISCLAIMER_TEMPLATE = """
⚠️ IMPORTANT RISK DISCLOSURES:

• Past performance does not guarantee future results
• All investments carry substantial risk of loss
• Trading involves significant financial risk and may not be suitable for all investors
• Consider consulting with a qualified financial advisor before making investment decisions
• This analysis is for informational and educational purposes only
• Not personalized investment advice - consider your individual financial situation
• Market conditions can change rapidly, invalidating technical analysis
• Leverage amplifies both gains and losses - use appropriate position sizing
• Always implement proper risk management and stop-loss strategies

The information provided is based on technical analysis and market data available at the time of analysis. 
Market conditions are subject to rapid change, and no representation is made that any account will or is likely to achieve profits similar to those discussed.
"""

def get_prompt_for_action(action: str, asset_class: str = "stock") -> PromptTemplate:
    """
    Returns the appropriate prompt template based on action and asset class.
    
    Args:
        action: The trading action ("buy", "sell", "hold")
        asset_class: The asset class ("stock", "crypto", "fx", "futures", "option")
    
    Returns:
        PromptTemplate: The appropriate prompt template
    """
    # Asset-specific prompts
    if asset_class == "option":
        return options_analysis_prompt
    elif asset_class == "fx":
        return fx_analysis_prompt
    elif asset_class == "futures":
        return futures_analysis_prompt
    
    # Action-specific prompts for other asset classes
    if action.lower() == "buy":
        return buy_signal_prompt
    elif action.lower() == "sell":
        return sell_signal_prompt
    elif action.lower() in ["hold", "reduce", "avoid"]:
        return hold_signal_prompt
    else:
        return financial_advice_prompt

def format_technical_indicators(indicators: Dict[str, Any]) -> str:
    """Format technical indicators dictionary into readable text."""
    if not indicators:
        return "No technical indicators available."
    
    formatted = []
    for key, value in indicators.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (e.g., MACD components)
            sub_items = [f"{k}: {v}" for k, v in value.items()]
            formatted.append(f"{key.upper()}: {', '.join(sub_items)}")
        elif isinstance(value, (int, float)):
            formatted.append(f"{key.upper()}: {value:.4f}")
        else:
            formatted.append(f"{key.upper()}: {value}")
    
    return "\n".join(formatted)

def format_risk_factors(risk_profile: str, market_conditions: Dict[str, Any] = None) -> str:
    """Format risk factors based on risk profile and market conditions."""
    risk_factors = []
    
    # Base risk factors by profile
    if risk_profile == "conservative":
        risk_factors.extend([
            "Conservative profile requires lower volatility exposure",
            "Focus on capital preservation over growth",
            "Consider reducing position sizes in uncertain markets"
        ])
    elif risk_profile == "aggressive":
        risk_factors.extend([
            "Aggressive profile allows for higher risk tolerance",
            "Potential for amplified gains and losses",
            "Monitor for excessive concentration risk"
        ])
    else:
        risk_factors.extend([
            "Balanced approach to risk and reward",
            "Moderate position sizing appropriate",
            "Regular portfolio rebalancing recommended"
        ])
    
    # Market-specific risks
    if market_conditions:
        if market_conditions.get("regime") == "high_vol":
            risk_factors.append("High volatility regime increases trade uncertainty")
        elif market_conditions.get("regime") == "low_vol":
            risk_factors.append("Low volatility may indicate complacency or upcoming volatility expansion")
    
    return "\n".join(f"• {factor}" for factor in risk_factors)

# Export all prompt templates and utility functions
__all__ = [
    "financial_advice_prompt",
    "buy_signal_prompt", 
    "sell_signal_prompt",
    "hold_signal_prompt",
    "options_analysis_prompt",
    "fx_analysis_prompt", 
    "futures_analysis_prompt",
    "get_prompt_for_action",
    "format_technical_indicators",
    "format_risk_factors",
    "RISK_DISCLAIMER_TEMPLATE"
]
# SuperNova Master Brief

## Vision & Mission
SuperNova is the AI-powered **financial advisor engine** for NovaSignal — your personal trading assistant.  
It is not a broker or trading platform. Instead, it helps traders and investors make informed decisions across multiple asset classes (stocks, crypto, FX, futures, and options).

SuperNova’s mission:
- Provide **personalized advice** by combining technical indicators, market sentiment, and user risk profiles.  
- Offer **transparent rationale** behind every recommendation.  
- Continuously **learn and improve** strategies through backtesting and recursive analysis.  
- Maintain **compliance awareness** and safe disclaimers.  

## Core Principles
1. **Educational & Research-Oriented**: All outputs framed as research, not actionable financial advice.
2. **Transparency**: Always provide rationale, confidence, and the signals behind advice.
3. **User-Centric**: Intake process gathers financial profile, goals, constraints, and risk tolerance.
4. **Explainability**: Advice is delivered in plain language with supporting metrics.
5. **Iterative Learning**: Recursive backtesting and strategy improvement loops.
6. **Compliance-Ready**: Designed with record-keeping, suitability checks, and disclaimers.

## Key Components
- **API Layer**: FastAPI service exposing intake, advice, backtest, alerts, and watchlist routes.
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger, ATR.
- **Strategy Engine**: MA crossover, RSI breakout, MACD trend, ensemble voting, and extended templates (options straddle, FX breakout, futures trend).
- **Sentiment Engine**: Lightweight sentiment scorer + hooks for social feeds.
- **Risk Profiling**: Converts questionnaire responses into a 0–100 score and adjusts confidence levels accordingly.
- **Alerting System**: Triggers breakout alerts for watchlisted assets; integrates with NovaSignal via webhook or API.
- **Backtesting Engine**: Fast vectorized simulator to evaluate strategies with Sharpe, CAGR, Max Drawdown, Win Rate.
- **Recursive Loop**: Evaluates and ranks strategies, storing learnings for improvement.
- **Journal**: Session log (`SuperJournal.txt`) documenting events, agent activity, and outputs.

## Collaboration Guidelines
- **All contributions** should preserve clarity, transparency, and explainability.
- **Agents (e.g., Claude sub-agents)** can take specific modules to refine (e.g., better indicators, more connectors, stronger tests).
- **Human collaborators** should review high-priority modules first (critical path items) and extend strategy libraries responsibly.
- **Logs & Journal** must always reflect what was changed or tested.


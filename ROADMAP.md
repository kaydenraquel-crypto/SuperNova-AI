# SuperNova Development Roadmap

## Phase 1 — Foundations (Critical Priority)
- ✅ Scaffold core project (API, DB, schemas, journal, sample indicators).
- ✅ Intake → risk profiling → advice → rationale flow.
- ✅ Watchlists, alerts, and sample RSI-based breakout detection.
- ✅ Backtesting engine with Sharpe, CAGR, MaxDD, Win Rate.
- ✅ Ensemble strategies and journal logging.
- 🔲 Unit tests expanded for all strategies and sentiment.

## Phase 2 — Multi-Asset Extension (High Priority)
- ✅ Options strategy template (straddle volatility breakout).
- ✅ FX strategy template (range breakout).
- ✅ Futures strategy template (MA trend).
- 🔲 Expand backtester for margin, leverage, and contract multipliers.
- 🔲 Add options Greeks calculator (Delta, Vega) for risk metrics.

## Phase 3 — Sentiment & Data Integration (High Priority)
- 🔲 Implement connectors for X (Twitter) and Reddit feeds (with ToS-compliance).
- 🔲 Blend sentiment scores with technical signals for improved confidence.
- 🔲 Add news sentiment (headlines via APIs).

## Phase 4 — Risk & Compliance (Medium Priority)
- 🔲 Enhance intake with goals, constraints, liquidity needs.
- 🔲 Block unsuitable strategies based on profile risk.
- 🔲 Add compliance audit trail (JSON export of advice + rationale).

## Phase 5 — Recursive Learning & Improvement (Medium Priority)
- 🔲 Automate nightly re-backtesting of stored strategies.
- 🔲 Rank and tune strategy parameters (genetic/evolutionary search).
- 🔲 Store historical advice vs. actual outcome metrics.

## Phase 6 — Integration with NovaSignal (High Priority)
- 🔲 Build connectors to stream OHLCV data from NovaSignal to SuperNova.
- 🔲 Send alerts back to NovaSignal dashboard.
- 🔲 Add user-facing explanations inside NovaSignal UI.

## Phase 7 — Future Enhancements (Lower Priority)
- 🔲 Portfolio optimization module (mean-variance / Black-Litterman).
- 🔲 NLP-powered Q&A interface (LLM integration).
- 🔲 Multi-objective optimization (risk-adjusted return vs. ESG factors).
- 🔲 Add voice-based interaction and personal assistant UX.

---
### Priority Tiers
- **Critical**: Required for minimal working product (Phases 1–2 core).  
- **High**: Strongly recommended for v1 release (Phases 2–3).  
- **Medium**: Important but can follow v1.  
- **Low/Future**: Nice-to-have, later roadmap items.  


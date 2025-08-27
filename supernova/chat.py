"""
SuperNova Conversational Agent System

Provides comprehensive conversational agent capabilities with:
- LangChain ReAct agent integration
- Multi-LLM provider support
- Memory and context management
- Financial expertise and analysis
- Tool integration and chaining
- Personalized responses
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import json
import logging
import hashlib
import time
from dataclasses import dataclass

# Core imports
from .config import settings
from .conversation_memory import (
    MemoryManager, ConversationMemory, UserContextMemory, FinancialContextMemory, 
    SessionMemory, ConversationRole, MemoryType, memory_manager
)
from .advisor import advise, _get_llm_model
from .db import SessionLocal, User, Profile

# LangChain imports with error handling
try:
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import BaseTool, Tool, tool
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.memory import ConversationBufferWindowMemory
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain.schema import AgentAction, AgentFinish
    
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
        from langchain_openai import ChatOpenAI as LLMModel
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}. Conversational agent will use fallback mode.")
    LANGCHAIN_AVAILABLE = False
    BaseLanguageModel = None
    LLMModel = None

logger = logging.getLogger(__name__)

class AgentPersonality(str, Enum):
    """Agent personality types"""
    CONSERVATIVE_ADVISOR = "conservative_advisor"
    AGGRESSIVE_TRADER = "aggressive_trader" 
    BALANCED_ANALYST = "balanced_analyst"
    EDUCATIONAL_MODE = "educational_mode"
    RESEARCH_ASSISTANT = "research_assistant"

class QueryType(str, Enum):
    """Types of user queries"""
    FINANCIAL_ANALYSIS = "financial_analysis"
    STRATEGY_ADVICE = "strategy_advice"
    MARKET_RESEARCH = "market_research"
    PORTFOLIO_REVIEW = "portfolio_review"
    EDUCATIONAL = "educational"
    GENERAL_CHAT = "general_chat"

@dataclass
class ConversationContext:
    """Current conversation context"""
    conversation_id: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    personality: AgentPersonality = AgentPersonality.BALANCED_ANALYST
    current_symbol: Optional[str] = None
    active_analysis: Optional[Dict[str, Any]] = None
    risk_profile: Optional[str] = None
    user_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}

class StreamingCallback(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses"""
    
    def __init__(self, callback_fn: Callable[[str], None]):
        self.callback_fn = callback_fn
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle streaming token"""
        self.tokens.append(token)
        self.callback_fn(token)
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle end of LLM generation"""
        full_response = "".join(self.tokens)
        self.callback_fn(f"\n[RESPONSE_COMPLETE: {len(full_response)} characters]")

class SuperNovaConversationalAgent:
    """
    Main conversational agent with sophisticated financial expertise
    """
    
    def __init__(self, 
                 memory_manager: MemoryManager = None,
                 personality: AgentPersonality = AgentPersonality.BALANCED_ANALYST):
        
        self.memory_manager = memory_manager or memory_manager
        self.personality = personality
        self.llm = None
        self.agent_executor = None
        self.tools = []
        
        # Initialize LLM and agent
        self._initialize_llm()
        self._initialize_tools()
        self._initialize_agent()
        
        # Performance tracking
        self._response_times = []
        self._query_counts = {}
        self._error_counts = {}
        
        logger.info(f"SuperNova Conversational Agent initialized with personality: {personality}")
    
    def _initialize_llm(self):
        """Initialize the language model"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, using fallback mode")
            return
        
        self.llm = _get_llm_model()
        if not self.llm:
            logger.error("Failed to initialize LLM model")
            return
            
        logger.info(f"LLM initialized: {settings.LLM_PROVIDER} - {settings.LLM_MODEL}")
    
    def _initialize_tools(self):
        """Initialize agent tools"""
        self.tools = []
        
        # Financial analysis tool
        @tool
        def analyze_symbol(symbol: str, timeframe: str = "1h", bars_data: str = None) -> str:
            """
            Analyze a financial symbol using technical analysis.
            
            Args:
                symbol: Stock/crypto symbol (e.g., AAPL, BTCUSD)
                timeframe: Analysis timeframe (1h, 4h, 1d)
                bars_data: Optional OHLCV data as JSON string
            
            Returns:
                Detailed technical analysis results
            """
            try:
                # This would integrate with real market data
                # For now, return mock analysis
                analysis_result = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat(),
                    "action": "hold",
                    "confidence": 0.65,
                    "key_indicators": {
                        "rsi": 45.2,
                        "macd": {"macd": 0.12, "signal": 0.08, "histogram": 0.04},
                        "moving_averages": {"sma_20": 150.5, "sma_50": 148.2}
                    },
                    "analysis": f"Technical analysis for {symbol} shows neutral conditions with RSI at 45.2 indicating balanced momentum."
                }
                return json.dumps(analysis_result, indent=2)
            except Exception as e:
                return f"Error analyzing {symbol}: {str(e)}"
        
        # Market research tool
        @tool
        def research_market_conditions(sector: str = None, timeframe: str = "1d") -> str:
            """
            Research current market conditions and sentiment.
            
            Args:
                sector: Optional sector focus (tech, finance, healthcare, etc.)
                timeframe: Analysis timeframe (1d, 1w, 1m)
            
            Returns:
                Market research summary
            """
            try:
                # Mock market research - would integrate with real data sources
                market_conditions = {
                    "overall_sentiment": "cautiously optimistic",
                    "volatility": "moderate",
                    "sector_rotation": "from growth to value",
                    "key_risks": ["inflation concerns", "geopolitical tensions"],
                    "opportunities": ["defensive stocks", "dividend plays"],
                    "market_regime": "range-bound with breakout potential"
                }
                
                if sector:
                    market_conditions["sector_focus"] = f"Sector analysis for {sector} shows mixed signals with earnings revisions trending neutral."
                
                return json.dumps(market_conditions, indent=2)
            except Exception as e:
                return f"Error researching market conditions: {str(e)}"
        
        # Portfolio analysis tool
        @tool
        def analyze_portfolio_correlation(symbols_list: str) -> str:
            """
            Analyze correlation between portfolio symbols.
            
            Args:
                symbols_list: Comma-separated list of symbols (e.g., "AAPL,MSFT,GOOGL")
            
            Returns:
                Portfolio correlation analysis
            """
            try:
                symbols = [s.strip() for s in symbols_list.split(",")]
                
                # Mock correlation analysis
                analysis = {
                    "symbols": symbols,
                    "overall_correlation": "moderate",
                    "diversification_score": 0.72,
                    "high_correlations": [f"{symbols[0]} - {symbols[1]}: 0.65"],
                    "recommendations": [
                        "Consider adding uncorrelated assets",
                        "Monitor sector concentration risk"
                    ]
                }
                
                return json.dumps(analysis, indent=2)
            except Exception as e:
                return f"Error analyzing portfolio correlation: {str(e)}"
        
        # Strategy backtesting tool
        @tool
        def backtest_strategy(strategy_name: str, symbol: str, parameters: str = None) -> str:
            """
            Backtest a trading strategy.
            
            Args:
                strategy_name: Strategy to test (rsi_mean_reversion, ma_crossover, etc.)
                symbol: Symbol to backtest on
                parameters: Optional strategy parameters as JSON string
            
            Returns:
                Backtesting results
            """
            try:
                # Mock backtesting results
                results = {
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "period": "1 year",
                    "total_return": "15.2%",
                    "sharpe_ratio": 1.34,
                    "max_drawdown": "-8.5%",
                    "win_rate": "58%",
                    "total_trades": 45,
                    "summary": f"{strategy_name} on {symbol} shows solid performance with good risk-adjusted returns"
                }
                
                if parameters:
                    results["parameters"] = json.loads(parameters)
                
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error backtesting strategy: {str(e)}"
        
        # Economic calendar tool
        @tool
        def check_economic_calendar(days_ahead: int = 7, importance: str = "high") -> str:
            """
            Check upcoming economic events.
            
            Args:
                days_ahead: Number of days to look ahead
                importance: Filter by importance (low, medium, high, all)
            
            Returns:
                Upcoming economic events
            """
            try:
                # Mock economic calendar
                events = [
                    {
                        "date": (datetime.now() + timedelta(days=2)).isoformat()[:10],
                        "event": "Federal Reserve Interest Rate Decision",
                        "importance": "high",
                        "expected": "5.25%",
                        "impact": "USD, equities"
                    },
                    {
                        "date": (datetime.now() + timedelta(days=5)).isoformat()[:10],
                        "event": "Non-Farm Payrolls",
                        "importance": "high",
                        "expected": "180K",
                        "impact": "USD, bonds, equities"
                    }
                ]
                
                filtered_events = [e for e in events if importance == "all" or e["importance"] == importance]
                
                return json.dumps({"upcoming_events": filtered_events}, indent=2)
            except Exception as e:
                return f"Error checking economic calendar: {str(e)}"
        
        self.tools.extend([
            analyze_symbol,
            research_market_conditions, 
            analyze_portfolio_correlation,
            backtest_strategy,
            check_economic_calendar
        ])
        
        logger.info(f"Initialized {len(self.tools)} agent tools")
    
    def _initialize_agent(self):
        """Initialize the ReAct agent"""
        if not LANGCHAIN_AVAILABLE or not self.llm:
            logger.warning("Cannot initialize agent - LangChain or LLM not available")
            return
        
        try:
            # Create agent prompt based on personality
            agent_prompt = self._get_personality_prompt()
            
            # Create ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate"
            )
            
            logger.info("ReAct agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            self.agent_executor = None
    
    def _get_personality_prompt(self) -> PromptTemplate:
        """Get personality-specific prompt template"""
        
        base_prompt = """You are SuperNova, an advanced AI financial advisor with expertise in trading, investment analysis, and market research.

PERSONALITY: {personality_description}

CAPABILITIES:
- Technical and fundamental analysis
- Market research and sentiment analysis
- Portfolio optimization and risk management
- Trading strategy development and backtesting
- Economic data interpretation
- Educational explanations of financial concepts

TOOLS AVAILABLE:
{tools}

INSTRUCTIONS:
1. Use tools when you need specific analysis or data
2. Provide detailed, educational explanations
3. Always include appropriate risk disclaimers
4. Adapt your communication style to user expertise level
5. Reference specific data points and analysis when available
6. Be proactive in suggesting follow-up analysis or questions

CONVERSATION CONTEXT:
{conversation_context}

USER QUERY: {input}

Think step by step about how to best help the user:

{agent_scratchpad}"""
        
        personality_descriptions = {
            AgentPersonality.CONSERVATIVE_ADVISOR: "You are a conservative financial advisor focused on capital preservation, risk management, and steady long-term growth. You emphasize diversification, blue-chip investments, and thorough risk assessment.",
            
            AgentPersonality.AGGRESSIVE_TRADER: "You are an aggressive trading specialist focused on growth opportunities, momentum strategies, and higher-risk/higher-reward scenarios. You emphasize technical analysis, market timing, and active portfolio management.",
            
            AgentPersonality.BALANCED_ANALYST: "You are a balanced financial analyst providing objective, data-driven analysis that considers both risk and reward. You emphasize thorough research, diversified strategies, and evidence-based decision making.",
            
            AgentPersonality.EDUCATIONAL_MODE: "You are an educational financial advisor focused on teaching concepts, explaining strategies, and building user knowledge. You emphasize clear explanations, examples, and step-by-step learning.",
            
            AgentPersonality.RESEARCH_ASSISTANT: "You are a research-focused analyst specializing in deep market research, data analysis, and comprehensive reporting. You emphasize thorough investigation, multiple data sources, and detailed findings."
        }
        
        personality_desc = personality_descriptions.get(
            self.personality, 
            personality_descriptions[AgentPersonality.BALANCED_ANALYST]
        )
        
        return PromptTemplate(
            input_variables=["personality_description", "tools", "conversation_context", "input", "agent_scratchpad"],
            template=base_prompt,
            partial_variables={
                "personality_description": personality_desc,
                "tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
            }
        )
    
    async def process_query(self,
                          user_input: str,
                          conversation_id: str,
                          user_id: Optional[int] = None,
                          session_id: Optional[str] = None,
                          streaming_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Process user query with full context and memory integration
        """
        start_time = time.time()
        
        try:
            # Set up conversation context
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                personality=self.personality
            )
            
            # Get memory instances
            conversation_memory = self.memory_manager.get_conversation_memory(conversation_id, user_id)
            user_context = self.memory_manager.get_user_context_memory(user_id) if user_id else None
            financial_context = self.memory_manager.get_financial_context_memory(conversation_id)
            session_memory = self.memory_manager.get_session_memory(session_id) if session_id else None
            
            # Add user message to conversation memory
            conversation_memory.add_message(
                role=ConversationRole.USER,
                content=user_input,
                importance_score=0.8,
                context_tags=self._extract_context_tags(user_input)
            )
            
            # Analyze query type
            query_type = self._classify_query(user_input)
            
            # Get relevant context for LLM
            llm_context = self._build_llm_context(
                context, conversation_memory, user_context, financial_context, session_memory
            )
            
            # Generate response
            if self.agent_executor and LANGCHAIN_AVAILABLE:
                response = await self._generate_agent_response(
                    user_input, llm_context, streaming_callback
                )
            else:
                response = await self._generate_fallback_response(
                    user_input, context, conversation_memory, query_type
                )
            
            # Add assistant response to memory
            conversation_memory.add_message(
                role=ConversationRole.ASSISTANT,
                content=response["content"],
                metadata=response.get("metadata", {}),
                importance_score=0.9,
                context_tags=response.get("context_tags", [])
            )
            
            # Update user interaction patterns
            if user_context:
                user_context.learn_from_interaction(
                    interaction_type=query_type.value,
                    context={
                        "query_length": len(user_input),
                        "response_length": len(response["content"]),
                        "tools_used": response.get("tools_used", []),
                        "query_type": query_type.value
                    }
                )
            
            # Update session state
            if session_memory:
                session_memory.add_flow_step(
                    step_type="query_response",
                    description=f"Processed {query_type.value} query",
                    data={
                        "query_length": len(user_input),
                        "response_time": time.time() - start_time,
                        "tools_used": response.get("tools_used", [])
                    }
                )
            
            # Track performance
            response_time = time.time() - start_time
            self._response_times.append(response_time)
            self._query_counts[query_type.value] = self._query_counts.get(query_type.value, 0) + 1
            
            # Prepare final response
            final_response = {
                "content": response["content"],
                "conversation_id": conversation_id,
                "query_type": query_type.value,
                "personality": self.personality.value,
                "response_time_ms": int(response_time * 1000),
                "tools_used": response.get("tools_used", []),
                "metadata": response.get("metadata", {}),
                "context_summary": conversation_memory.summarize_conversation(),
                "suggestions": response.get("suggestions", [])
            }
            
            # Add disclaimer for financial content
            if query_type in [QueryType.FINANCIAL_ANALYSIS, QueryType.STRATEGY_ADVICE, QueryType.PORTFOLIO_REVIEW]:
                final_response["disclaimer"] = self._get_financial_disclaimer()
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            self._error_counts[type(e).__name__] = self._error_counts.get(type(e).__name__, 0) + 1
            
            return {
                "content": f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or contact support if the issue persists.",
                "conversation_id": conversation_id,
                "query_type": "error",
                "error": str(e),
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def _generate_agent_response(self,
                                     user_input: str,
                                     context: str,
                                     streaming_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Generate response using LangChain agent"""
        
        try:
            # Set up streaming if requested
            callbacks = []
            if streaming_callback:
                callbacks.append(StreamingCallback(streaming_callback))
            
            # Run agent
            result = await self.agent_executor.ainvoke(
                {
                    "input": user_input,
                    "conversation_context": context
                },
                config={"callbacks": callbacks} if callbacks else None
            )
            
            # Extract tools used from agent steps
            tools_used = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, _ = step
                        if hasattr(action, 'tool'):
                            tools_used.append(action.tool)
            
            return {
                "content": result.get("output", "I apologize, but I couldn't generate a proper response."),
                "tools_used": tools_used,
                "metadata": {
                    "agent_steps": len(result.get("intermediate_steps", [])),
                    "agent_mode": "langchain_react"
                }
            }
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            raise e
    
    async def _generate_fallback_response(self,
                                        user_input: str,
                                        context: ConversationContext,
                                        conversation_memory: ConversationMemory,
                                        query_type: QueryType) -> Dict[str, Any]:
        """Generate fallback response when agent is not available"""
        
        # Get conversation history for context
        recent_history = conversation_memory.get_conversation_history(limit=5)
        
        # Simple pattern-based responses
        if query_type == QueryType.FINANCIAL_ANALYSIS:
            if any(symbol in user_input.upper() for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']):
                symbol = next((s for s in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'] if s in user_input.upper()), None)
                response = f"I would analyze {symbol} for you, but I need access to real-time market data. In general, for any stock analysis, I'd look at technical indicators like RSI, MACD, moving averages, as well as fundamental factors like earnings, revenue growth, and market conditions."
            else:
                response = "To provide a proper financial analysis, I would need to access real-time market data and use technical analysis tools. Could you specify which symbol you'd like me to analyze?"
        
        elif query_type == QueryType.STRATEGY_ADVICE:
            response = f"Based on your {context.personality.value} profile, I would typically recommend strategies that align with your risk tolerance. However, I need more information about your current portfolio, investment timeline, and specific goals to provide personalized advice."
        
        elif query_type == QueryType.MARKET_RESEARCH:
            response = "For market research, I would analyze current market sentiment, sector rotations, economic indicators, and institutional flows. Key factors to monitor include Federal Reserve policy, inflation data, earnings trends, and geopolitical developments."
        
        elif query_type == QueryType.EDUCATIONAL:
            response = "I'd be happy to explain financial concepts. What specific topic would you like to learn about? I can cover technical analysis, fundamental analysis, portfolio theory, risk management, or any other financial subject."
        
        else:
            # General chat
            response = f"I understand you're asking about: {user_input[:100]}{'...' if len(user_input) > 100 else ''}. While I'm designed to help with financial analysis and trading strategies, I can certainly discuss this topic. How can I assist you further?"
        
        return {
            "content": response,
            "tools_used": [],
            "metadata": {"fallback_mode": True},
            "suggestions": [
                "Ask about specific stock analysis",
                "Request portfolio optimization advice", 
                "Learn about technical indicators"
            ]
        }
    
    def _classify_query(self, user_input: str) -> QueryType:
        """Classify the type of user query"""
        
        input_lower = user_input.lower()
        
        # Financial analysis keywords
        if any(keyword in input_lower for keyword in [
            'analyze', 'analysis', 'stock', 'price', 'chart', 'technical', 'rsi', 'macd', 'moving average'
        ]):
            return QueryType.FINANCIAL_ANALYSIS
        
        # Strategy advice keywords  
        if any(keyword in input_lower for keyword in [
            'strategy', 'invest', 'buy', 'sell', 'portfolio', 'recommend', 'advice', 'should i'
        ]):
            return QueryType.STRATEGY_ADVICE
        
        # Market research keywords
        if any(keyword in input_lower for keyword in [
            'market', 'sector', 'economy', 'economic', 'sentiment', 'outlook', 'forecast'
        ]):
            return QueryType.MARKET_RESEARCH
        
        # Portfolio review keywords
        if any(keyword in input_lower for keyword in [
            'portfolio', 'holdings', 'diversification', 'correlation', 'risk', 'allocation'
        ]):
            return QueryType.PORTFOLIO_REVIEW
        
        # Educational keywords
        if any(keyword in input_lower for keyword in [
            'what is', 'how does', 'explain', 'learn', 'understand', 'difference between', 'definition'
        ]):
            return QueryType.EDUCATIONAL
        
        return QueryType.GENERAL_CHAT
    
    def _extract_context_tags(self, text: str) -> List[str]:
        """Extract context tags from user input"""
        
        tags = []
        text_lower = text.lower()
        
        # Financial symbols (basic detection)
        symbols = ['aapl', 'msft', 'googl', 'amzn', 'tsla', 'nvda', 'meta', 'nflx']
        for symbol in symbols:
            if symbol in text_lower:
                tags.append(f"symbol_{symbol.upper()}")
        
        # Analysis types
        analysis_types = {
            'technical': ['technical', 'rsi', 'macd', 'chart', 'indicator'],
            'fundamental': ['fundamental', 'earnings', 'revenue', 'valuation'],
            'sentiment': ['sentiment', 'mood', 'bullish', 'bearish'],
            'portfolio': ['portfolio', 'diversification', 'allocation', 'risk']
        }
        
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(analysis_type)
        
        return tags
    
    def _build_llm_context(self,
                          context: ConversationContext,
                          conversation_memory: ConversationMemory,
                          user_context: Optional[UserContextMemory],
                          financial_context: FinancialContextMemory,
                          session_memory: Optional[SessionMemory]) -> str:
        """Build comprehensive context for LLM"""
        
        context_parts = []
        
        # User profile context
        if user_context:
            personalization = user_context.get_personalization_context()
            context_parts.append(f"USER PROFILE: {json.dumps(personalization, indent=2)}")
        
        # Conversation context
        llm_context = conversation_memory.get_context_for_llm(
            max_tokens=2000,
            include_financial_context=True,
            symbol_focus=context.current_symbol
        )
        context_parts.append(llm_context)
        
        # Financial context
        financial_ctx = financial_context.get_relevant_context(
            symbol=context.current_symbol,
            hours_back=6
        )
        if financial_ctx:
            context_parts.append(f"FINANCIAL CONTEXT: {json.dumps(financial_ctx, indent=2)}")
        
        # Session context
        if session_memory:
            session_summary = session_memory.get_session_summary()
            context_parts.append(f"SESSION STATE: {json.dumps(session_summary, indent=2)}")
        
        return "\n\n".join(context_parts)
    
    def _get_financial_disclaimer(self) -> str:
        """Get appropriate financial disclaimer"""
        return (
            "⚠️ IMPORTANT: This analysis is for informational purposes only and should not be considered "
            "personalized investment advice. All investments carry risk of loss. Past performance does not "
            "guarantee future results. Please consult with a qualified financial advisor before making "
            "investment decisions and consider your individual financial situation, risk tolerance, and "
            "investment objectives."
        )
    
    def switch_personality(self, new_personality: AgentPersonality):
        """Switch agent personality"""
        self.personality = new_personality
        logger.info(f"Switched to personality: {new_personality.value}")
        
        # Reinitialize agent with new personality
        if self.agent_executor:
            self._initialize_agent()
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        
        avg_response_time = sum(self._response_times) / len(self._response_times) if self._response_times else 0
        
        return {
            "personality": self.personality.value,
            "llm_available": self.llm is not None,
            "agent_available": self.agent_executor is not None,
            "tools_count": len(self.tools),
            "total_queries": sum(self._query_counts.values()),
            "query_breakdown": self._query_counts.copy(),
            "avg_response_time_ms": int(avg_response_time * 1000),
            "error_counts": self._error_counts.copy(),
            "memory_stats": self.memory_manager.get_memory_stats()
        }

# Agent factory functions
def create_conservative_advisor() -> SuperNovaConversationalAgent:
    """Create conservative financial advisor agent"""
    return SuperNovaConversationalAgent(personality=AgentPersonality.CONSERVATIVE_ADVISOR)

def create_aggressive_trader() -> SuperNovaConversationalAgent:
    """Create aggressive trader agent"""
    return SuperNovaConversationalAgent(personality=AgentPersonality.AGGRESSIVE_TRADER)

def create_balanced_analyst() -> SuperNovaConversationalAgent:
    """Create balanced analyst agent"""
    return SuperNovaConversationalAgent(personality=AgentPersonality.BALANCED_ANALYST)

def create_educational_agent() -> SuperNovaConversationalAgent:
    """Create educational mode agent"""
    return SuperNovaConversationalAgent(personality=AgentPersonality.EDUCATIONAL_MODE)

def create_research_assistant() -> SuperNovaConversationalAgent:
    """Create research assistant agent"""
    return SuperNovaConversationalAgent(personality=AgentPersonality.RESEARCH_ASSISTANT)

# Global agent instance (default balanced analyst)
default_agent = SuperNovaConversationalAgent()

# Convenience functions
async def chat(user_input: str,
              conversation_id: str,
              user_id: Optional[int] = None,
              personality: Optional[AgentPersonality] = None,
              streaming_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Convenient chat function for simple interactions
    """
    agent = default_agent
    
    if personality and personality != agent.personality:
        agent.switch_personality(personality)
    
    return await agent.process_query(
        user_input=user_input,
        conversation_id=conversation_id,
        user_id=user_id,
        streaming_callback=streaming_callback
    )

async def analyze_financial_query(symbol: str,
                                 query: str,
                                 conversation_id: str,
                                 user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Specialized function for financial analysis queries
    """
    # Set up financial context
    financial_context = memory_manager.get_financial_context_memory(conversation_id)
    
    # Add symbol to context
    financial_context.add_market_context(
        context_type="query_symbol",
        data={"symbol": symbol, "query": query},
        symbol=symbol
    )
    
    # Process with balanced analyst personality
    agent = SuperNovaConversationalAgent(personality=AgentPersonality.BALANCED_ANALYST)
    
    return await agent.process_query(
        user_input=f"Analyze {symbol}: {query}",
        conversation_id=conversation_id,
        user_id=user_id
    )
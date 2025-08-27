"""
SuperNova Conversation Memory Management System

Provides comprehensive memory management for conversational agents including:
- Conversation history tracking
- User context and preferences
- Financial context accumulation
- Session-based temporary memory
- Memory summarization and optimization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, desc, text, func
import json
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

from .config import settings
from .db import SessionLocal

logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    """Types of memory entries"""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    FINANCIAL_CONTEXT = "financial_context"
    USER_PREFERENCE = "user_preference"
    MARKET_CONTEXT = "market_context"
    SESSION_STATE = "session_state"

class ConversationRole(str, Enum):
    """Conversation participant roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class MemoryEntry:
    """Individual memory entry"""
    id: Optional[int] = None
    conversation_id: str = ""
    user_id: Optional[int] = None
    role: ConversationRole = ConversationRole.USER
    content: str = ""
    memory_type: MemoryType = MemoryType.USER_MESSAGE
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    importance_score: float = 0.5
    context_tags: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.context_tags is None:
            self.context_tags = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ConversationMemory:
    """
    Manages conversation history and context for individual conversations
    """
    
    def __init__(self, conversation_id: str, user_id: Optional[int] = None, max_history: int = 100):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.max_history = max_history
        self.memory_entries: List[MemoryEntry] = []
        self._context_summary: Optional[str] = None
        self._last_summary_at: Optional[datetime] = None
        
        # Load existing conversation history
        self._load_conversation_history()
    
    def add_message(self, 
                   role: ConversationRole, 
                   content: str,
                   memory_type: MemoryType = None,
                   metadata: Dict[str, Any] = None,
                   importance_score: float = 0.5,
                   context_tags: List[str] = None) -> MemoryEntry:
        """Add a message to conversation memory"""
        
        if memory_type is None:
            memory_type = {
                ConversationRole.USER: MemoryType.USER_MESSAGE,
                ConversationRole.ASSISTANT: MemoryType.ASSISTANT_MESSAGE,
                ConversationRole.SYSTEM: MemoryType.SYSTEM_MESSAGE
            }.get(role, MemoryType.USER_MESSAGE)
        
        entry = MemoryEntry(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            role=role,
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            timestamp=datetime.now(),
            importance_score=importance_score,
            context_tags=context_tags or []
        )
        
        self.memory_entries.append(entry)
        
        # Persist to database
        self._save_memory_entry(entry)
        
        # Trigger memory optimization if needed
        if len(self.memory_entries) > self.max_history:
            self._optimize_memory()
        
        return entry
    
    def add_financial_context(self, 
                            context_type: str,
                            content: Dict[str, Any],
                            symbol: Optional[str] = None,
                            importance_score: float = 0.7) -> MemoryEntry:
        """Add financial context to memory"""
        
        metadata = {
            "context_type": context_type,
            "symbol": symbol
        }
        
        context_tags = ["financial", context_type]
        if symbol:
            context_tags.append(f"symbol_{symbol}")
        
        return self.add_message(
            role=ConversationRole.SYSTEM,
            content=json.dumps(content),
            memory_type=MemoryType.FINANCIAL_CONTEXT,
            metadata=metadata,
            importance_score=importance_score,
            context_tags=context_tags
        )
    
    def get_conversation_history(self, 
                               limit: int = 50,
                               include_system: bool = False,
                               context_window_minutes: int = None) -> List[MemoryEntry]:
        """Get conversation history with filtering options"""
        
        entries = self.memory_entries.copy()
        
        # Filter by time window if specified
        if context_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=context_window_minutes)
            entries = [e for e in entries if e.timestamp >= cutoff_time]
        
        # Filter system messages if requested
        if not include_system:
            entries = [e for e in entries if e.role != ConversationRole.SYSTEM]
        
        # Sort by timestamp and limit
        entries.sort(key=lambda x: x.timestamp)
        return entries[-limit:] if limit else entries
    
    def get_financial_context(self, 
                            symbol: Optional[str] = None,
                            context_type: Optional[str] = None,
                            hours_back: int = 24) -> List[MemoryEntry]:
        """Get financial context entries"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        entries = [
            e for e in self.memory_entries
            if (e.memory_type == MemoryType.FINANCIAL_CONTEXT and
                e.timestamp >= cutoff_time)
        ]
        
        # Filter by symbol if specified
        if symbol:
            entries = [e for e in entries if f"symbol_{symbol}" in e.context_tags]
        
        # Filter by context type if specified
        if context_type:
            entries = [e for e in entries if e.metadata.get("context_type") == context_type]
        
        return sorted(entries, key=lambda x: x.timestamp, reverse=True)
    
    def summarize_conversation(self, force_update: bool = False) -> str:
        """Generate or retrieve conversation summary"""
        
        # Check if summary needs update
        if (not force_update and 
            self._context_summary and 
            self._last_summary_at and
            (datetime.now() - self._last_summary_at).total_seconds() < 1800):  # 30 minutes
            return self._context_summary
        
        # Get recent conversation history
        recent_entries = self.get_conversation_history(limit=20, include_system=False)
        
        if not recent_entries:
            return "No conversation history available."
        
        # Simple summarization (can be enhanced with LLM)
        summary_parts = []
        user_messages = [e for e in recent_entries if e.role == ConversationRole.USER]
        assistant_messages = [e for e in recent_entries if e.role == ConversationRole.ASSISTANT]
        
        summary_parts.append(f"Conversation contains {len(user_messages)} user messages and {len(assistant_messages)} assistant responses.")
        
        # Identify key topics
        all_tags = []
        for entry in recent_entries:
            all_tags.extend(entry.context_tags)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_topics = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_topics:
            topics_str = ", ".join([f"{topic} ({count})" for topic, count in top_topics])
            summary_parts.append(f"Key discussion topics: {topics_str}")
        
        # Get financial symbols discussed
        financial_entries = [e for e in recent_entries if "financial" in e.context_tags]
        symbols = set()
        for entry in financial_entries:
            for tag in entry.context_tags:
                if tag.startswith("symbol_"):
                    symbols.add(tag.replace("symbol_", ""))
        
        if symbols:
            summary_parts.append(f"Financial instruments discussed: {', '.join(symbols)}")
        
        self._context_summary = ". ".join(summary_parts)
        self._last_summary_at = datetime.now()
        
        return self._context_summary
    
    def get_context_for_llm(self, 
                          max_tokens: int = 4000,
                          include_financial_context: bool = True,
                          symbol_focus: Optional[str] = None) -> str:
        """Get formatted context string for LLM consumption"""
        
        context_parts = []
        
        # Add conversation summary
        summary = self.summarize_conversation()
        context_parts.append(f"CONVERSATION SUMMARY:\n{summary}\n")
        
        # Add recent conversation history
        recent_history = self.get_conversation_history(limit=10, include_system=False)
        if recent_history:
            context_parts.append("RECENT CONVERSATION:")
            for entry in recent_history:
                role_str = entry.role.value.upper()
                timestamp_str = entry.timestamp.strftime("%H:%M")
                context_parts.append(f"{timestamp_str} {role_str}: {entry.content}")
            context_parts.append("")
        
        # Add financial context if requested
        if include_financial_context:
            financial_context = self.get_financial_context(symbol=symbol_focus, hours_back=6)
            if financial_context:
                context_parts.append("RELEVANT FINANCIAL CONTEXT:")
                for entry in financial_context[:5]:  # Limit to most recent 5
                    context_type = entry.metadata.get("context_type", "unknown")
                    symbol = entry.metadata.get("symbol", "")
                    symbol_str = f" ({symbol})" if symbol else ""
                    context_parts.append(f"- {context_type}{symbol_str}: {entry.content}")
                context_parts.append("")
        
        # Join and truncate if needed
        full_context = "\n".join(context_parts)
        
        # Simple token approximation (4 chars per token)
        if len(full_context) > max_tokens * 4:
            # Truncate while preserving structure
            target_length = max_tokens * 4
            truncated = full_context[:target_length]
            # Find last complete line
            last_newline = truncated.rfind("\n")
            if last_newline > 0:
                truncated = truncated[:last_newline]
            full_context = truncated + "\n... [context truncated]"
        
        return full_context
    
    def _load_conversation_history(self):
        """Load existing conversation history from database"""
        # Implementation would query conversation_messages table
        # For now, starting with empty memory
        pass
    
    def _save_memory_entry(self, entry: MemoryEntry):
        """Save memory entry to database"""
        # Implementation would save to conversation_messages table
        # For now, just log
        logger.info(f"Saving memory entry: {entry.role.value} - {entry.content[:50]}...")
    
    def _optimize_memory(self):
        """Optimize memory by removing less important old entries"""
        if len(self.memory_entries) <= self.max_history:
            return
        
        # Sort by importance and recency
        entries_with_scores = []
        now = datetime.now()
        
        for entry in self.memory_entries:
            # Calculate composite score (importance + recency)
            age_hours = (now - entry.timestamp).total_seconds() / 3600
            recency_score = max(0, 1 - age_hours / 168)  # Decay over week
            composite_score = (entry.importance_score * 0.7) + (recency_score * 0.3)
            entries_with_scores.append((composite_score, entry))
        
        # Keep top entries by score
        entries_with_scores.sort(reverse=True)
        self.memory_entries = [entry for _, entry in entries_with_scores[:self.max_history]]
        
        logger.info(f"Optimized memory: kept {len(self.memory_entries)} entries")

class UserContextMemory:
    """
    Manages user profile, preferences, and learning over time
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.preferences: Dict[str, Any] = {}
        self.financial_profile: Dict[str, Any] = {}
        self.interaction_patterns: Dict[str, Any] = {}
        self.learned_preferences: Dict[str, Any] = {}
        
        self._load_user_context()
    
    def update_preference(self, key: str, value: Any, confidence: float = 1.0):
        """Update user preference with confidence score"""
        self.preferences[key] = {
            "value": value,
            "confidence": confidence,
            "updated_at": datetime.now().isoformat()
        }
        self._save_preferences()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference value"""
        pref = self.preferences.get(key)
        if pref:
            return pref.get("value", default)
        return default
    
    def learn_from_interaction(self, 
                              interaction_type: str,
                              context: Dict[str, Any],
                              user_feedback: Optional[str] = None):
        """Learn from user interaction patterns"""
        
        pattern_key = f"{interaction_type}_patterns"
        if pattern_key not in self.interaction_patterns:
            self.interaction_patterns[pattern_key] = []
        
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "feedback": user_feedback
        }
        
        self.interaction_patterns[pattern_key].append(interaction_record)
        
        # Keep only recent patterns (last 100)
        if len(self.interaction_patterns[pattern_key]) > 100:
            self.interaction_patterns[pattern_key] = self.interaction_patterns[pattern_key][-100:]
        
        # Update learned preferences based on patterns
        self._update_learned_preferences(interaction_type, context, user_feedback)
        
        self._save_interaction_patterns()
    
    def get_personalization_context(self) -> Dict[str, Any]:
        """Get context for personalizing responses"""
        return {
            "preferences": self.preferences,
            "financial_profile": self.financial_profile,
            "learned_preferences": self.learned_preferences,
            "interaction_summary": self._summarize_interactions()
        }
    
    def _load_user_context(self):
        """Load user context from database"""
        # Implementation would query user_context tables
        # For now, initialize empty
        pass
    
    def _save_preferences(self):
        """Save preferences to database"""
        logger.info(f"Saving preferences for user {self.user_id}")
    
    def _save_interaction_patterns(self):
        """Save interaction patterns to database"""
        logger.info(f"Saving interaction patterns for user {self.user_id}")
    
    def _update_learned_preferences(self, interaction_type: str, context: Dict[str, Any], feedback: Optional[str]):
        """Update learned preferences based on interaction"""
        # Simple learning logic - can be enhanced with ML
        
        if interaction_type == "financial_query" and feedback:
            symbol = context.get("symbol")
            if symbol and "helpful" in feedback.lower():
                # User found analysis helpful for this symbol
                symbol_prefs = self.learned_preferences.get("preferred_symbols", {})
                symbol_prefs[symbol] = symbol_prefs.get(symbol, 0) + 1
                self.learned_preferences["preferred_symbols"] = symbol_prefs
            
            analysis_type = context.get("analysis_type")
            if analysis_type and "detailed" in feedback.lower():
                # User prefers detailed analysis
                self.learned_preferences["analysis_detail_level"] = "detailed"
    
    def _summarize_interactions(self) -> Dict[str, Any]:
        """Summarize interaction patterns"""
        summary = {
            "total_interactions": sum(len(patterns) for patterns in self.interaction_patterns.values()),
            "most_common_interaction": None,
            "recent_activity": None
        }
        
        # Find most common interaction type
        interaction_counts = {}
        for pattern_type, patterns in self.interaction_patterns.items():
            interaction_counts[pattern_type] = len(patterns)
        
        if interaction_counts:
            summary["most_common_interaction"] = max(interaction_counts, key=interaction_counts.get)
        
        return summary

class FinancialContextMemory:
    """
    Manages financial data context, market conditions, and analysis results
    """
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.market_context: Dict[str, Any] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.portfolio_context: Dict[str, Any] = {}
        self.watchlist_context: List[str] = []
        
    def add_market_context(self, 
                          context_type: str,
                          data: Dict[str, Any],
                          symbol: Optional[str] = None):
        """Add market context data"""
        
        key = f"{context_type}_{symbol}" if symbol else context_type
        
        self.market_context[key] = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }
    
    def add_analysis_result(self,
                          symbol: str,
                          analysis_type: str,
                          result: Dict[str, Any]):
        """Add analysis result to history"""
        
        analysis_record = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.analysis_history.append(analysis_record)
        
        # Keep last 50 analyses
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
    
    def get_relevant_context(self, 
                           symbol: Optional[str] = None,
                           hours_back: int = 6) -> Dict[str, Any]:
        """Get relevant financial context for current query"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        relevant_context = {
            "market_data": {},
            "recent_analyses": [],
            "portfolio_info": self.portfolio_context.copy()
        }
        
        # Filter market context by time and symbol
        for key, context in self.market_context.items():
            context_time = datetime.fromisoformat(context["timestamp"])
            if context_time >= cutoff_time:
                if symbol is None or context["symbol"] == symbol:
                    relevant_context["market_data"][key] = context
        
        # Filter analysis history
        for analysis in self.analysis_history:
            analysis_time = datetime.fromisoformat(analysis["timestamp"])
            if analysis_time >= cutoff_time:
                if symbol is None or analysis["symbol"] == symbol:
                    relevant_context["recent_analyses"].append(analysis)
        
        return relevant_context
    
    def update_portfolio_context(self, portfolio_data: Dict[str, Any]):
        """Update portfolio context"""
        self.portfolio_context.update(portfolio_data)
        self.portfolio_context["last_updated"] = datetime.now().isoformat()
    
    def update_watchlist(self, symbols: List[str]):
        """Update watchlist context"""
        self.watchlist_context = symbols

class SessionMemory:
    """
    Manages temporary session state and conversation flow
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_state: Dict[str, Any] = {}
        self.active_context: Dict[str, Any] = {}
        self.conversation_flow: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def set_state(self, key: str, value: Any):
        """Set session state variable"""
        self.session_state[key] = value
        self.last_activity = datetime.now()
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get session state variable"""
        return self.session_state.get(key, default)
    
    def update_active_context(self, context_update: Dict[str, Any]):
        """Update active conversation context"""
        self.active_context.update(context_update)
        self.last_activity = datetime.now()
    
    def add_flow_step(self, 
                     step_type: str,
                     description: str,
                     data: Dict[str, Any] = None):
        """Add step to conversation flow"""
        
        flow_step = {
            "step_type": step_type,
            "description": description,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_flow.append(flow_step)
        
        # Keep last 20 flow steps
        if len(self.conversation_flow) > 20:
            self.conversation_flow = self.conversation_flow[-20:]
        
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session is expired"""
        return (datetime.now() - self.last_activity).total_seconds() > (timeout_hours * 3600)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "state_keys": list(self.session_state.keys()),
            "active_context_keys": list(self.active_context.keys()),
            "flow_steps_count": len(self.conversation_flow),
            "is_expired": self.is_expired()
        }

class MemoryManager:
    """
    Central memory management coordinator
    """
    
    def __init__(self):
        self._conversations: Dict[str, ConversationMemory] = {}
        self._user_contexts: Dict[int, UserContextMemory] = {}
        self._financial_contexts: Dict[str, FinancialContextMemory] = {}
        self._sessions: Dict[str, SessionMemory] = {}
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = datetime.now()
    
    def get_conversation_memory(self, conversation_id: str, user_id: Optional[int] = None) -> ConversationMemory:
        """Get or create conversation memory"""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationMemory(conversation_id, user_id)
        return self._conversations[conversation_id]
    
    def get_user_context_memory(self, user_id: int) -> UserContextMemory:
        """Get or create user context memory"""
        if user_id not in self._user_contexts:
            self._user_contexts[user_id] = UserContextMemory(user_id)
        return self._user_contexts[user_id]
    
    def get_financial_context_memory(self, conversation_id: str) -> FinancialContextMemory:
        """Get or create financial context memory"""
        if conversation_id not in self._financial_contexts:
            self._financial_contexts[conversation_id] = FinancialContextMemory(conversation_id)
        return self._financial_contexts[conversation_id]
    
    def get_session_memory(self, session_id: str) -> SessionMemory:
        """Get or create session memory"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id)
        return self._sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions and memory"""
        current_time = datetime.now()
        
        # Only run cleanup periodically
        if (current_time - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return
        
        # Clean up expired sessions
        expired_sessions = []
        for session_id, session in self._sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        # Clean up old conversation memories (keep only active ones)
        # This would be enhanced with database-based cleanup
        
        self._last_cleanup = current_time
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "active_conversations": len(self._conversations),
            "active_user_contexts": len(self._user_contexts),
            "active_financial_contexts": len(self._financial_contexts),
            "active_sessions": len(self._sessions),
            "memory_manager_uptime": (datetime.now() - self._last_cleanup).total_seconds()
        }

# Global memory manager instance
memory_manager = MemoryManager()

# Convenience functions for common operations
def get_conversation_memory(conversation_id: str, user_id: Optional[int] = None) -> ConversationMemory:
    """Get conversation memory instance"""
    return memory_manager.get_conversation_memory(conversation_id, user_id)

def get_user_context_memory(user_id: int) -> UserContextMemory:
    """Get user context memory instance"""
    return memory_manager.get_user_context_memory(user_id)

def get_financial_context_memory(conversation_id: str) -> FinancialContextMemory:
    """Get financial context memory instance"""
    return memory_manager.get_financial_context_memory(conversation_id)

def get_session_memory(session_id: str) -> SessionMemory:
    """Get session memory instance"""
    return memory_manager.get_session_memory(session_id)
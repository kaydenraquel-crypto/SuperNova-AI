"""
Integrated LLM Service with Intelligent Fallback System

This module provides:
- Unified LLM service interface
- Intelligent provider routing and fallback
- Automatic error recovery and retry logic  
- Performance optimization and load balancing
- Cost-aware provider selection
- Comprehensive monitoring and analytics
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta

# Core imports
from .config import settings
from .llm_key_manager import get_key_manager, ProviderType
from .llm_provider_manager import get_provider_manager, RequestPriority, ProviderStatus
from .llm_cost_tracker import get_cost_tracker, record_llm_usage

# LangChain imports
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ResponseQuality(str, Enum):
    """Response quality levels"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class TaskComplexity(str, Enum):
    """Task complexity levels for provider selection"""
    SIMPLE = "simple"      # Basic Q&A, simple chat
    MODERATE = "moderate"  # Analysis, summaries
    COMPLEX = "complex"    # Deep analysis, reasoning
    CRITICAL = "critical"  # High-stakes decisions

@dataclass
class LLMRequest:
    """Structured LLM request"""
    messages: Union[str, List[BaseMessage]]
    task_type: str = "chat"
    priority: RequestPriority = RequestPriority.MEDIUM
    quality: ResponseQuality = ResponseQuality.STANDARD
    complexity: TaskComplexity = TaskComplexity.MODERATE
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    streaming: bool = False
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    provider: str
    model: str
    tokens_used: int
    cost: float
    response_time: float
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProviderSelectionStrategy:
    """Provider selection strategies based on request characteristics"""
    
    @staticmethod
    def select_by_quality(request: LLMRequest, available_providers: List[ProviderType]) -> List[ProviderType]:
        """Select providers based on quality requirements"""
        
        if request.quality == ResponseQuality.ENTERPRISE:
            # Prefer premium providers for enterprise quality
            priority_order = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OLLAMA, ProviderType.HUGGINGFACE]
        elif request.quality == ResponseQuality.PREMIUM:
            priority_order = [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.OLLAMA, ProviderType.HUGGINGFACE]
        elif request.quality == ResponseQuality.STANDARD:
            priority_order = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OLLAMA, ProviderType.HUGGINGFACE]
        else:  # BASIC
            # For basic quality, prefer cost-effective options
            priority_order = [ProviderType.OLLAMA, ProviderType.HUGGINGFACE, ProviderType.OPENAI, ProviderType.ANTHROPIC]
        
        return [p for p in priority_order if p in available_providers]
    
    @staticmethod
    def select_by_complexity(request: LLMRequest, available_providers: List[ProviderType]) -> List[ProviderType]:
        """Select providers based on task complexity"""
        
        if request.complexity == TaskComplexity.CRITICAL:
            # For critical tasks, use most reliable providers
            return [p for p in [ProviderType.OPENAI, ProviderType.ANTHROPIC] if p in available_providers]
        elif request.complexity == TaskComplexity.COMPLEX:
            # Complex tasks need capable models
            priority_order = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OLLAMA]
        elif request.complexity == TaskComplexity.MODERATE:
            # Moderate complexity - balanced approach
            priority_order = [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.OLLAMA, ProviderType.HUGGINGFACE]
        else:  # SIMPLE
            # Simple tasks can use any provider
            priority_order = [ProviderType.OLLAMA, ProviderType.HUGGINGFACE, ProviderType.OPENAI, ProviderType.ANTHROPIC]
        
        return [p for p in priority_order if p in available_providers]
    
    @staticmethod
    def select_by_cost(request: LLMRequest, available_providers: List[ProviderType]) -> List[ProviderType]:
        """Select providers based on cost optimization"""
        
        # Order by typical cost (free/local first, then by actual cost)
        cost_order = [
            ProviderType.OLLAMA,      # Free (local)
            ProviderType.HUGGINGFACE, # Free tier
            ProviderType.ANTHROPIC,   # Moderate cost
            ProviderType.OPENAI       # Higher cost
        ]
        
        return [p for p in cost_order if p in available_providers]

class IntelligentFallbackSystem:
    """Intelligent fallback system with learning capabilities"""
    
    def __init__(self):
        self.failure_patterns: Dict[str, List[datetime]] = {}
        self.success_patterns: Dict[str, List[datetime]] = {}
        self.provider_recovery_times: Dict[ProviderType, datetime] = {}
        self.adaptive_timeouts: Dict[ProviderType, float] = {}
        
    def record_failure(self, provider: ProviderType, error_type: str, request_type: str):
        """Record provider failure for pattern analysis"""
        pattern_key = f"{provider.value}:{error_type}:{request_type}"
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = []
        
        self.failure_patterns[pattern_key].append(datetime.utcnow())
        
        # Keep only recent failures (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.failure_patterns[pattern_key] = [
            ts for ts in self.failure_patterns[pattern_key] if ts > cutoff
        ]
    
    def record_success(self, provider: ProviderType, request_type: str, response_time: float):
        """Record provider success for pattern analysis"""
        pattern_key = f"{provider.value}:{request_type}"
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = []
        
        self.success_patterns[pattern_key].append(datetime.utcnow())
        
        # Update adaptive timeout
        current_timeout = self.adaptive_timeouts.get(provider, 30.0)
        # Exponential moving average of response times
        self.adaptive_timeouts[provider] = current_timeout * 0.8 + response_time * 1.2 + 5.0
        
        # Clear recovery time if provider is working
        if provider in self.provider_recovery_times:
            del self.provider_recovery_times[provider]
    
    def should_skip_provider(self, provider: ProviderType, request_type: str) -> bool:
        """Determine if provider should be skipped based on failure patterns"""
        
        # Check if provider is in recovery cooldown
        if provider in self.provider_recovery_times:
            recovery_time = self.provider_recovery_times[provider]
            if datetime.utcnow() - recovery_time < timedelta(minutes=15):
                return True
        
        # Analyze failure patterns
        error_types = ["timeout", "api_error", "rate_limit", "service_unavailable"]
        recent_failures = 0
        
        for error_type in error_types:
            pattern_key = f"{provider.value}:{error_type}:{request_type}"
            if pattern_key in self.failure_patterns:
                # Count failures in last hour
                cutoff = datetime.utcnow() - timedelta(hours=1)
                recent_failures += len([
                    ts for ts in self.failure_patterns[pattern_key] if ts > cutoff
                ])
        
        # Skip if too many recent failures
        return recent_failures >= 3
    
    def get_adaptive_timeout(self, provider: ProviderType) -> float:
        """Get adaptive timeout for provider"""
        return self.adaptive_timeouts.get(provider, 30.0)
    
    def mark_provider_recovery_needed(self, provider: ProviderType):
        """Mark provider as needing recovery time"""
        self.provider_recovery_times[provider] = datetime.utcnow()

class LLMService:
    """
    Unified LLM service with intelligent routing and comprehensive monitoring
    """
    
    def __init__(self):
        """Initialize LLM service"""
        self.key_manager = get_key_manager()
        self.provider_manager = get_provider_manager()
        self.cost_tracker = get_cost_tracker()
        self.fallback_system = IntelligentFallbackSystem()
        
        # Service configuration
        self.max_retry_attempts = getattr(settings, 'LLM_MAX_RETRIES', 3)
        self.default_timeout = getattr(settings, 'LLM_TIMEOUT', 60)
        self.enable_fallback = getattr(settings, 'LLM_FALLBACK_ENABLED', True)
        self.enable_cost_optimization = getattr(settings, 'LLM_COST_OPTIMIZATION', True)
        
        # Performance tracking
        self.request_count = 0
        self.total_cost = 0.0
        self.average_response_time = 0.0
        self.success_rate = 100.0
        
        # Circuit breaker pattern
        self.circuit_breakers: Dict[ProviderType, Dict[str, Any]] = {}
        
        logger.info("LLM Service initialized with intelligent fallback system")
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response with intelligent provider selection and fallback
        """
        start_time = time.time()
        self.request_count += 1
        
        # Validate request
        if not request.messages:
            return LLMResponse(
                content="",
                provider="none",
                model="none",
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                success=False,
                error_message="Empty request"
            )
        
        # Convert string to message format
        if isinstance(request.messages, str):
            messages = [HumanMessage(content=request.messages)]
        else:
            messages = request.messages
        
        # Get available providers
        available_providers = self._get_available_providers()
        if not available_providers:
            return LLMResponse(
                content="",
                provider="none",
                model="none", 
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                success=False,
                error_message="No available providers"
            )
        
        # Select provider strategy based on request
        provider_candidates = self._select_providers(request, available_providers)
        
        # Attempt generation with fallback
        last_error = None
        for attempt, provider in enumerate(provider_candidates):
            
            # Check circuit breaker
            if self._is_circuit_open(provider, request.task_type):
                logger.debug(f"Circuit breaker open for {provider.value}")
                continue
            
            # Check intelligent fallback system
            if self.fallback_system.should_skip_provider(provider, request.task_type):
                logger.debug(f"Intelligent fallback skipping {provider.value}")
                continue
            
            try:
                # Generate response using provider
                response = await self._generate_with_provider(provider, messages, request)
                
                if response.success:
                    # Record success
                    self.fallback_system.record_success(provider, request.task_type, response.response_time)
                    self._record_circuit_success(provider, request.task_type)
                    
                    # Update service metrics
                    self._update_success_metrics(response.cost, response.response_time)
                    
                    return response
                else:
                    # Record failure but continue to next provider
                    last_error = response.error_message
                    self.fallback_system.record_failure(provider, "generation_error", request.task_type)
                    self._record_circuit_failure(provider, request.task_type)
                    
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                logger.error(f"Provider {provider.value} failed: {error_msg}")
                
                # Record failure
                error_type = self._classify_error(error_msg)
                self.fallback_system.record_failure(provider, error_type, request.task_type)
                self._record_circuit_failure(provider, request.task_type)
                
                # If rate limited, mark for recovery
                if "rate limit" in error_msg.lower():
                    self.fallback_system.mark_provider_recovery_needed(provider)
        
        # All providers failed
        response_time = time.time() - start_time
        self._update_failure_metrics(response_time)
        
        # Try fallback response if enabled
        if self.enable_fallback and getattr(settings, 'LLM_FALLBACK_SIMPLE_RATIONALE', True):
            fallback_content = self._generate_fallback_response(messages, request)
            return LLMResponse(
                content=fallback_content,
                provider="fallback",
                model="simple",
                tokens_used=0,
                cost=0.0,
                response_time=response_time,
                success=True,
                metadata={"fallback_reason": last_error}
            )
        
        return LLMResponse(
            content="",
            provider="none",
            model="none",
            tokens_used=0,
            cost=0.0,
            response_time=response_time,
            success=False,
            error_message=f"All providers failed. Last error: {last_error}"
        )
    
    def _get_available_providers(self) -> List[ProviderType]:
        """Get list of available providers"""
        available = []
        
        for provider_type in ProviderType:
            provider_config = self.provider_manager.provider_configs.get(provider_type)
            provider_status = self.provider_manager.provider_status.get(provider_type)
            
            if (provider_config and provider_config.enabled and 
                provider_status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]):
                available.append(provider_type)
        
        return available
    
    def _select_providers(self, request: LLMRequest, available_providers: List[ProviderType]) -> List[ProviderType]:
        """Select and order providers based on request characteristics"""
        
        # Apply multiple selection strategies
        quality_selection = ProviderSelectionStrategy.select_by_quality(request, available_providers)
        complexity_selection = ProviderSelectionStrategy.select_by_complexity(request, available_providers)
        
        # If cost optimization is enabled and quality allows, prefer cost-effective providers
        if (self.enable_cost_optimization and 
            request.quality in [ResponseQuality.BASIC, ResponseQuality.STANDARD]):
            cost_selection = ProviderSelectionStrategy.select_by_cost(request, available_providers)
            # Merge cost preferences with quality/complexity
            final_selection = []
            for provider in cost_selection:
                if provider in quality_selection or provider in complexity_selection:
                    final_selection.append(provider)
            # Add remaining providers
            for provider in quality_selection:
                if provider not in final_selection:
                    final_selection.append(provider)
        else:
            # Prioritize quality and complexity over cost
            final_selection = quality_selection
            for provider in complexity_selection:
                if provider not in final_selection:
                    final_selection.append(provider)
        
        # Add any remaining available providers as fallbacks
        for provider in available_providers:
            if provider not in final_selection:
                final_selection.append(provider)
        
        logger.debug(f"Selected provider order: {[p.value for p in final_selection]}")
        return final_selection
    
    async def _generate_with_provider(self, provider: ProviderType, messages: List[BaseMessage], request: LLMRequest) -> LLMResponse:
        """Generate response using specific provider"""
        start_time = time.time()
        
        try:
            # Get adaptive timeout
            timeout = request.timeout or self.fallback_system.get_adaptive_timeout(provider)
            
            # Set up streaming callback if needed
            streaming_callback = None
            if request.streaming and hasattr(request, 'streaming_callback'):
                streaming_callback = request.streaming_callback
            
            # Generate response using provider manager
            content = await asyncio.wait_for(
                self.provider_manager.generate_response(
                    messages=messages,
                    task_type=request.task_type,
                    priority=request.priority,
                    streaming_callback=streaming_callback,
                    preferred_provider=provider,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                timeout=timeout
            )
            
            if not content:
                return LLMResponse(
                    content="",
                    provider=provider.value,
                    model="unknown",
                    tokens_used=0,
                    cost=0.0,
                    response_time=time.time() - start_time,
                    success=False,
                    error_message="Empty response from provider"
                )
            
            # Estimate token usage
            input_tokens = self._estimate_tokens(" ".join([msg.content for msg in messages if hasattr(msg, 'content')]))
            output_tokens = self._estimate_tokens(content)
            
            # Get cost rates
            provider_config = self.provider_manager.provider_configs[provider]
            input_cost_rate = provider_config.cost_per_1k_input_tokens
            output_cost_rate = provider_config.cost_per_1k_output_tokens
            
            # Calculate cost
            total_cost = (input_tokens / 1000 * input_cost_rate + 
                         output_tokens / 1000 * output_cost_rate)
            
            # Record usage
            if self.cost_tracker.enabled:
                await self.cost_tracker.record_usage(
                    provider=provider,
                    model=provider_config.models.get(request.task_type, "unknown"),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_cost_per_1k=input_cost_rate,
                    output_cost_per_1k=output_cost_rate,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_type=request.task_type,
                    response_time=time.time() - start_time,
                    success=True,
                    metadata=request.metadata
                )
            
            return LLMResponse(
                content=content,
                provider=provider.value,
                model=provider_config.models.get(request.task_type, "unknown"),
                tokens_used=input_tokens + output_tokens,
                cost=total_cost,
                response_time=time.time() - start_time,
                success=True,
                quality_score=self._assess_response_quality(content, request),
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "timeout_used": timeout
                }
            )
            
        except asyncio.TimeoutError:
            return LLMResponse(
                content="",
                provider=provider.value,
                model="unknown",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            return LLMResponse(
                content="",
                provider=provider.value,
                model="unknown",
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 characters per token average)"""
        return max(1, len(text) // 4)
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type for pattern analysis"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "api" in error_lower and ("key" in error_lower or "auth" in error_lower):
            return "auth_error"
        elif "service unavailable" in error_lower or "502" in error_lower or "503" in error_lower:
            return "service_unavailable"
        elif "quota" in error_lower or "limit" in error_lower:
            return "quota_exceeded"
        else:
            return "api_error"
    
    def _is_circuit_open(self, provider: ProviderType, request_type: str) -> bool:
        """Check if circuit breaker is open for provider/request type"""
        circuit_key = f"{provider.value}:{request_type}"
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half_open
            }
        
        circuit = self.circuit_breakers[circuit_key]
        
        # If circuit is open, check if it should move to half-open
        if circuit["state"] == "open":
            if circuit["last_failure"] and datetime.utcnow() - circuit["last_failure"] > timedelta(minutes=5):
                circuit["state"] = "half_open"
                circuit["failures"] = 0
                return False
            return True
        
        return False
    
    def _record_circuit_failure(self, provider: ProviderType, request_type: str):
        """Record circuit breaker failure"""
        circuit_key = f"{provider.value}:{request_type}"
        
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"
            }
        
        circuit = self.circuit_breakers[circuit_key]
        circuit["failures"] += 1
        circuit["last_failure"] = datetime.utcnow()
        
        # Open circuit if too many failures
        if circuit["failures"] >= 5:
            circuit["state"] = "open"
            logger.warning(f"Circuit breaker opened for {provider.value}:{request_type}")
    
    def _record_circuit_success(self, provider: ProviderType, request_type: str):
        """Record circuit breaker success"""
        circuit_key = f"{provider.value}:{request_type}"
        
        if circuit_key in self.circuit_breakers:
            circuit = self.circuit_breakers[circuit_key]
            if circuit["state"] == "half_open":
                # Success in half-open state closes the circuit
                circuit["state"] = "closed"
                circuit["failures"] = 0
                logger.info(f"Circuit breaker closed for {provider.value}:{request_type}")
            else:
                # Reduce failure count on success
                circuit["failures"] = max(0, circuit["failures"] - 1)
    
    def _assess_response_quality(self, content: str, request: LLMRequest) -> float:
        """Assess response quality score (0.0 to 1.0)"""
        score = 0.5  # Base score
        
        # Length check
        if len(content) > 50:
            score += 0.1
        if len(content) > 200:
            score += 0.1
        
        # Content quality indicators
        if any(indicator in content.lower() for indicator in ['analysis', 'because', 'therefore', 'however', 'additionally']):
            score += 0.1
        
        # Structure indicators
        if '\n' in content:
            score += 0.05
        if any(marker in content for marker in ['1.', '2.', '•', '-']):
            score += 0.1
        
        # Completeness check
        if not content.endswith('...') and len(content) > 20:
            score += 0.05
        
        return min(1.0, score)
    
    def _generate_fallback_response(self, messages: List[BaseMessage], request: LLMRequest) -> str:
        """Generate simple fallback response when all providers fail"""
        
        last_message = messages[-1] if messages else None
        if not last_message or not hasattr(last_message, 'content'):
            return "I apologize, but I'm currently experiencing technical difficulties and cannot process your request."
        
        user_input = last_message.content.lower()
        
        # Simple pattern matching for fallback responses
        if any(word in user_input for word in ['analyze', 'analysis']):
            return f"I understand you're asking for an analysis. While I'm experiencing technical difficulties with my advanced analysis capabilities, I recommend reviewing the key factors related to your query and consulting additional resources for detailed analysis."
        
        elif any(word in user_input for word in ['buy', 'sell', 'invest', 'trade']):
            return "I'm currently unable to provide specific trading advice due to technical issues. Please consult with a qualified financial advisor and consider your risk tolerance before making any investment decisions."
        
        elif any(word in user_input for word in ['what', 'how', 'why', 'explain']):
            return "I'm experiencing technical difficulties with my knowledge systems. For reliable information on this topic, I recommend consulting authoritative sources or educational materials."
        
        else:
            return f"I apologize, but I'm currently experiencing technical difficulties and cannot provide my usual detailed response to your question: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'. Please try again later or rephrase your question."
    
    def _update_success_metrics(self, cost: float, response_time: float):
        """Update service success metrics"""
        self.total_cost += cost
        
        # Update average response time with exponential moving average
        if self.request_count == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = self.average_response_time * 0.9 + response_time * 0.1
        
        # Update success rate
        successful_requests = self.request_count - self._get_failure_count()
        self.success_rate = (successful_requests / self.request_count) * 100 if self.request_count > 0 else 100
    
    def _update_failure_metrics(self, response_time: float):
        """Update service failure metrics"""
        # Update average response time (including failures)
        if self.request_count == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = self.average_response_time * 0.9 + response_time * 0.1
        
        # Success rate will be updated by _get_failure_count()
        successful_requests = self.request_count - self._get_failure_count()
        self.success_rate = (successful_requests / self.request_count) * 100 if self.request_count > 0 else 100
    
    def _get_failure_count(self) -> int:
        """Get total failure count from circuit breakers"""
        total_failures = 0
        for circuit in self.circuit_breakers.values():
            total_failures += circuit.get("failures", 0)
        return min(total_failures, self.request_count)
    
    async def generate_chat_response(self, 
                                   user_message: str,
                                   conversation_history: Optional[List[BaseMessage]] = None,
                                   user_id: Optional[int] = None,
                                   session_id: Optional[str] = None,
                                   streaming: bool = False) -> LLMResponse:
        """Convenient method for chat responses"""
        
        messages = conversation_history or []
        messages.append(HumanMessage(content=user_message))
        
        request = LLMRequest(
            messages=messages,
            task_type="chat",
            priority=RequestPriority.MEDIUM,
            quality=ResponseQuality.STANDARD,
            complexity=TaskComplexity.MODERATE,
            user_id=user_id,
            session_id=session_id,
            streaming=streaming
        )
        
        return await self.generate_response(request)
    
    async def generate_analysis(self,
                               content: str,
                               analysis_type: str = "general",
                               quality: ResponseQuality = ResponseQuality.PREMIUM,
                               user_id: Optional[int] = None) -> LLMResponse:
        """Generate detailed analysis"""
        
        system_message = SystemMessage(content=f"You are an expert analyst. Provide a comprehensive {analysis_type} analysis of the following content.")
        user_message = HumanMessage(content=content)
        
        request = LLMRequest(
            messages=[system_message, user_message],
            task_type="analysis",
            priority=RequestPriority.HIGH,
            quality=quality,
            complexity=TaskComplexity.COMPLEX,
            user_id=user_id,
            max_tokens=2000
        )
        
        return await self.generate_response(request)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        
        provider_stats = self.provider_manager.get_provider_stats()
        cost_summary = self.cost_tracker.get_current_usage_summary()
        
        return {
            "service_metrics": {
                "total_requests": self.request_count,
                "success_rate": self.success_rate,
                "average_response_time": self.average_response_time,
                "total_cost": self.total_cost,
                "average_cost_per_request": self.total_cost / max(1, self.request_count)
            },
            "provider_stats": provider_stats,
            "cost_summary": cost_summary,
            "circuit_breakers": {
                key: {
                    "state": circuit["state"],
                    "failures": circuit["failures"],
                    "last_failure": circuit["last_failure"].isoformat() if circuit["last_failure"] else None
                }
                for key, circuit in self.circuit_breakers.items()
            },
            "fallback_stats": {
                "failure_patterns": len(self.fallback_system.failure_patterns),
                "success_patterns": len(self.fallback_system.success_patterns),
                "adaptive_timeouts": {
                    provider.value: timeout 
                    for provider, timeout in self.fallback_system.adaptive_timeouts.items()
                }
            }
        }

# Global LLM service instance
llm_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    """Get global LLM service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service

# Convenient wrapper functions
async def chat(message: str,
               conversation_history: Optional[List[BaseMessage]] = None,
               user_id: Optional[int] = None,
               streaming: bool = False) -> LLMResponse:
    """Simple chat function"""
    service = get_llm_service()
    return await service.generate_chat_response(
        user_message=message,
        conversation_history=conversation_history,
        user_id=user_id,
        streaming=streaming
    )

async def analyze(content: str,
                  analysis_type: str = "financial",
                  user_id: Optional[int] = None) -> LLMResponse:
    """Simple analysis function"""
    service = get_llm_service()
    return await service.generate_analysis(
        content=content,
        analysis_type=analysis_type,
        user_id=user_id
    )
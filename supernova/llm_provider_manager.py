"""
Enhanced LLM Provider Manager with Failover and Health Monitoring

This module provides:
- Multi-provider LLM management with intelligent routing
- Provider health monitoring and automatic failover
- Load balancing and performance optimization
- Cost-aware provider selection
- Streaming response support for all providers
- Provider-specific configuration and optimization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import logging
import time
import statistics
from pathlib import Path

# Core imports
from .config import settings
from .llm_key_manager import get_key_manager, ProviderType

# LangChain imports with error handling
try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.outputs import LLMResult
    
    # Provider-specific imports
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_ollama import OllamaLLM
    from langchain_huggingface import HuggingFacePipeline
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ProviderStatus(str, Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_BASED = "performance_based"

class RequestPriority(str, Enum):
    """Request priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ProviderMetrics:
    """Provider performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    tokens_processed: int = 0
    cost_incurred: float = 0.0
    last_request_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def failure_rate(self) -> float:
        return 100.0 - self.success_rate
    
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.response_times.append(response_time)
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        self.average_response_time = statistics.mean(self.response_times)

@dataclass
class ProviderConfig:
    """Provider configuration"""
    provider_type: ProviderType
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    weight: float = 1.0  # For weighted load balancing
    max_concurrent_requests: int = 10
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    models: Dict[str, str] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Enhanced streaming callback handler"""
    
    def __init__(self, callback_fn: Callable[[str], None]):
        self.callback_fn = callback_fn
        self.tokens = []
        self.start_time = time.time()
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts processing"""
        self.start_time = time.time()
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle new token from streaming"""
        self.tokens.append(token)
        if self.callback_fn:
            self.callback_fn(token)
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle end of LLM generation"""
        total_time = time.time() - self.start_time
        if self.callback_fn:
            self.callback_fn(f"\n[STREAM_COMPLETE: {len(self.tokens)} tokens, {total_time:.2f}s]")

class LLMProviderManager:
    """
    Enhanced LLM provider manager with health monitoring and intelligent routing
    """
    
    def __init__(self):
        """Initialize provider manager"""
        self.providers: Dict[ProviderType, BaseLanguageModel] = {}
        self.provider_configs: Dict[ProviderType, ProviderConfig] = {}
        self.provider_metrics: Dict[ProviderType, ProviderMetrics] = {}
        self.provider_status: Dict[ProviderType, ProviderStatus] = {}
        
        self.key_manager = get_key_manager()
        self.load_balancer_index = 0
        self.active_requests: Dict[ProviderType, int] = {}
        
        # Health monitoring
        self.health_check_interval = getattr(settings, 'LLM_HEALTH_CHECK_INTERVAL', 300)
        self.health_check_timeout = getattr(settings, 'LLM_HEALTH_CHECK_TIMEOUT', 10)
        self.failover_threshold = getattr(settings, 'LLM_FAILOVER_THRESHOLD', 3)
        self.failover_cooldown = getattr(settings, 'LLM_FAILOVER_COOLDOWN', 900)
        
        # Load balancing
        self.load_balancing_strategy = LoadBalancingStrategy(
            getattr(settings, 'LLM_LOAD_BALANCING', 'round_robin')
        )
        
        self._initialize_providers()
        self._start_health_monitoring()
        
        logger.info("LLM Provider Manager initialized")
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        # OpenAI
        if getattr(settings, 'LLM_PROVIDER', None) == 'openai' or \
           getattr(settings, 'OPENAI_API_KEY', None):
            self._initialize_openai()
        
        # Anthropic
        if getattr(settings, 'LLM_PROVIDER', None) == 'anthropic' or \
           getattr(settings, 'ANTHROPIC_API_KEY', None):
            self._initialize_anthropic()
        
        # Ollama
        if getattr(settings, 'LLM_PROVIDER', None) == 'ollama' or \
           getattr(settings, 'OLLAMA_BASE_URL', None):
            self._initialize_ollama()
        
        # HuggingFace
        if getattr(settings, 'LLM_PROVIDER', None) == 'huggingface' or \
           getattr(settings, 'HUGGINGFACE_API_TOKEN', None):
            self._initialize_huggingface()
        
        logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    def _initialize_openai(self):
        """Initialize OpenAI provider"""
        try:
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                priority=1,
                weight=1.0,
                max_concurrent_requests=getattr(settings, 'OPENAI_MAX_REQUESTS_PER_MINUTE', 3000) // 60,
                timeout=getattr(settings, 'LLM_TIMEOUT', 60),
                cost_per_1k_input_tokens=getattr(settings, 'OPENAI_GPT4_INPUT_COST', 0.03),
                cost_per_1k_output_tokens=getattr(settings, 'OPENAI_GPT4_OUTPUT_COST', 0.06),
                models={
                    'chat': getattr(settings, 'OPENAI_CHAT_MODEL', 'gpt-4o'),
                    'analysis': getattr(settings, 'OPENAI_ANALYSIS_MODEL', 'gpt-4o'),
                    'summary': getattr(settings, 'OPENAI_SUMMARY_MODEL', 'gpt-3.5-turbo')
                }
            )
            
            self.provider_configs[ProviderType.OPENAI] = config
            self.provider_metrics[ProviderType.OPENAI] = ProviderMetrics()
            self.provider_status[ProviderType.OPENAI] = ProviderStatus.HEALTHY
            self.active_requests[ProviderType.OPENAI] = 0
            
            # Initialize provider will be done lazily when first used
            logger.info("OpenAI provider configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
    
    def _initialize_anthropic(self):
        """Initialize Anthropic provider"""
        try:
            config = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                priority=2,
                weight=0.8,
                max_concurrent_requests=getattr(settings, 'ANTHROPIC_MAX_REQUESTS_PER_MINUTE', 1000) // 60,
                timeout=getattr(settings, 'LLM_TIMEOUT', 60),
                cost_per_1k_input_tokens=getattr(settings, 'ANTHROPIC_SONNET_INPUT_COST', 0.003),
                cost_per_1k_output_tokens=getattr(settings, 'ANTHROPIC_SONNET_OUTPUT_COST', 0.015),
                models={
                    'chat': getattr(settings, 'ANTHROPIC_CHAT_MODEL', 'claude-3-sonnet-20240229'),
                    'analysis': getattr(settings, 'ANTHROPIC_ANALYSIS_MODEL', 'claude-3-opus-20240229'),
                    'summary': getattr(settings, 'ANTHROPIC_SUMMARY_MODEL', 'claude-3-haiku-20240307')
                }
            )
            
            self.provider_configs[ProviderType.ANTHROPIC] = config
            self.provider_metrics[ProviderType.ANTHROPIC] = ProviderMetrics()
            self.provider_status[ProviderType.ANTHROPIC] = ProviderStatus.HEALTHY
            self.active_requests[ProviderType.ANTHROPIC] = 0
            
            logger.info("Anthropic provider configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
    
    def _initialize_ollama(self):
        """Initialize Ollama provider"""
        try:
            config = ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                priority=3,
                weight=0.6,
                max_concurrent_requests=getattr(settings, 'OLLAMA_MAX_CONCURRENT', 2),
                timeout=getattr(settings, 'OLLAMA_TIMEOUT', 120),
                cost_per_1k_input_tokens=0.0,  # Local models are free
                cost_per_1k_output_tokens=0.0,
                models={
                    'chat': getattr(settings, 'OLLAMA_CHAT_MODEL', 'llama2:13b'),
                    'analysis': getattr(settings, 'OLLAMA_ANALYSIS_MODEL', 'codellama:13b'),
                    'summary': getattr(settings, 'OLLAMA_SUMMARY_MODEL', 'llama2:7b')
                },
                custom_settings={
                    'base_url': getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434'),
                    'num_gpu': getattr(settings, 'OLLAMA_NUM_GPU', 1),
                    'num_thread': getattr(settings, 'OLLAMA_NUM_THREAD', 8)
                }
            )
            
            self.provider_configs[ProviderType.OLLAMA] = config
            self.provider_metrics[ProviderType.OLLAMA] = ProviderMetrics()
            self.provider_status[ProviderType.OLLAMA] = ProviderStatus.HEALTHY
            self.active_requests[ProviderType.OLLAMA] = 0
            
            logger.info("Ollama provider configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace provider"""
        try:
            config = ProviderConfig(
                provider_type=ProviderType.HUGGINGFACE,
                priority=4,
                weight=0.4,
                max_concurrent_requests=getattr(settings, 'HUGGINGFACE_DAILY_REQUEST_LIMIT', 5000) // 24,
                timeout=getattr(settings, 'HUGGINGFACE_INFERENCE_TIMEOUT', 30),
                cost_per_1k_input_tokens=0.0,  # Free tier
                cost_per_1k_output_tokens=0.0,
                models={
                    'chat': getattr(settings, 'HUGGINGFACE_CHAT_MODEL', 'microsoft/DialoGPT-large'),
                    'analysis': getattr(settings, 'HUGGINGFACE_ANALYSIS_MODEL', 'facebook/blenderbot-400M-distill'),
                    'summary': getattr(settings, 'HUGGINGFACE_SUMMARY_MODEL', 'facebook/bart-large-cnn')
                }
            )
            
            self.provider_configs[ProviderType.HUGGINGFACE] = config
            self.provider_metrics[ProviderType.HUGGINGFACE] = ProviderMetrics()
            self.provider_status[ProviderType.HUGGINGFACE] = ProviderStatus.HEALTHY
            self.active_requests[ProviderType.HUGGINGFACE] = 0
            
            logger.info("HuggingFace provider configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace provider: {e}")
    
    async def _get_or_create_provider(self, provider_type: ProviderType, model_type: str = 'chat') -> Optional[BaseLanguageModel]:
        """Get or create provider instance"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available")
            return None
        
        # Check if provider already exists
        provider_key = f"{provider_type.value}:{model_type}"
        if provider_key in self.providers:
            return self.providers[provider_key]
        
        try:
            config = self.provider_configs.get(provider_type)
            if not config or not config.enabled:
                return None
            
            # Get API key
            api_key = await self.key_manager.get_api_key(provider_type)
            
            # Create provider based on type
            if provider_type == ProviderType.OPENAI:
                if not api_key:
                    logger.error("OpenAI API key not found")
                    return None
                
                model_name = config.models.get(model_type, config.models.get('chat', 'gpt-4o'))
                provider = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    organization=getattr(settings, 'OPENAI_ORG_ID', None),
                    base_url=getattr(settings, 'OPENAI_BASE_URL', None),
                    temperature=getattr(settings, 'LLM_TEMPERATURE', 0.2),
                    max_tokens=getattr(settings, 'LLM_MAX_TOKENS', 4000),
                    timeout=config.timeout,
                    max_retries=config.max_retries,
                    streaming=True
                )
            
            elif provider_type == ProviderType.ANTHROPIC:
                if not api_key:
                    logger.error("Anthropic API key not found")
                    return None
                
                model_name = config.models.get(model_type, config.models.get('chat', 'claude-3-sonnet-20240229'))
                provider = ChatAnthropic(
                    model=model_name,
                    api_key=api_key,
                    temperature=getattr(settings, 'LLM_TEMPERATURE', 0.2),
                    max_tokens=getattr(settings, 'LLM_MAX_TOKENS', 4000),
                    timeout=config.timeout,
                    max_retries=config.max_retries
                )
            
            elif provider_type == ProviderType.OLLAMA:
                model_name = config.models.get(model_type, config.models.get('chat', 'llama2'))
                provider = OllamaLLM(
                    model=model_name,
                    base_url=config.custom_settings.get('base_url', 'http://localhost:11434'),
                    temperature=getattr(settings, 'LLM_TEMPERATURE', 0.2),
                    timeout=config.timeout
                )
            
            elif provider_type == ProviderType.HUGGINGFACE:
                if not api_key:
                    logger.error("HuggingFace API token not found")
                    return None
                
                model_name = config.models.get(model_type, config.models.get('chat', 'microsoft/DialoGPT-large'))
                provider = HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    huggingfacehub_api_token=api_key,
                    model_kwargs={
                        "temperature": getattr(settings, 'LLM_TEMPERATURE', 0.2),
                        "max_length": getattr(settings, 'HUGGINGFACE_MAX_LENGTH', 512)
                    }
                )
            else:
                return None
            
            # Cache the provider
            self.providers[provider_key] = provider
            
            logger.info(f"Created {provider_type.value} provider for {model_type}")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create {provider_type.value} provider: {e}")
            self.provider_status[provider_type] = ProviderStatus.UNHEALTHY
            return None
    
    def _select_provider(self, 
                        task_type: str = 'chat',
                        priority: RequestPriority = RequestPriority.MEDIUM,
                        exclude: List[ProviderType] = None) -> Optional[ProviderType]:
        """Select best provider based on strategy and health"""
        exclude = exclude or []
        
        # Filter healthy providers
        available_providers = [
            provider_type for provider_type, status in self.provider_status.items()
            if status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED] and 
            provider_type not in exclude and
            self.provider_configs.get(provider_type, ProviderConfig(provider_type)).enabled
        ]
        
        if not available_providers:
            logger.error("No healthy providers available")
            return None
        
        # Apply selection strategy
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED:
            return self._select_weighted(available_providers)
        elif self.load_balancing_strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            return self._select_cost_optimized(available_providers, task_type)
        elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._select_performance_based(available_providers)
        else:
            # Default to priority-based selection
            return min(available_providers, key=lambda p: self.provider_configs[p].priority)
    
    def _select_round_robin(self, providers: List[ProviderType]) -> ProviderType:
        """Round-robin provider selection"""
        if not providers:
            return None
        
        selected = providers[self.load_balancer_index % len(providers)]
        self.load_balancer_index += 1
        return selected
    
    def _select_least_loaded(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider with least active requests"""
        return min(providers, key=lambda p: self.active_requests.get(p, 0))
    
    def _select_weighted(self, providers: List[ProviderType]) -> ProviderType:
        """Weighted random selection"""
        import random
        
        weights = [self.provider_configs[p].weight for p in providers]
        return random.choices(providers, weights=weights)[0]
    
    def _select_cost_optimized(self, providers: List[ProviderType], task_type: str) -> ProviderType:
        """Select lowest cost provider for task type"""
        def get_cost(provider_type: ProviderType) -> float:
            config = self.provider_configs[provider_type]
            # Estimate cost based on typical token usage for task type
            if task_type == 'summary':
                estimated_input_tokens = 1000
                estimated_output_tokens = 200
            elif task_type == 'analysis':
                estimated_input_tokens = 2000
                estimated_output_tokens = 1000
            else:  # chat
                estimated_input_tokens = 500
                estimated_output_tokens = 500
            
            cost = (estimated_input_tokens / 1000 * config.cost_per_1k_input_tokens +
                    estimated_output_tokens / 1000 * config.cost_per_1k_output_tokens)
            
            return cost
        
        return min(providers, key=get_cost)
    
    def _select_performance_based(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider based on performance metrics"""
        def get_performance_score(provider_type: ProviderType) -> float:
            metrics = self.provider_metrics[provider_type]
            # Higher score is better
            score = metrics.success_rate * 0.4 + (100 - min(metrics.average_response_time, 10) * 10) * 0.6
            return score
        
        return max(providers, key=get_performance_score)
    
    async def generate_response(self,
                              messages: List[BaseMessage],
                              task_type: str = 'chat',
                              priority: RequestPriority = RequestPriority.MEDIUM,
                              streaming_callback: Optional[Callable[[str], None]] = None,
                              preferred_provider: Optional[ProviderType] = None,
                              **kwargs) -> Optional[str]:
        """
        Generate response using best available provider with automatic failover
        """
        excluded_providers = []
        max_attempts = len(self.provider_configs)
        
        for attempt in range(max_attempts):
            # Select provider
            if preferred_provider and attempt == 0:
                selected_provider = preferred_provider
            else:
                selected_provider = self._select_provider(task_type, priority, excluded_providers)
            
            if not selected_provider:
                logger.error("No providers available for request")
                return None
            
            # Track request start
            start_time = time.time()
            self.active_requests[selected_provider] = self.active_requests.get(selected_provider, 0) + 1
            
            try:
                # Get provider instance
                provider = await self._get_or_create_provider(selected_provider, task_type)
                if not provider:
                    excluded_providers.append(selected_provider)
                    continue
                
                # Set up streaming callback if needed
                callbacks = []
                if streaming_callback and hasattr(provider, 'streaming') and provider.streaming:
                    callbacks.append(StreamingCallbackHandler(streaming_callback))
                
                # Generate response
                if len(messages) == 1 and isinstance(messages[0], HumanMessage):
                    # Simple text generation
                    response = await provider.ainvoke(
                        messages[0].content,
                        config={"callbacks": callbacks} if callbacks else None,
                        **kwargs
                    )
                else:
                    # Chat-style conversation
                    if hasattr(provider, 'ainvoke'):
                        response = await provider.ainvoke(
                            messages,
                            config={"callbacks": callbacks} if callbacks else None,
                            **kwargs
                        )
                    else:
                        # Fallback for providers that don't support chat format
                        text_input = "\n".join([msg.content for msg in messages if hasattr(msg, 'content')])
                        response = await provider.ainvoke(text_input, **kwargs)
                
                # Track successful request
                response_time = time.time() - start_time
                await self._record_success(selected_provider, response_time, len(str(response)))
                
                # Extract response text
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
                    
            except Exception as e:
                logger.error(f"Request failed with {selected_provider.value}: {e}")
                
                # Track failed request
                response_time = time.time() - start_time
                await self._record_failure(selected_provider, response_time, str(e))
                
                # Add to excluded providers for next attempt
                excluded_providers.append(selected_provider)
                
                # If this was the last provider, return error
                if attempt == max_attempts - 1:
                    logger.error("All providers failed for request")
                    return None
                
            finally:
                # Decrease active request count
                self.active_requests[selected_provider] = max(0, self.active_requests[selected_provider] - 1)
        
        return None
    
    async def _record_success(self, provider_type: ProviderType, response_time: float, tokens: int):
        """Record successful request metrics"""
        metrics = self.provider_metrics[provider_type]
        metrics.total_requests += 1
        metrics.successful_requests += 1
        metrics.tokens_processed += tokens
        metrics.last_request_time = datetime.utcnow()
        metrics.consecutive_failures = 0
        metrics.update_response_time(response_time)
        
        # Calculate cost
        config = self.provider_configs[provider_type]
        estimated_cost = tokens / 1000 * config.cost_per_1k_output_tokens
        metrics.cost_incurred += estimated_cost
        
        # Update provider status if it was degraded
        if self.provider_status[provider_type] == ProviderStatus.DEGRADED:
            self.provider_status[provider_type] = ProviderStatus.HEALTHY
    
    async def _record_failure(self, provider_type: ProviderType, response_time: float, error: str):
        """Record failed request metrics"""
        metrics = self.provider_metrics[provider_type]
        metrics.total_requests += 1
        metrics.failed_requests += 1
        metrics.last_failure_time = datetime.utcnow()
        metrics.consecutive_failures += 1
        
        # Update provider status based on consecutive failures
        if metrics.consecutive_failures >= self.failover_threshold:
            self.provider_status[provider_type] = ProviderStatus.UNHEALTHY
            logger.warning(f"Provider {provider_type.value} marked as unhealthy after {metrics.consecutive_failures} consecutive failures")
        elif metrics.consecutive_failures >= 2:
            self.provider_status[provider_type] = ProviderStatus.DEGRADED
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if not getattr(settings, 'LLM_HEALTH_CHECK_ENABLED', True):
            return
        
        async def health_check_loop():
            while True:
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    await asyncio.sleep(60)  # Shorter retry on error
        
        # Start health check task
        asyncio.create_task(health_check_loop())
        logger.info("Health monitoring started")
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        for provider_type in self.provider_configs.keys():
            try:
                is_healthy = await self._health_check_provider(provider_type)
                
                if is_healthy:
                    if self.provider_status[provider_type] in [ProviderStatus.UNHEALTHY, ProviderStatus.OFFLINE]:
                        self.provider_status[provider_type] = ProviderStatus.HEALTHY
                        logger.info(f"Provider {provider_type.value} recovered")
                else:
                    if self.provider_status[provider_type] == ProviderStatus.HEALTHY:
                        self.provider_status[provider_type] = ProviderStatus.DEGRADED
                        logger.warning(f"Provider {provider_type.value} degraded")
                
            except Exception as e:
                logger.error(f"Health check failed for {provider_type.value}: {e}")
                self.provider_status[provider_type] = ProviderStatus.OFFLINE
    
    async def _health_check_provider(self, provider_type: ProviderType) -> bool:
        """Perform health check on specific provider"""
        try:
            # Use key manager to validate API key and connection
            return await self.key_manager.validate_api_key(provider_type)
        except Exception as e:
            logger.debug(f"Health check failed for {provider_type.value}: {e}")
            return False
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get comprehensive provider statistics"""
        stats = {
            "total_providers": len(self.provider_configs),
            "healthy_providers": len([s for s in self.provider_status.values() if s == ProviderStatus.HEALTHY]),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "providers": {}
        }
        
        for provider_type, metrics in self.provider_metrics.items():
            status = self.provider_status[provider_type]
            config = self.provider_configs[provider_type]
            
            stats["providers"][provider_type.value] = {
                "status": status.value,
                "enabled": config.enabled,
                "priority": config.priority,
                "total_requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "tokens_processed": metrics.tokens_processed,
                "cost_incurred": metrics.cost_incurred,
                "active_requests": self.active_requests.get(provider_type, 0),
                "last_request": metrics.last_request_time.isoformat() if metrics.last_request_time else None
            }
        
        return stats
    
    async def enable_provider(self, provider_type: ProviderType, enabled: bool = True):
        """Enable or disable a provider"""
        if provider_type in self.provider_configs:
            self.provider_configs[provider_type].enabled = enabled
            if enabled:
                self.provider_status[provider_type] = ProviderStatus.HEALTHY
            else:
                self.provider_status[provider_type] = ProviderStatus.OFFLINE
            
            logger.info(f"Provider {provider_type.value} {'enabled' if enabled else 'disabled'}")
    
    async def update_provider_config(self, provider_type: ProviderType, config_updates: Dict[str, Any]):
        """Update provider configuration"""
        if provider_type in self.provider_configs:
            config = self.provider_configs[provider_type]
            
            for key, value in config_updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Clear cached provider to force recreation with new config
            for key in list(self.providers.keys()):
                if key.startswith(f"{provider_type.value}:"):
                    del self.providers[key]
            
            logger.info(f"Updated configuration for {provider_type.value}")

# Global provider manager instance
provider_manager: Optional[LLMProviderManager] = None

def get_provider_manager() -> LLMProviderManager:
    """Get global provider manager instance"""
    global provider_manager
    if provider_manager is None:
        provider_manager = LLMProviderManager()
    return provider_manager

async def generate_llm_response(messages: Union[str, List[BaseMessage]],
                               task_type: str = 'chat',
                               streaming_callback: Optional[Callable[[str], None]] = None,
                               **kwargs) -> Optional[str]:
    """
    Convenient function to generate LLM response with automatic provider selection
    """
    manager = get_provider_manager()
    
    # Convert string to message format
    if isinstance(messages, str):
        messages = [HumanMessage(content=messages)]
    
    return await manager.generate_response(
        messages=messages,
        task_type=task_type,
        streaming_callback=streaming_callback,
        **kwargs
    )
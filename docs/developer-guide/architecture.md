# SuperNova AI Architecture Overview

This document provides a comprehensive overview of SuperNova AI's system architecture, including component relationships, data flow, technology stack, and design patterns.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Component Architecture](#component-architecture)
4. [Data Architecture](#data-architecture)
5. [API Architecture](#api-architecture)
6. [Security Architecture](#security-architecture)
7. [Scalability and Performance](#scalability-and-performance)
8. [Integration Architecture](#integration-architecture)
9. [Development Patterns](#development-patterns)

## System Architecture

### High-Level Architecture

SuperNova AI follows a modern, microservices-inspired architecture with clear separation of concerns:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Frontend Layer    │    │   API Gateway       │    │   External APIs     │
│                     │    │                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ React/TypeScript│ │◄──►│ │ FastAPI         │ │◄──►│ │ OpenAI/Anthropic│ │
│ │ Material-UI     │ │    │ │ Authentication  │ │    │ │ NovaSignal      │ │
│ │ WebSocket Client│ │    │ │ Rate Limiting   │ │    │ │ Market Data     │ │
│ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                         │                         │
           │                         │                         │
           ▼                         ▼                         ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Business Logic    │    │   Data Layer        │    │   Infrastructure    │
│                     │    │                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ AI Advisors     │ │    │ │ SQLAlchemy ORM  │ │    │ │ Docker          │ │
│ │ Backtester      │ │    │ │ PostgreSQL/     │ │    │ │ Redis Cache     │ │
│ │ Risk Engine     │ │    │ │ TimescaleDB     │ │    │ │ WebSocket       │ │
│ │ Strategy Engine │ │    │ │ SQLite (dev)    │ │    │ │ Monitoring      │ │
│ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Core Design Principles

1. **Separation of Concerns**: Clear boundaries between presentation, business logic, and data layers
2. **Modularity**: Loosely coupled components with well-defined interfaces
3. **Scalability**: Horizontal and vertical scaling capabilities
4. **Maintainability**: Clean code, comprehensive testing, and documentation
5. **Security**: Defense in depth with multiple security layers
6. **Performance**: Optimized for low-latency financial data processing

## Technology Stack

### Backend Technologies

#### Core Framework
- **FastAPI**: High-performance, modern Python web framework
- **Python 3.9+**: Primary programming language
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and settings management

#### Database Stack
- **SQLAlchemy**: ORM for database operations
- **PostgreSQL**: Primary production database
- **TimescaleDB**: Time-series data for sentiment and market data
- **SQLite**: Development and testing database
- **Redis**: Caching and session storage

#### AI/ML Stack
- **LangChain**: LLM orchestration and integration
- **OpenAI GPT-4**: Primary language model
- **Anthropic Claude**: Secondary language model
- **VectorBT**: High-performance backtesting library
- **scikit-learn**: Machine learning utilities
- **NumPy/Pandas**: Data processing and analysis

#### Infrastructure
- **Docker**: Containerization and deployment
- **Prefect**: Workflow orchestration
- **WebSockets**: Real-time communication
- **Celery**: Background task processing (optional)

### Frontend Technologies

#### Core Framework
- **React 18**: User interface library
- **TypeScript**: Type-safe JavaScript development
- **Material-UI (MUI) 5**: Component library and design system
- **React Router**: Client-side routing

#### State Management
- **React Query**: Server state management
- **React Context**: Local state management
- **React Hook Form**: Form state management

#### Visualization
- **Recharts**: Chart and graph library
- **Lightweight Charts**: Professional trading charts
- **Plotly.js**: Advanced data visualization
- **D3.js**: Custom visualizations

#### Build Tools
- **Webpack**: Module bundler
- **Babel**: JavaScript compiler
- **ESLint**: Code linting
- **Prettier**: Code formatting

## Component Architecture

### Backend Components

#### API Layer (`supernova/api.py`)
```python
# Primary responsibilities:
# - HTTP request handling
# - Authentication and authorization
# - Request validation and response formatting
# - Error handling and logging

FastAPI Application
├── Middleware Stack
│   ├── CORS Middleware
│   ├── Authentication Middleware
│   └── Rate Limiting Middleware
├── Route Handlers
│   ├── User Management (/intake, /profile)
│   ├── Advisory Services (/advice, /chat)
│   ├── Portfolio Management (/portfolio)
│   ├── Backtesting (/backtest)
│   └── Market Data (/sentiment, /novasignal)
└── Exception Handlers
```

#### Business Logic Layer

**Advisory Engine (`supernova/advisor.py`)**
```python
class AdvisoryEngine:
    """Core investment advisory logic"""
    
    def score_risk(self, risk_questions: List[int]) -> int:
        """Calculate risk score from questionnaire"""
        
    def advise(self, bars: List[dict], risk_score: int, 
               sentiment: float, **kwargs) -> AdviceOut:
        """Generate investment advice based on multiple factors"""
        
    def explain_advice(self, advice: AdviceOut) -> str:
        """Provide human-readable explanation of advice"""
```

**Strategy Engine (`supernova/strategy_engine.py`)**
```python
class StrategyEngine:
    """Trading strategy implementation and evaluation"""
    
    def evaluate_strategy(self, template: str, params: dict, 
                         bars: List[dict]) -> dict:
        """Evaluate strategy performance on historical data"""
        
    def generate_signals(self, strategy: Strategy, 
                        bars: List[dict]) -> List[Signal]:
        """Generate buy/sell signals from strategy"""
```

**Backtesting Engine (`supernova/backtester.py`)**
```python
class BacktestEngine:
    """High-performance backtesting with VectorBT"""
    
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame,
                     **config) -> BacktestResults:
        """Execute backtest with specified parameters"""
        
    def calculate_metrics(self, results: BacktestResults) -> dict:
        """Calculate performance and risk metrics"""
```

#### Data Layer

**Database Models (`supernova/db.py`)**
```python
# Core data models using SQLAlchemy

class User(Base):
    """User account information"""
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class Profile(Base):
    """User investment profile and risk assessment"""
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    risk_score = Column(Integer, nullable=False)
    time_horizon_yrs = Column(Integer)
    objectives = Column(Text)
    constraints = Column(Text)
    
class Asset(Base):
    """Financial instruments and securities"""
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    asset_class = Column(String(50))
    exchange = Column(String(20))
    metadata = Column(JSON)
```

**Sentiment Models (`supernova/sentiment_models.py`)**
```python
# TimescaleDB models for time-series sentiment data

class SentimentData(Base):
    """Historical sentiment data points"""
    __tablename__ = 'sentiment_data'
    
    timestamp = Column(DateTime, primary_key=True)
    symbol = Column(String(20), primary_key=True)
    overall_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    social_momentum = Column(Float)
    news_sentiment = Column(Float)
    source_counts = Column(JSON)
```

### Frontend Components

#### Application Structure
```
src/
├── components/           # Reusable UI components
│   ├── charts/          # Chart components
│   ├── common/          # Shared components
│   ├── dashboard/       # Dashboard-specific components
│   └── layout/          # Layout components
├── hooks/               # Custom React hooks
├── pages/               # Page-level components
├── services/            # API integration services
├── theme/               # Material-UI theme configuration
└── utils/               # Utility functions
```

#### Key Components

**Dashboard Components**
```typescript
// Portfolio overview with real-time updates
export const PortfolioOverviewCard: React.FC = () => {
  const { data: portfolio } = usePortfolio();
  const { isConnected } = useWebSocket();
  
  return (
    <Card>
      <CardContent>
        <PortfolioSummary data={portfolio} />
        <AllocationChart data={portfolio.allocation} />
        <PerformanceMetrics data={portfolio.metrics} />
      </CardContent>
    </Card>
  );
};

// Real-time financial charts
export const FinancialChart: React.FC<ChartProps> = ({ 
  symbol, timeframe, height 
}) => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const ws = useWebSocket();
  
  useEffect(() => {
    ws.subscribe(`market_data.${symbol}`, (data) => {
      setChartData(prev => [...prev, data]);
    });
  }, [symbol, ws]);
  
  return (
    <LightweightChart 
      data={chartData}
      options={{ height, width: '100%' }}
    />
  );
};
```

**Chat Interface**
```typescript
// AI chat interface with conversation management
export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentAdvisor, setCurrentAdvisor] = useState<AdvisorType>('balanced');
  const { sendMessage, isTyping } = useChatAPI();
  
  const handleSendMessage = async (message: string) => {
    const response = await sendMessage({
      message,
      advisor_type: currentAdvisor,
      profile_id: user.profile_id
    });
    
    setMessages(prev => [...prev, response]);
  };
  
  return (
    <ChatContainer>
      <MessageList messages={messages} />
      <AdvisorSelector 
        current={currentAdvisor}
        onChange={setCurrentAdvisor}
      />
      <MessageInput onSend={handleSendMessage} />
    </ChatContainer>
  );
};
```

## Data Architecture

### Database Schema

#### Core Tables Structure
```sql
-- Users and Profiles
users (id, name, email, created_at, updated_at)
profiles (id, user_id, risk_score, time_horizon_yrs, objectives, constraints)

-- Assets and Markets
assets (id, symbol, asset_class, exchange, metadata)
market_data (timestamp, symbol, open, high, low, close, volume)

-- Portfolio Management
portfolios (id, profile_id, name, created_at)
positions (id, portfolio_id, asset_id, shares, cost_basis, created_at)
transactions (id, portfolio_id, asset_id, type, shares, price, timestamp)

-- Watchlists and Alerts
watchlist_items (id, profile_id, asset_id, notes, active, created_at)
alerts (id, profile_id, symbol, alert_type, conditions, active, created_at)

-- AI and Analysis
chat_sessions (id, profile_id, advisor_type, created_at)
chat_messages (id, session_id, user_message, ai_response, timestamp)
backtest_results (id, profile_id, strategy, parameters, metrics, created_at)
```

#### TimescaleDB Schema (Time-Series Data)
```sql
-- Sentiment analysis data (hypertable)
CREATE TABLE sentiment_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    overall_score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    social_momentum FLOAT,
    news_sentiment FLOAT,
    twitter_sentiment FLOAT,
    reddit_sentiment FLOAT,
    source_counts JSONB,
    PRIMARY KEY (timestamp, symbol)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sentiment_data', 'timestamp');

-- Create indexes for efficient querying
CREATE INDEX ON sentiment_data (symbol, timestamp DESC);
CREATE INDEX ON sentiment_data (timestamp DESC, overall_score);
```

### Data Flow Architecture

#### Request Processing Flow
```
Client Request → API Gateway → Authentication → Rate Limiting → 
Business Logic → Data Layer → Database/External APIs → Response
```

#### Real-time Data Flow
```
External Data Source → WebSocket Server → Client Subscriptions → 
UI Updates → Local State Management → User Interface
```

#### AI Advisory Flow
```
User Input → Profile Context → Market Data → Technical Analysis → 
Sentiment Analysis → LLM Processing → Risk Assessment → 
Personalized Advice → Response Formatting → Client Delivery
```

## API Architecture

### RESTful API Design

#### Resource-Based URLs
```
/users                    # User management
/profiles                 # Investment profiles
/portfolios              # Portfolio management
/advice                  # Investment advice
/backtest                # Strategy backtesting
/sentiment              # Sentiment data
/alerts                 # Alert management
/chat                   # AI chat interface
```

#### HTTP Methods and Semantics
- **GET**: Retrieve resources (idempotent)
- **POST**: Create new resources or trigger actions
- **PUT**: Update entire resources (idempotent)
- **PATCH**: Partial resource updates
- **DELETE**: Remove resources (idempotent)

#### Response Format Standards
```json
{
  "success": true,
  "data": {
    // Actual response data
  },
  "metadata": {
    "timestamp": "2025-08-26T16:00:00Z",
    "request_id": "req_abc123",
    "version": "1.0.0"
  },
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 1000,
    "pages": 20
  }
}
```

### WebSocket Architecture

#### Connection Management
```python
class WebSocketManager:
    """Manages WebSocket connections and subscriptions"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connections"""
        await websocket.accept()
        self.connections[client_id] = websocket
        
    async def subscribe(self, client_id: str, channel: str):
        """Subscribe client to data channel"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        self.subscriptions[client_id].add(channel)
        
    async def broadcast(self, channel: str, message: dict):
        """Broadcast message to all channel subscribers"""
        for client_id, channels in self.subscriptions.items():
            if channel in channels:
                websocket = self.connections.get(client_id)
                if websocket:
                    await websocket.send_json(message)
```

#### Channel Types
- **market_data.{symbol}**: Real-time price updates
- **portfolio.{profile_id}**: Portfolio changes
- **alerts.{profile_id}**: Alert notifications
- **chat.{session_id}**: Chat messages
- **system**: System-wide notifications

## Security Architecture

### Authentication and Authorization

#### JWT Token Flow
```
1. User Login → Credentials Validation → JWT Token Generation
2. API Request → Token Validation → Permission Check → Resource Access
3. Token Refresh → New Token Generation → Updated Expiration
```

#### Role-Based Access Control (RBAC)
```python
class Permission:
    READ_PORTFOLIO = "portfolio:read"
    WRITE_PORTFOLIO = "portfolio:write"
    ACCESS_CHAT = "chat:access"
    RUN_BACKTEST = "backtest:run"
    ADMIN_ACCESS = "admin:access"

class UserRole:
    FREE_USER = ["portfolio:read", "chat:access"]
    PREMIUM_USER = ["portfolio:read", "portfolio:write", "chat:access", "backtest:run"]
    ADMIN_USER = ["*"]  # All permissions
```

### Security Layers

#### Input Validation
- **Pydantic Models**: Automatic request validation
- **SQL Injection Prevention**: ORM-based queries
- **XSS Protection**: Input sanitization
- **CSRF Protection**: Token-based validation

#### Rate Limiting
```python
class RateLimiter:
    """Token bucket rate limiting implementation"""
    
    def __init__(self, requests_per_hour: int, burst_size: int):
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.tokens: Dict[str, TokenBucket] = {}
        
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is within rate limits"""
        bucket = self.tokens.get(client_id)
        if not bucket:
            bucket = TokenBucket(self.requests_per_hour, self.burst_size)
            self.tokens[client_id] = bucket
            
        return bucket.consume()
```

## Scalability and Performance

### Performance Optimization

#### Database Optimization
- **Indexing Strategy**: Optimized indexes for common queries
- **Query Optimization**: Efficient SQL query patterns
- **Connection Pooling**: Reuse database connections
- **Read Replicas**: Separate read and write workloads

#### Caching Strategy
```python
# Multi-layer caching implementation
class CacheManager:
    """Hierarchical caching system"""
    
    def __init__(self):
        self.memory_cache = {}  # In-memory cache
        self.redis_cache = redis.Redis()  # Distributed cache
        
    async def get(self, key: str, ttl: int = 3600):
        # Try memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        # Fall back to Redis
        value = await self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = value
            return value
            
        return None
```

#### Application Performance
- **Async/Await**: Non-blocking I/O operations
- **Background Tasks**: Celery for heavy computations
- **Connection Pooling**: Efficient resource utilization
- **Code Profiling**: Performance monitoring and optimization

### Scalability Patterns

#### Horizontal Scaling
- **Load Balancing**: Multiple API server instances
- **Database Sharding**: Partition data across instances
- **Microservices**: Independent service scaling
- **CDN Integration**: Static asset distribution

#### Vertical Scaling
- **Resource Optimization**: CPU and memory efficiency
- **Database Tuning**: Optimized database configuration
- **Caching Layers**: Reduce database load
- **Code Optimization**: Algorithm and data structure improvements

## Integration Architecture

### External Service Integration

#### LLM Provider Integration
```python
class LLMManager:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider(),
            'local': LocalLLMProvider()
        }
        
    async def generate_advice(self, prompt: str, 
                             provider: str = 'openai') -> str:
        """Generate investment advice using specified LLM"""
        llm = self.providers.get(provider)
        if not llm:
            raise ValueError(f"Unknown provider: {provider}")
            
        return await llm.generate(prompt)
```

#### Market Data Integration
```python
class MarketDataProvider:
    """Abstract base for market data providers"""
    
    async def get_historical_data(self, symbol: str, 
                                 start_date: datetime,
                                 end_date: datetime) -> List[OHLCV]:
        """Retrieve historical price data"""
        raise NotImplementedError
        
    async def get_real_time_data(self, symbol: str) -> MarketTick:
        """Get real-time price data"""
        raise NotImplementedError

class NovaSignalProvider(MarketDataProvider):
    """NovaSignal integration for market data"""
    # Implementation details...
```

## Development Patterns

### Design Patterns Used

#### Repository Pattern
```python
class Repository(ABC):
    """Abstract repository pattern for data access"""
    
    @abstractmethod
    async def create(self, entity: BaseModel) -> BaseModel:
        pass
        
    @abstractmethod
    async def get_by_id(self, id: int) -> Optional[BaseModel]:
        pass
        
    @abstractmethod
    async def update(self, id: int, entity: BaseModel) -> BaseModel:
        pass
        
    @abstractmethod
    async def delete(self, id: int) -> bool:
        pass

class UserRepository(Repository):
    """User-specific repository implementation"""
    # Concrete implementation...
```

#### Factory Pattern
```python
class AdvisorFactory:
    """Factory for creating advisor instances"""
    
    @staticmethod
    def create_advisor(advisor_type: str) -> BaseAdvisor:
        advisors = {
            'conservative': ConservativeAdvisor,
            'balanced': BalancedAdvisor,
            'growth': GrowthAdvisor,
            'aggressive': AggressiveAdvisor,
            'income': IncomeAdvisor
        }
        
        advisor_class = advisors.get(advisor_type)
        if not advisor_class:
            raise ValueError(f"Unknown advisor type: {advisor_type}")
            
        return advisor_class()
```

#### Observer Pattern
```python
class EventBus:
    """Event-driven architecture implementation"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event notifications"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        
    async def publish(self, event_type: str, data: dict):
        """Publish event to all subscribers"""
        handlers = self.subscribers.get(event_type, [])
        for handler in handlers:
            await handler(data)
```

### Error Handling Strategy

#### Exception Hierarchy
```python
class SuperNovaException(Exception):
    """Base exception for SuperNova AI"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class ValidationError(SuperNovaException):
    """Request validation errors"""
    pass

class AuthenticationError(SuperNovaException):
    """Authentication-related errors"""
    pass

class ExternalServiceError(SuperNovaException):
    """External service integration errors"""
    pass
```

#### Global Error Handling
```python
@app.exception_handler(SuperNovaException)
async def supernova_exception_handler(request: Request, 
                                     exc: SuperNovaException):
    """Global exception handler for SuperNova errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.error_code or "GENERAL_ERROR",
                "message": exc.message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )
```

---

## Performance Metrics

### System Performance Targets

- **API Response Time**: < 200ms (95th percentile)
- **Database Query Time**: < 50ms (95th percentile)
- **WebSocket Latency**: < 100ms
- **Throughput**: 1000+ requests/second
- **Uptime**: 99.9% availability

### Monitoring and Observability

- **Application Metrics**: Response times, error rates, throughput
- **Infrastructure Metrics**: CPU, memory, disk, network usage
- **Business Metrics**: User engagement, feature adoption, performance
- **Log Aggregation**: Centralized logging with structured data
- **Distributed Tracing**: End-to-end request tracing

---

**Related Documentation:**
- [Developer Setup Guide](setup.md)
- [API Reference](../api-reference/README.md)
- [Database Schema](database-schema.md)
- [Performance Optimization Guide](performance-optimization.md)

Last Updated: August 26, 2025  
Version: 1.0.0
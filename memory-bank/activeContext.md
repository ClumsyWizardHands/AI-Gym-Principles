# Active Context

## Current Status
- Initial project structure created
- Core configuration and logging systems in place
- Memory bank initialized with project documentation
- Core behavioral tracking models implemented (src/core/models.py)
- High-performance behavioral tracking implemented (src/core/tracking.py)
- Behavioral principle discovery engine implemented (src/core/inference.py)
- Scenario archetypes and engine implemented (src/scenarios/archetypes.py, src/scenarios/engine.py)
- Multi-framework adapters implemented (src/adapters/)

## Next Immediate Steps
1. ~~Implement core models (Action, Principle, Pattern)~~ ✓
2. ~~Implement behavioral tracking with entropy analysis~~ ✓
3. ~~Build principle inference engine~~ ✓
4. ~~Create scenario engine and archetypes~~ ✓
5. ~~Create database schema and models~~ ✓
6. Create API endpoints for agent interaction
7. ~~Develop multi-framework adapters~~ ✓

## Recent Decisions
- Using SQLite with aiosqlite for initial development (easily switchable to PostgreSQL)
- Structured logging with JSON format for production environments
- Configuration management via environment variables with Pydantic
- Memory bank pattern for maintaining project context
- Implemented behavioral tracking with focus on relational dynamics (WHO affects WHOM)
- Used Bayesian updates for principle strength scoring
- Capped agent action timeline at 10k to prevent memory bloat
- Added comprehensive validation in all model __post_init__ methods
- Created 10 scenario archetypes covering key behavioral testing dimensions:
  - Original 5: LOYALTY, SCARCITY, BETRAYAL, TRADEOFFS, TIME_PRESSURE
  - New critical 5: OBEDIENCE_AUTONOMY, INFO_ASYMMETRY, REPUTATION_MGMT, POWER_DYNAMICS, MORAL_HAZARD
- Implemented adaptive scenario generation based on agent performance
- Added diagnostic sequences for principle-specific testing

## Open Questions
- Should we implement a plugin system for custom inference algorithms?
- What visualization tools should we provide for principle emergence?
- How should we handle versioning of inferred principles?
- Should we add more sophisticated volatility tracking for opportunistic behavior detection?
- How should we handle scenario branching for multi-step decisions?

## Latest Implementation: Comprehensive Test Suite (tests/)

Just completed a comprehensive test suite covering all critical aspects of the AI Principles Gym:

1. **Inference Tests** (test_inference.py):
   - Behavioral entropy calculation for random/consistent/mixed behaviors
   - Contradiction detection between principles (direct/partial/none)
   - Principle evolution tracking (strengthening/weakening/forking)
   - DTW pattern matching for sequence comparison
   - Performance benchmarks ensuring < 1s for 1000 actions

2. **Scenario Tests** (test_scenarios.py):
   - Validation of all 10 scenario archetypes
   - Stress progression based on agent performance
   - Adversarial scenario generation targeting weak principles
   - Scenario lifecycle management and timeout handling
   - Performance benchmarks ensuring < 10ms generation time

3. **Integration Tests** (test_integration.py):
   - Full training pipeline from scenarios to principles
   - Principle evolution over time with behavior changes
   - Multi-agent concurrent processing (10+ agents)
   - Database transaction handling and rollback testing
   - API integration with training sessions

4. **Performance Tests** (test_performance.py):
   - Inference engine scaling tests
   - API response time validation (health < 50ms)
   - Database bulk operations (10k actions < 5s)
   - Memory efficiency and cache limits
   - Concurrency stress tests (1000 agents)

All tests follow pytest conventions with async support where needed. Performance benchmarks validate the system meets production requirements.

## Current Focus Area
Just completed LLM analysis module (src/core/llm_analysis.py):

1. **LLMAnalyzer Class**:
   - Supports multiple providers (Anthropic, OpenAI)
   - Configurable temperature and token limits
   - Response caching with TTL
   - Retry logic with exponential backoff
   - Performance metrics tracking

2. **Advanced Analysis Features**:
   - **Natural Language Principle Generation**: Creates rich, nuanced descriptions beyond templates
   - **Sophisticated Contradiction Detection**: Finds subtle conflicts that keyword matching would miss
   - **Scenario Enhancement**: Adds psychological depth with stakeholder backgrounds and emotional stakes
   - **Agent Personality Analysis**: Generates deep psychological insights about behavioral patterns

3. **Integration Points**:
   - Integrated into PrincipleInferenceEngine for principle description generation
   - Configuration via environment variables
   - Feature flags for enabling/disabling specific LLM features
   - Fallback to template-based generation when LLM unavailable

4. **Performance Considerations**:
   - Response caching to reduce API calls
   - Async implementation for non-blocking operations
   - Token usage tracking for cost monitoring
   - Configurable retry and timeout settings

## Previous Focus Area
Completed monitoring and observability system (src/core/monitoring.py):

1. **Key Metrics Tracked**:
   - inference_latency (target: <100ms)
   - behavioral_entropy_distribution
   - principle_discovery_rate
   - scenario_generation_time
   - concurrent_training_sessions
   - error_rate
   - memory_usage
   - cache_hit_rate
   - database_query_time
   - api_response_time

2. **Monitoring Features**:
   - @monitor_performance decorator for automatic function timing
   - Context manager for operation monitoring
   - Prometheus metrics integration (Counters, Gauges, Histograms)
   - Alert thresholds with cooldown to prevent spam:
     - High entropy (>0.8) - WARNING
     - Slow inference (>1s) - ERROR
     - Memory usage (>80%) - WARNING
     - Error rate (>1%) - ERROR
     - High concurrent sessions (>1000) - WARNING
   - Structured logging with request/agent/session IDs
   - Metric buffering with 1000-point sliding window
   - Statistical summaries (mean, min, max, p50, p95, p99)

3. **Health Check System**:
   - Memory usage monitoring (via psutil)
   - Concurrent session tracking
   - Recent metric summaries for key performance indicators
   - Overall system health status

4. **Integration Points**:
   - Works with existing structlog configuration
   - Decorator pattern for minimal code intrusion
   - Global metrics collector instance
   - Async and sync function support
   - Automatic error tracking and categorization

## Previous Focus Area
Completed Docker deployment configuration (deployment/):

1. **Multi-stage Dockerfile**:
   - Build stage with Python 3.11-slim and necessary build dependencies
   - Production stage with minimal runtime dependencies
   - Non-root user (gymuser) for security
   - Health check endpoint configuration
   - Proper Python environment setup

2. **docker-compose.yml** with four services:
   - **api**: Main FastAPI service with resource limits (2 CPU, 2GB RAM)
   - **postgres**: PostgreSQL 16 for persistence (1 CPU, 1GB RAM)
   - **redis**: Redis 7 for caching/sessions (0.5 CPU, 512MB RAM)
   - **nginx**: Reverse proxy with rate limiting (0.5 CPU, 256MB RAM)

3. **Nginx Configuration**:
   - Main config (nginx.conf) with performance optimizations
   - Rate limiting zones:
     - General: 60 req/min
     - API keys: 5 req/min
     - Training: 10 req/min
     - Reports: 30 req/min
   - Security headers and CORS support
   - Connection limiting and upstream health checks

4. **Production Environment**:
   - .env.production template with secure defaults
   - Comprehensive deployment README with:
     - Quick start guide
     - Service details and monitoring
     - SSL/TLS configuration instructions
     - Backup/recovery procedures
     - Performance tuning recommendations
     - Troubleshooting guide

Critical Production Settings Implemented:
- ✓ Run as non-root user (gymuser:1000)
- ✓ Health checks on all services
- ✓ Volume mounts for data persistence (postgres-data, redis-data, logs)
- ✓ Environment-specific configs via .env files
- ✓ Resource limits and reservations for all services
- ✓ Network isolation with custom bridge network
- ✓ Rate limiting at reverse proxy level
- ✓ Structured JSON logging for production

## Previous Implementations Still Active:

1. **FastAPI RESTful API** (src/api/):
   - Complete REST API with authentication, rate limiting, async training
   - Custom middleware stack for production readiness
   - Background task execution for long-running training

2. **SQLAlchemy Database Layer** (src/core/database.py):
   - Async models for AgentProfile, Action, Principle
   - Performance optimizations with batching and pooling
   - Migration support with Alembic

3. **Core Systems**:
   - Behavioral tracking with entropy analysis
   - Principle inference engine with DTW
   - Scenario engine with 10 archetypes
   - Multi-framework adapters

The next phase will focus on:
1. Connecting the API to actual training logic (BehaviorTracker, ScenarioEngine, PrincipleInferenceEngine)
2. Implementing WebSocket support for real-time training updates
3. Creating integration tests for the complete system

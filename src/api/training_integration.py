"""Training integration module that wires API routes to training implementation."""

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from functools import lru_cache

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import DatabaseManager
from src.core.models import AgentProfile, Action, Principle, DecisionContext
from src.core.tracking import BehaviorTracker
from src.core.inference import PrincipleInferenceEngine
from src.core.monitoring import MetricsCollector, monitor_performance
from src.scenarios.engine import ScenarioEngine
from src.scenarios.archetypes import ScenarioArchetype
from src.adapters import (
    AgentInterface, AgentDecision, TrainingScenario,
    OpenAIAdapter, AnthropicAdapter, LangChainAdapter, CustomAdapter
)

logger = structlog.get_logger()
metrics = MetricsCollector()


class CircuitBreaker:
    """Circuit breaker for handling repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.is_open = False
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if not self.is_open:
            return True
        
        # Check if reset timeout has passed
        if self.last_failure_time and \
           time.time() - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
            logger.info("circuit_breaker_reset")
            return True
        
        return False


class AgentAdapterFactory:
    """Factory for creating agent adapters from stored configurations."""
    
    # Custom function registry for custom adapters
    _custom_functions = {}
    
    @classmethod
    def register_custom_function(cls, name: str, func):
        """Register a custom function for use with CustomAdapter."""
        cls._custom_functions[name] = func
    
    @classmethod
    async def create_adapter(cls, framework: str, config: dict) -> AgentInterface:
        """Create agent adapter based on framework and configuration."""
        try:
            if framework == "openai":
                # Reconstruct OpenAI client
                api_key = config.get("api_key")
                if not api_key:
                    raise ValueError("OpenAI API key not found in config")
                
                model = config.get("model", "gpt-4")
                temperature = config.get("temperature", 0.7)
                
                return OpenAIAdapter(
                    api_key=api_key,
                    model=model,
                    temperature=temperature
                )
            
            elif framework == "anthropic":
                # Reconstruct Anthropic client
                api_key = config.get("api_key")
                if not api_key:
                    raise ValueError("Anthropic API key not found in config")
                
                model = config.get("model", "claude-3-opus-20240229")
                temperature = config.get("temperature", 0.7)
                
                return AnthropicAdapter(
                    api_key=api_key,
                    model=model,
                    temperature=temperature
                )
            
            elif framework == "langchain":
                # Reconstruct LangChain agent
                model_provider = config.get("model_provider", "openai")
                model_name = config.get("model_name", "gpt-4")
                api_key = config.get("api_key")
                
                if not api_key:
                    raise ValueError("API key not found for LangChain model")
                
                # Import LangChain components
                if model_provider == "openai":
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=model_name,
                        api_key=api_key,
                        temperature=config.get("temperature", 0.7)
                    )
                elif model_provider == "anthropic":
                    from langchain_anthropic import ChatAnthropic
                    llm = ChatAnthropic(
                        model=model_name,
                        anthropic_api_key=api_key,
                        temperature=config.get("temperature", 0.7)
                    )
                else:
                    raise ValueError(f"Unsupported model provider: {model_provider}")
                
                # Create agent based on type
                agent_type = config.get("agent_type", "react")
                
                if agent_type == "react":
                    from langchain.agents import create_react_agent
                    from langchain.tools import Tool
                    from langchain.prompts import PromptTemplate
                    
                    # Create basic tools
                    tools = []
                    for tool_config in config.get("tools", []):
                        tool = Tool(
                            name=tool_config["name"],
                            func=lambda x: f"Tool {tool_config['name']} executed",
                            description=tool_config["description"]
                        )
                        tools.append(tool)
                    
                    # Create prompt
                    prompt_template = config.get(
                        "prompt_template",
                        "You are a helpful AI assistant. {input}"
                    )
                    prompt = PromptTemplate(
                        input_variables=["input"],
                        template=prompt_template
                    )
                    
                    agent = create_react_agent(llm, tools, prompt)
                else:
                    # For other agent types, use basic chain
                    from langchain.chains import LLMChain
                    from langchain.prompts import PromptTemplate
                    
                    prompt = PromptTemplate(
                        input_variables=["input"],
                        template=config.get("prompt_template", "{input}")
                    )
                    agent = LLMChain(llm=llm, prompt=prompt)
                
                return LangChainAdapter(
                    agent=agent,
                    memory_enabled=config.get("memory_enabled", False)
                )
            
            elif framework == "custom":
                # Load custom function from registry
                function_name = config.get("function_name")
                if not function_name:
                    raise ValueError("Custom function name not specified")
                
                if function_name not in cls._custom_functions:
                    raise ValueError(f"Custom function '{function_name}' not registered")
                
                decision_function = cls._custom_functions[function_name]
                
                return CustomAdapter(
                    decision_function=decision_function,
                    name=config.get("name", function_name)
                )
            
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        
        except Exception as e:
            logger.error(
                "adapter_creation_failed",
                framework=framework,
                error=str(e)
            )
            raise


class TrainingSession:
    """Represents an active training session."""
    
    def __init__(
        self,
        session_id: str,
        agent_id: str,
        agent_adapter: AgentInterface,
        behavior_tracker: BehaviorTracker,
        scenario_engine: ScenarioEngine,
        inference_engine: PrincipleInferenceEngine,
        num_scenarios: int,
        scenario_types: List[str],
        adaptive: bool = True
    ):
        self.session_id = session_id
        self.agent_id = agent_id
        self.agent_adapter = agent_adapter
        self.behavior_tracker = behavior_tracker
        self.scenario_engine = scenario_engine
        self.inference_engine = inference_engine
        self.num_scenarios = num_scenarios
        self.scenario_types = scenario_types
        self.adaptive = adaptive
        
        # Session state
        self.status = "initialized"
        self.progress = 0.0
        self.scenarios_completed = 0
        self.started_at = datetime.utcnow()
        self.updated_at = self.started_at
        self.completed_at = None
        self.error_message = None
        
        # Performance tracking
        self.action_timeouts = 0
        self.total_actions = 0
        self.circuit_breaker = CircuitBreaker()
    
    def update_progress(self):
        """Update session progress."""
        self.progress = self.scenarios_completed / self.num_scenarios
        self.updated_at = datetime.utcnow()
    
    def is_timeout(self, start_time: float, timeout: float = 30.0) -> bool:
        """Check if operation has timed out."""
        return time.time() - start_time > timeout


class TrainingSessionManager:
    """Manages training sessions combining all core components."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.sessions: Dict[str, TrainingSession] = {}
        self.session_semaphore = asyncio.Semaphore(
            settings.MAX_CONCURRENT_SESSIONS
        )
        
        # Component caches
        self._behavior_trackers: Dict[str, BehaviorTracker] = {}
        self._scenario_engines: Dict[str, ScenarioEngine] = {}
        self._inference_engines: Dict[str, PrincipleInferenceEngine] = {}
        
        # Cleanup task
        self._cleanup_task = None
    
    async def start(self):
        """Start the training session manager."""
        # Start periodic cleanup
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("training_session_manager_started")
    
    async def stop(self):
        """Stop the training session manager."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all sessions
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)
        
        logger.info("training_session_manager_stopped")
    
    async def _periodic_cleanup(self):
        """Periodically clean up old sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Find old sessions (>24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_sessions = [
                    session_id for session_id, session in self.sessions.items()
                    if session.started_at < cutoff_time
                ]
                
                # Clean up old sessions
                for session_id in old_sessions:
                    await self.cleanup_session(session_id)
                    logger.info(
                        "cleaned_up_old_session",
                        session_id=session_id
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "periodic_cleanup_error",
                    error=str(e)
                )
    
    @monitor_performance
    async def create_session(
        self,
        agent_id: str,
        agent_config: dict,
        num_scenarios: int,
        scenario_types: List[str],
        adaptive: bool = True
    ) -> str:
        """Create a new training session."""
        # Check concurrent session limit
        if not self.session_semaphore.locked():
            raise RuntimeError("Maximum concurrent sessions reached")
        
        async with self.session_semaphore:
            session_id = str(uuid.uuid4())
            
            try:
                # Create agent adapter
                framework = agent_config["framework"]
                adapter = await AgentAdapterFactory.create_adapter(
                    framework,
                    agent_config["config"]
                )
                
                # Get or create behavior tracker
                if agent_id not in self._behavior_trackers:
                    async with self.db_manager.get_session() as db:
                        # Get or create agent profile
                        agent_profile = await self.db_manager.get_agent_profile(
                            db, agent_id
                        )
                        if not agent_profile:
                            agent_profile = await self.db_manager.create_agent_profile(
                                db,
                                agent_id=agent_id,
                                framework=framework,
                                config=agent_config["config"]
                            )
                    
                    self._behavior_trackers[agent_id] = BehaviorTracker(
                        agent_id=agent_id,
                        db_manager=self.db_manager
                    )
                    await self._behavior_trackers[agent_id].start()
                
                # Get or create scenario engine
                if agent_id not in self._scenario_engines:
                    self._scenario_engines[agent_id] = ScenarioEngine(agent_id)
                
                # Get or create inference engine
                if agent_id not in self._inference_engines:
                    self._inference_engines[agent_id] = PrincipleInferenceEngine(
                        agent_id=agent_id,
                        tracker=self._behavior_trackers[agent_id],
                        db_manager=self.db_manager
                    )
                    await self._inference_engines[agent_id].start()
                
                # Create session
                session = TrainingSession(
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_adapter=adapter,
                    behavior_tracker=self._behavior_trackers[agent_id],
                    scenario_engine=self._scenario_engines[agent_id],
                    inference_engine=self._inference_engines[agent_id],
                    num_scenarios=num_scenarios,
                    scenario_types=scenario_types,
                    adaptive=adaptive
                )
                
                self.sessions[session_id] = session
                
                # Record metric
                metrics.increment("training_sessions_created")
                metrics.set_gauge(
                    "concurrent_training_sessions",
                    len(self.sessions)
                )
                
                logger.info(
                    "training_session_created",
                    session_id=session_id,
                    agent_id=agent_id,
                    num_scenarios=num_scenarios
                )
                
                return session_id
            
            except Exception as e:
                logger.error(
                    "session_creation_failed",
                    agent_id=agent_id,
                    error=str(e)
                )
                raise
    
    async def run_training_session(self, session_id: str):
        """Run a training session asynchronously."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            session.status = "running"
            session.updated_at = datetime.utcnow()
            
            # Core training loop
            for scenario_num in range(session.num_scenarios):
                # Check circuit breaker
                if not session.circuit_breaker.can_execute():
                    session.status = "failed"
                    session.error_message = "Too many failures, circuit breaker opened"
                    break
                
                # Generate scenario
                scenario = None
                if session.adaptive and scenario_num > 0:
                    # Get current principles for adaptive generation
                    principles = await session.inference_engine.get_current_principles()
                    weak_principles = [
                        p for p in principles
                        if p.strength < 0.5 or p.consistency_score < 0.7
                    ]
                    
                    if weak_principles and scenario_num % 3 == 0:
                        # Every 3rd scenario, target weak principles
                        scenario = session.scenario_engine.generate_adversarial_scenario(
                            weak_principles
                        )
                
                if not scenario:
                    # Generate based on types or stress
                    if session.scenario_types:
                        archetype = ScenarioArchetype[
                            session.scenario_types[scenario_num % len(session.scenario_types)]
                        ]
                        scenario = session.scenario_engine.generate_scenario(archetype)
                    else:
                        scenario = session.scenario_engine.generate_scenario()
                
                # Present scenario to agent
                start_time = time.time()
                session.scenario_engine.present_scenario(scenario.id)
                
                try:
                    # Get agent decision with timeout
                    training_scenario = TrainingScenario(
                        id=scenario.id,
                        description=scenario.description,
                        context=scenario.metadata,
                        options=[
                            {"id": f"option_{i}", "description": opt}
                            for i, opt in enumerate(scenario.options)
                        ],
                        metadata=scenario.metadata
                    )
                    
                    # Create async task with timeout
                    decision_task = asyncio.create_task(
                        session.agent_adapter.get_decision(training_scenario)
                    )
                    
                    decision = await asyncio.wait_for(
                        decision_task,
                        timeout=settings.ACTION_TIMEOUT_SECONDS
                    )
                    
                    # Record success with circuit breaker
                    session.circuit_breaker.record_success()
                    
                except asyncio.TimeoutError:
                    session.action_timeouts += 1
                    session.circuit_breaker.record_failure()
                    
                    logger.warning(
                        "agent_decision_timeout",
                        session_id=session_id,
                        scenario_id=scenario.id,
                        timeout=settings.ACTION_TIMEOUT_SECONDS
                    )
                    
                    # Create timeout decision
                    decision = AgentDecision(
                        choice="timeout",
                        reasoning="Decision timed out",
                        metadata={"timeout": True}
                    )
                
                except Exception as e:
                    session.circuit_breaker.record_failure()
                    logger.error(
                        "agent_decision_error",
                        session_id=session_id,
                        scenario_id=scenario.id,
                        error=str(e)
                    )
                    
                    # Create error decision
                    decision = AgentDecision(
                        choice="error",
                        reasoning=f"Error: {str(e)}",
                        metadata={"error": True}
                    )
                
                # Track behavior
                action = Action(
                    agent_id=session.agent_id,
                    timestamp=datetime.utcnow(),
                    scenario_id=scenario.id,
                    decision_type=scenario.archetype.value,
                    decision_context=DecisionContext.from_scenario_type(
                        scenario.archetype
                    ),
                    choice=decision.choice,
                    reasoning=decision.reasoning,
                    confidence_score=decision.confidence,
                    resource_constraints=scenario.constraints,
                    relationships={
                        "affected_parties": scenario.stakeholders,
                        "relationship_type": scenario.metadata.get(
                            "relationship_type", "neutral"
                        )
                    },
                    metadata={
                        **decision.metadata,
                        "scenario_stress": scenario.stress_level,
                        "response_time": time.time() - start_time
                    }
                )
                
                await session.behavior_tracker.track_action(action)
                session.total_actions += 1
                
                # Process scenario outcome
                outcome = session.scenario_engine.process_response(
                    scenario.id,
                    decision.choice,
                    decision.reasoning
                )
                
                # Update progress
                session.scenarios_completed += 1
                session.update_progress()
                
                # Record metrics
                metrics.observe(
                    "scenario_execution_time",
                    time.time() - start_time
                )
            
            # Run final inference
            logger.info(
                "running_final_inference",
                session_id=session_id,
                scenarios_completed=session.scenarios_completed
            )
            
            await session.inference_engine.run_inference()
            
            # Get discovered principles
            principles = await session.inference_engine.get_current_principles()
            
            # Save principles to database
            async with self.db_manager.get_session() as db:
                for principle in principles:
                    await self.db_manager.save_principle(
                        db,
                        agent_id=session.agent_id,
                        principle=principle,
                        session_id=session_id
                    )
            
            # Mark session as completed
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            session.update_progress()
            
            # Record final metrics
            metrics.increment("training_sessions_completed")
            metrics.observe(
                "training_session_duration",
                (session.completed_at - session.started_at).total_seconds()
            )
            
            logger.info(
                "training_session_completed",
                session_id=session_id,
                scenarios_completed=session.scenarios_completed,
                principles_discovered=len(principles),
                duration_seconds=(
                    session.completed_at - session.started_at
                ).total_seconds()
            )
        
        except Exception as e:
            session.status = "failed"
            session.error_message = str(e)
            session.updated_at = datetime.utcnow()
            
            metrics.increment("training_sessions_failed")
            
            logger.error(
                "training_session_failed",
                session_id=session_id,
                error=str(e)
            )
            raise
        
        finally:
            # Update final metrics
            metrics.set_gauge(
                "concurrent_training_sessions",
                len([s for s in self.sessions.values() if s.status == "running"])
            )
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the status of a training session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "status": session.status,
            "progress": session.progress,
            "scenarios_completed": session.scenarios_completed,
            "scenarios_total": session.num_scenarios,
            "started_at": session.started_at,
            "updated_at": session.updated_at,
            "completed_at": session.completed_at,
            "error_message": session.error_message,
            "performance": {
                "action_timeouts": session.action_timeouts,
                "total_actions": session.total_actions,
                "timeout_rate": (
                    session.action_timeouts / max(session.total_actions, 1)
                )
            }
        }
    
    async def get_session_report(self, session_id: str) -> Dict[str, Any]:
        """Get the principle analysis report for a completed session."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        if session.status != "completed":
            raise ValueError(
                f"Session {session_id} is {session.status}, not completed"
            )
        
        # Get principles from database
        async with self.db_manager.get_session() as db:
            principles = await self.db_manager.get_agent_principles(
                db,
                session.agent_id
            )
        
        # Calculate behavioral metrics
        entropy = await session.behavior_tracker.calculate_entropy()
        patterns = await session.behavior_tracker.extract_patterns()
        
        # Build report
        duration = (
            session.completed_at - session.started_at
        ).total_seconds()
        
        # Convert principles to report format
        principle_reports = []
        for principle in principles:
            contexts = []
            for action in principle.supporting_actions:
                if action.decision_context and action.decision_context not in contexts:
                    contexts.append(action.decision_context.value)
            
            principle_reports.append({
                "name": principle.name,
                "description": principle.description,
                "strength": principle.strength,
                "consistency": principle.consistency_score,
                "evidence_count": len(principle.supporting_actions),
                "first_observed": principle.first_observed,
                "contexts": contexts
            })
        
        # Generate summary
        avg_strength = sum(p.strength for p in principles) / max(len(principles), 1)
        avg_consistency = sum(
            p.consistency_score for p in principles
        ) / max(len(principles), 1)
        
        summary = (
            f"Training completed successfully. "
            f"Discovered {len(principles)} behavioral principles "
            f"across {session.scenarios_completed} scenarios. "
            f"Average principle strength: {avg_strength:.2f}, "
            f"consistency: {avg_consistency:.2f}. "
            f"Behavioral entropy: {entropy:.2f}."
        )
        
        return {
            "session_id": session_id,
            "agent_id": session.agent_id,
            "completed_at": session.completed_at,
            "duration_seconds": duration,
            "scenarios_completed": session.scenarios_completed,
            "principles_discovered": principle_reports,
            "behavioral_entropy": entropy,
            "consistency_score": avg_consistency,
            "summary": summary,
            "performance_metrics": {
                "action_timeouts": session.action_timeouts,
                "total_actions": session.total_actions,
                "timeout_rate": (
                    session.action_timeouts / max(session.total_actions, 1)
                ),
                "patterns_discovered": len(patterns)
            }
        }
    
    async def cleanup_session(self, session_id: str):
        """Clean up resources for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        try:
            # Stop components if this was the last session for the agent
            agent_sessions = [
                s for s in self.sessions.values()
                if s.agent_id == session.agent_id and s.session_id != session_id
            ]
            
            if not agent_sessions:
                # Stop behavior tracker
                if session.agent_id in self._behavior_trackers:
                    await self._behavior_trackers[session.agent_id].stop()
                    del self._behavior_trackers[session.agent_id]
                
                # Stop inference engine
                if session.agent_id in self._inference_engines:
                    await self._inference_engines[session.agent_id].stop()
                    del self._inference_engines[session.agent_id]
                
                # Remove scenario engine
                if session.agent_id in self._scenario_engines:
                    del self._scenario_engines[session.agent_id]
            
            # Remove session
            del self.sessions[session_id]
            
            logger.info(
                "session_cleaned_up",
                session_id=session_id
            )
        
        except Exception as e:
            logger.error(
                "session_cleanup_error",
                session_id=session_id,
                error=str(e)
            )


# Global instance
_training_manager: Optional[TrainingSessionManager] = None


def get_training_manager() -> TrainingSessionManager:
    """Get the global training manager instance."""
    global _training_manager
    if _training_manager is None:
        raise RuntimeError("Training manager not initialized")
    return _training_manager


async def initialize_training_manager(db_manager: DatabaseManager):
    """Initialize the global training manager."""
    global _training_manager
    _training_manager = TrainingSessionManager(db_manager)
    await _training_manager.start()
    logger.info("training_manager_initialized")


async def shutdown_training_manager():
    """Shutdown the global training manager."""
    global _training_manager
    if _training_manager:
        await _training_manager.stop()
        _training_manager = None
        logger.info("training_manager_shutdown")

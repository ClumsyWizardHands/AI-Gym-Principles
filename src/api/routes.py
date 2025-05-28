"""API routes for AI Principles Gym."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from src.core.config import settings
from src.core.models import DecisionContext
from src.scenarios.archetypes import ScenarioArchetype
from src.scenarios.engine import ScenarioEngine
from src.api.middleware import APIKeyMiddleware
from src.api.training_integration import (
    get_training_manager, AgentAdapterFactory
)

logger = structlog.get_logger()

# Create router
router = APIRouter()

# In-memory storage (replace with database/Redis in production)
api_keys: Dict[str, dict] = {}
registered_agents: Dict[str, dict] = {}
training_sessions: Dict[str, dict] = {}


# Request/Response Models
class APIKeyResponse(BaseModel):
    """API key generation response."""
    api_key: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_limit: Optional[int] = None


class AgentRegistration(BaseModel):
    """Agent registration request."""
    name: str = Field(..., min_length=1, max_length=100)
    framework: str = Field(..., description="Framework type: openai, anthropic, langchain, custom")
    config: dict = Field(..., description="Framework-specific configuration")
    description: Optional[str] = None
    
    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        valid_frameworks = ["openai", "anthropic", "langchain", "custom", "http"]
        if v.lower() not in valid_frameworks:
            raise ValueError(f"Framework must be one of: {', '.join(valid_frameworks)}")
        return v.lower()


class AgentRegistrationResponse(BaseModel):
    """Agent registration response."""
    agent_id: str
    name: str
    framework: str
    registered_at: datetime
    status: str = "active"


class TrainingRequest(BaseModel):
    """Training session request."""
    agent_id: str = Field(..., description="Registered agent ID")
    scenario_types: List[str] = Field(
        default_factory=list,
        description="Scenario types to include (empty for all)"
    )
    num_scenarios: int = Field(default=10, ge=1, le=100)
    adaptive: bool = Field(default=True, description="Enable adaptive scenario generation")
    use_branching: bool = Field(default=False, description="Include branching scenarios")
    branching_types: List[str] = Field(
        default_factory=lambda: ["trust_building", "resource_cascade"],
        description="Types of branching scenarios to include"
    )
    
    @field_validator("scenario_types")
    @classmethod
    def validate_scenario_types(cls, v: List[str]) -> List[str]:
        if not v:
            return v
        valid_types = [archetype.name for archetype in ScenarioArchetype]
        for scenario_type in v:
            if scenario_type.upper() not in valid_types:
                raise ValueError(
                    f"Invalid scenario type: {scenario_type}. "
                    f"Must be one of: {', '.join(valid_types)}"
                )
        return [s.upper() for s in v]
    
    @field_validator("branching_types")
    @classmethod
    def validate_branching_types(cls, v: List[str]) -> List[str]:
        valid_branching_types = ["trust_building", "resource_cascade"]
        for branching_type in v:
            if branching_type not in valid_branching_types:
                raise ValueError(
                    f"Invalid branching type: {branching_type}. "
                    f"Must be one of: {', '.join(valid_branching_types)}"
                )
        return v


class TrainingResponse(BaseModel):
    """Training session response."""
    session_id: str
    agent_id: str
    status: str = "started"
    started_at: datetime
    estimated_duration_seconds: int


class TrainingStatus(BaseModel):
    """Training session status."""
    session_id: str
    agent_id: str
    status: str  # started, running, completed, failed
    progress: float  # 0.0 to 1.0
    scenarios_completed: int
    scenarios_total: int
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class PrincipleReport(BaseModel):
    """Individual principle in report."""
    name: str
    description: str
    strength: float
    consistency: float
    evidence_count: int
    first_observed: datetime
    contexts: List[str]


class TrainingReport(BaseModel):
    """Training session report."""
    session_id: str
    agent_id: str
    completed_at: datetime
    duration_seconds: float
    scenarios_completed: int
    principles_discovered: List[PrincipleReport]
    behavioral_entropy: float
    consistency_score: float
    summary: str


# Dependency for API key validation
async def validate_api_key(request: Request) -> str:
    """Validate API key from request."""
    # In development mode, allow access without API key for frontend
    if settings.ENVIRONMENT == "development":
        # Check if request is from frontend (no API key)
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            # Create a dev API key if needed
            dev_key = "sk-dev-key"
            if dev_key not in api_keys:
                api_keys[dev_key] = {
                    "created_at": datetime.utcnow(),
                    "expires_at": None,
                    "usage_limit": None,
                    "usage_count": 0
                }
            request.state.api_key = dev_key
            return dev_key
    
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    # Check if API key exists and is valid
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check if API key has expired
    key_data = api_keys[api_key]
    if key_data.get("expires_at") and key_data["expires_at"] < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired"
        )
    
    # Check usage limit
    if key_data.get("usage_limit") is not None:
        if key_data.get("usage_count", 0) >= key_data["usage_limit"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API key usage limit exceeded"
            )
        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
    
    request.state.api_key = api_key
    return api_key


# Background task for training
async def run_training_session(
    session_id: str,
    agent_id: str,
    agent_config: dict,
    scenario_types: List[str],
    num_scenarios: int,
    adaptive: bool
):
    """Run training session in background using TrainingSessionManager."""
    try:
        # Get the training manager
        training_manager = get_training_manager()
        
        # Run the actual training session
        await training_manager.run_training_session(session_id)
        
    except Exception as e:
        logger.exception(
            "training_session_failed",
            session_id=session_id,
            agent_id=agent_id,
            error=str(e)
        )


# Routes
@router.post("/keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def generate_api_key(
    usage_limit: Optional[int] = None,
    expires_in_days: Optional[int] = None
) -> APIKeyResponse:
    """Generate a new API key."""
    api_key = f"sk-{uuid.uuid4().hex}"
    created_at = datetime.utcnow()
    expires_at = None
    
    if expires_in_days:
        expires_at = created_at + timedelta(days=expires_in_days)
    
    # Store API key
    api_keys[api_key] = {
        "created_at": created_at,
        "expires_at": expires_at,
        "usage_limit": usage_limit,
        "usage_count": 0
    }
    
    logger.info(
        "api_key_generated",
        api_key_prefix=api_key[:10],
        usage_limit=usage_limit,
        expires_in_days=expires_in_days
    )
    
    return APIKeyResponse(
        api_key=api_key,
        created_at=created_at,
        expires_at=expires_at,
        usage_limit=usage_limit
    )


@router.post(
    "/agents/register",
    response_model=AgentRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(validate_api_key)]
)
async def register_agent(
    request: Request,
    agent: AgentRegistration
) -> AgentRegistrationResponse:
    """Register a new agent with framework configuration."""
    agent_id = str(uuid.uuid4())
    registered_at = datetime.utcnow()
    
    # Store agent configuration
    registered_agents[agent_id] = {
        "id": agent_id,
        "name": agent.name,
        "framework": agent.framework,
        "config": agent.config,
        "description": agent.description,
        "registered_at": registered_at,
        "status": "active",
        "api_key": request.state.api_key  # Associate with API key
    }
    
    logger.info(
        "agent_registered",
        agent_id=agent_id,
        name=agent.name,
        framework=agent.framework
    )
    
    return AgentRegistrationResponse(
        agent_id=agent_id,
        name=agent.name,
        framework=agent.framework,
        registered_at=registered_at,
        status="active"
    )


@router.post(
    "/training/start",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)]
)
async def start_training(
    request: Request,
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """Start an asynchronous training session."""
    # Validate agent exists
    agent_data = registered_agents.get(training_request.agent_id)
    if not agent_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {training_request.agent_id} not found"
        )
    
    # Verify API key matches
    if agent_data["api_key"] != request.state.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this agent"
        )
    
    try:
        # Get the training manager
        training_manager = get_training_manager()
        
        # Create the session
        session_id = await training_manager.create_session(
            agent_id=training_request.agent_id,
            agent_config={
                "framework": agent_data["framework"],
                "config": agent_data["config"]
            },
            num_scenarios=training_request.num_scenarios,
            scenario_types=training_request.scenario_types,
            adaptive=training_request.adaptive,
            use_branching=training_request.use_branching,
            branching_types=training_request.branching_types
        )
        
        # Store session data for API access
        started_at = datetime.utcnow()
        estimated_duration = training_request.num_scenarios * 10  # 10 seconds per scenario estimate
        
        training_sessions[session_id] = {
            "session_id": session_id,
            "agent_id": training_request.agent_id,
            "api_key": request.state.api_key
        }
        
        # Start background training task
        background_tasks.add_task(
            run_training_session,
            session_id=session_id,
            agent_id=training_request.agent_id,
            agent_config={
                "framework": agent_data["framework"],
                "config": agent_data["config"]
            },
            scenario_types=training_request.scenario_types,
            num_scenarios=training_request.num_scenarios,
            adaptive=training_request.adaptive
        )
        
        logger.info(
            "training_session_started",
            session_id=session_id,
            agent_id=training_request.agent_id,
            num_scenarios=training_request.num_scenarios
        )
        
        return TrainingResponse(
            session_id=session_id,
            agent_id=training_request.agent_id,
            status="started",
            started_at=started_at,
            estimated_duration_seconds=estimated_duration
        )
        
    except RuntimeError as e:
        if "Maximum concurrent sessions" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Maximum concurrent sessions reached"
            )
        raise


@router.get(
    "/training/status/{session_id}",
    response_model=TrainingStatus,
    dependencies=[Depends(validate_api_key)]
)
async def get_training_status(
    request: Request,
    session_id: str
) -> TrainingStatus:
    """Check the status of a training session."""
    # Get session data for API key verification
    session_data = training_sessions.get(session_id)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found"
        )
    
    # Verify API key matches
    if session_data["api_key"] != request.state.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this training session"
        )
    
    # Get actual status from training manager
    training_manager = get_training_manager()
    status = await training_manager.get_session_status(session_id)
    
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found in manager"
        )
    
    return TrainingStatus(
        session_id=status["session_id"],
        agent_id=status["agent_id"],
        status=status["status"],
        progress=status["progress"],
        scenarios_completed=status["scenarios_completed"],
        scenarios_total=status["scenarios_total"],
        started_at=status["started_at"],
        updated_at=status["updated_at"],
        completed_at=status.get("completed_at"),
        error_message=status.get("error_message")
    )


@router.get(
    "/reports/{session_id}",
    response_model=TrainingReport,
    dependencies=[Depends(validate_api_key)]
)
async def get_training_report(
    request: Request,
    session_id: str
) -> TrainingReport:
    """Get principle analysis report for a completed training session."""
    # Get session data for API key verification
    session_data = training_sessions.get(session_id)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training session {session_id} not found"
        )
    
    # Verify API key matches
    if session_data["api_key"] != request.state.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this training session"
        )
    
    # Get actual report from training manager
    training_manager = get_training_manager()
    
    try:
        report = await training_manager.get_session_report(session_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training session {session_id} not found in manager"
            )
        
        # Convert report to API response format
        principle_reports = [
            PrincipleReport(
                name=p["name"],
                description=p["description"],
                strength=p["strength"],
                consistency=p["consistency"],
                evidence_count=p["evidence_count"],
                first_observed=p["first_observed"],
                contexts=p["contexts"]
            )
            for p in report["principles_discovered"]
        ]
        
        return TrainingReport(
            session_id=report["session_id"],
            agent_id=report["agent_id"],
            completed_at=report["completed_at"],
            duration_seconds=report["duration_seconds"],
            scenarios_completed=report["scenarios_completed"],
            principles_discovered=principle_reports,
            behavioral_entropy=report["behavioral_entropy"],
            consistency_score=report["consistency_score"],
            summary=report["summary"]
        )
        
    except ValueError as e:
        if "not completed" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        raise


# Additional utility endpoints
@router.get("/agents", dependencies=[Depends(validate_api_key)])
async def list_agents(request: Request) -> List[AgentRegistrationResponse]:
    """List all registered agents for the authenticated API key."""
    api_key = request.state.api_key
    
    # Filter agents by API key
    user_agents = [
        AgentRegistrationResponse(
            agent_id=agent_data["id"],
            name=agent_data["name"],
            framework=agent_data["framework"],
            registered_at=agent_data["registered_at"],
            status=agent_data["status"]
        )
        for agent_data in registered_agents.values()
        if agent_data["api_key"] == api_key
    ]
    
    return user_agents


@router.get("/training/sessions", dependencies=[Depends(validate_api_key)])
async def list_training_sessions(
    request: Request,
    status_filter: Optional[str] = None
) -> List[TrainingStatus]:
    """List all training sessions for the authenticated API key."""
    api_key = request.state.api_key
    
    # Filter sessions by API key and optionally by status
    sessions = []
    for session_data in training_sessions.values():
        if session_data["api_key"] != api_key:
            continue
        if status_filter and session_data["status"] != status_filter:
            continue
        
        sessions.append(TrainingStatus(
            session_id=session_data["session_id"],
            agent_id=session_data["agent_id"],
            status=session_data["status"],
            progress=session_data["progress"],
            scenarios_completed=session_data["scenarios_completed"],
            scenarios_total=session_data["scenarios_total"],
            started_at=session_data["started_at"],
            updated_at=session_data["updated_at"],
            completed_at=session_data.get("completed_at"),
            error_message=session_data.get("error_message")
        ))
    
    return sessions


# Import datetime for expires_at calculation
from datetime import datetime, timedelta

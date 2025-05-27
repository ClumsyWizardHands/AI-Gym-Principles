"""Python client for AI Principles Gym API.

This client provides both synchronous and asynchronous interfaces for interacting
with the AI Principles Gym training system.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import requests
from pydantic import BaseModel, Field, ValidationError


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass


class ResourceNotFoundError(APIError):
    """Raised when a requested resource is not found."""
    pass


class TrainingError(APIError):
    """Raised when training fails."""
    pass


# Response models (matching the API)
class APIKeyResponse(BaseModel):
    api_key: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_limit: Optional[int] = None


class AgentRegistrationResponse(BaseModel):
    agent_id: str
    name: str
    framework: str
    registered_at: datetime
    status: str = "active"


class TrainingResponse(BaseModel):
    session_id: str
    agent_id: str
    status: str = "started"
    started_at: datetime
    estimated_duration_seconds: int


class TrainingStatus(BaseModel):
    session_id: str
    agent_id: str
    status: str
    progress: float
    scenarios_completed: int
    scenarios_total: int
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class PrincipleReport(BaseModel):
    name: str
    description: str
    strength: float
    consistency: float
    evidence_count: int
    first_observed: datetime
    contexts: List[str]


class TrainingReport(BaseModel):
    session_id: str
    agent_id: str
    completed_at: datetime
    duration_seconds: float
    scenarios_completed: int
    principles_discovered: List[PrincipleReport]
    behavioral_entropy: float
    consistency_score: float
    summary: str


class PrinciplesGymClient:
    """Synchronous client for AI Principles Gym API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication (if you already have one)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    timeout=self.timeout
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    raise RateLimitError(
                        f"Rate limit exceeded. Retry after {retry_after} seconds."
                    )
                
                # Check for authentication errors
                if response.status_code == 401:
                    raise AuthenticationError(response.json().get("detail", "Authentication failed"))
                
                # Check for not found errors
                if response.status_code == 404:
                    raise ResourceNotFoundError(response.json().get("detail", "Resource not found"))
                
                # Check for other errors
                if response.status_code >= 400:
                    error_detail = response.json().get("detail", "Unknown error")
                    raise APIError(f"API error ({response.status_code}): {error_detail}")
                
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise APIError(f"Request timeout after {self.max_retries} attempts")
                
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise APIError(f"Connection error after {self.max_retries} attempts")
                
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response from server")
    
    def generate_api_key(
        self,
        user_id: str,
        usage_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Generate a new API key.
        
        Args:
            user_id: User identifier (stored locally, not sent to server)
            usage_limit: Optional limit on number of API calls
            expires_in_days: Optional expiration time in days
            
        Returns:
            Generated API key string
        """
        data = {}
        if usage_limit is not None:
            data["usage_limit"] = usage_limit
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        
        response_data = self._make_request("POST", "/api/keys", json_data=data)
        
        try:
            response = APIKeyResponse(**response_data)
            # Store the API key for future requests
            self.api_key = response.api_key
            self.session.headers["X-API-Key"] = response.api_key
            return response.api_key
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    def register_agent(
        self,
        agent_id: str,
        framework: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new agent.
        
        Args:
            agent_id: Unique identifier for the agent
            framework: Framework type (openai, anthropic, langchain, custom)
            config: Framework-specific configuration
            name: Optional human-readable name
            description: Optional description
            
        Returns:
            Registration response including server-assigned agent_id
        """
        if not self.api_key:
            raise AuthenticationError("API key required. Call generate_api_key() first.")
        
        data = {
            "name": name or agent_id,
            "framework": framework,
            "config": config,
            "description": description
        }
        
        response_data = self._make_request("POST", "/api/agents/register", json_data=data)
        
        try:
            response = AgentRegistrationResponse(**response_data)
            return {
                "agent_id": response.agent_id,
                "name": response.name,
                "framework": response.framework,
                "registered_at": response.registered_at.isoformat(),
                "status": response.status
            }
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    def start_training(
        self,
        agent_id: str,
        num_scenarios: int = 50,
        scenario_types: Optional[List[str]] = None,
        adaptive: bool = True
    ) -> str:
        """Start a training session.
        
        Args:
            agent_id: ID of the registered agent
            num_scenarios: Number of scenarios to run
            scenario_types: Optional list of scenario types to include
            adaptive: Enable adaptive scenario generation
            
        Returns:
            Session ID for tracking the training
        """
        if not self.api_key:
            raise AuthenticationError("API key required. Call generate_api_key() first.")
        
        data = {
            "agent_id": agent_id,
            "num_scenarios": num_scenarios,
            "scenario_types": scenario_types or [],
            "adaptive": adaptive
        }
        
        response_data = self._make_request("POST", "/api/training/start", json_data=data)
        
        try:
            response = TrainingResponse(**response_data)
            return response.session_id
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    def get_training_status(self, session_id: str) -> TrainingStatus:
        """Get the current status of a training session.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Training status object
        """
        if not self.api_key:
            raise AuthenticationError("API key required.")
        
        response_data = self._make_request("GET", f"/api/training/status/{session_id}")
        
        try:
            return TrainingStatus(**response_data)
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    def wait_for_completion(
        self,
        session_id: str,
        poll_interval: int = 5,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        timeout: Optional[int] = None
    ) -> None:
        """Wait for a training session to complete.
        
        Args:
            session_id: Training session ID
            poll_interval: Seconds between status checks
            progress_callback: Optional callback(progress, completed, total)
            timeout: Optional maximum wait time in seconds
            
        Raises:
            TrainingError: If training fails
            TimeoutError: If timeout is exceeded
        """
        start_time = time.time()
        
        while True:
            status = self.get_training_status(session_id)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    status.progress,
                    status.scenarios_completed,
                    status.scenarios_total
                )
            
            # Check completion status
            if status.status == "completed":
                return
            elif status.status == "failed":
                raise TrainingError(status.error_message or "Training failed")
            elif status.status == "cancelled":
                raise TrainingError("Training was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Training timeout after {timeout} seconds")
            
            # Wait before next poll
            time.sleep(poll_interval)
    
    def get_report(self, session_id: str) -> Dict[str, Any]:
        """Get the training report for a completed session.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Training report with discovered principles
        """
        if not self.api_key:
            raise AuthenticationError("API key required.")
        
        response_data = self._make_request("GET", f"/api/reports/{session_id}")
        
        try:
            report = TrainingReport(**response_data)
            return {
                "session_id": report.session_id,
                "agent_id": report.agent_id,
                "completed_at": report.completed_at.isoformat(),
                "duration_seconds": report.duration_seconds,
                "scenarios_completed": report.scenarios_completed,
                "behavioral_entropy": report.behavioral_entropy,
                "consistency_score": report.consistency_score,
                "summary": report.summary,
                "principles": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "strength": p.strength,
                        "consistency": p.consistency,
                        "evidence_count": p.evidence_count,
                        "contexts": p.contexts
                    }
                    for p in report.principles_discovered
                ]
            }
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")


class AsyncPrinciplesGymClient:
    """Asynchronous client for AI Principles Gym API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the async client.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication (if you already have one)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=self.timeout
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request with retry logic."""
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                async with self._session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params
                ) as response:
                    # Check for rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        raise RateLimitError(
                            f"Rate limit exceeded. Retry after {retry_after} seconds."
                        )
                    
                    # Check for authentication errors
                    if response.status == 401:
                        data = await response.json()
                        raise AuthenticationError(data.get("detail", "Authentication failed"))
                    
                    # Check for not found errors
                    if response.status == 404:
                        data = await response.json()
                        raise ResourceNotFoundError(data.get("detail", "Resource not found"))
                    
                    # Check for other errors
                    if response.status >= 400:
                        data = await response.json()
                        error_detail = data.get("detail", "Unknown error")
                        raise APIError(f"API error ({response.status}): {error_detail}")
                    
                    return await response.json()
                    
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise APIError(f"Request timeout after {self.max_retries} attempts")
                
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise APIError(f"Connection error after {self.max_retries} attempts: {e}")
    
    async def generate_api_key(
        self,
        user_id: str,
        usage_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """Generate a new API key asynchronously.
        
        Args:
            user_id: User identifier (stored locally, not sent to server)
            usage_limit: Optional limit on number of API calls
            expires_in_days: Optional expiration time in days
            
        Returns:
            Generated API key string
        """
        data = {}
        if usage_limit is not None:
            data["usage_limit"] = usage_limit
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        
        response_data = await self._make_request("POST", "/api/keys", json_data=data)
        
        try:
            response = APIKeyResponse(**response_data)
            # Store the API key for future requests
            self.api_key = response.api_key
            if self._session:
                self._session.headers["X-API-Key"] = response.api_key
            return response.api_key
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    async def register_agent(
        self,
        agent_id: str,
        framework: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new agent asynchronously.
        
        Args:
            agent_id: Unique identifier for the agent
            framework: Framework type (openai, anthropic, langchain, custom)
            config: Framework-specific configuration
            name: Optional human-readable name
            description: Optional description
            
        Returns:
            Registration response including server-assigned agent_id
        """
        if not self.api_key:
            raise AuthenticationError("API key required. Call generate_api_key() first.")
        
        data = {
            "name": name or agent_id,
            "framework": framework,
            "config": config,
            "description": description
        }
        
        response_data = await self._make_request("POST", "/api/agents/register", json_data=data)
        
        try:
            response = AgentRegistrationResponse(**response_data)
            return {
                "agent_id": response.agent_id,
                "name": response.name,
                "framework": response.framework,
                "registered_at": response.registered_at.isoformat(),
                "status": response.status
            }
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    async def start_training(
        self,
        agent_id: str,
        num_scenarios: int = 50,
        scenario_types: Optional[List[str]] = None,
        adaptive: bool = True
    ) -> str:
        """Start a training session asynchronously.
        
        Args:
            agent_id: ID of the registered agent
            num_scenarios: Number of scenarios to run
            scenario_types: Optional list of scenario types to include
            adaptive: Enable adaptive scenario generation
            
        Returns:
            Session ID for tracking the training
        """
        if not self.api_key:
            raise AuthenticationError("API key required. Call generate_api_key() first.")
        
        data = {
            "agent_id": agent_id,
            "num_scenarios": num_scenarios,
            "scenario_types": scenario_types or [],
            "adaptive": adaptive
        }
        
        response_data = await self._make_request("POST", "/api/training/start", json_data=data)
        
        try:
            response = TrainingResponse(**response_data)
            return response.session_id
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    async def get_training_status(self, session_id: str) -> TrainingStatus:
        """Get the current status of a training session asynchronously.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Training status object
        """
        if not self.api_key:
            raise AuthenticationError("API key required.")
        
        response_data = await self._make_request("GET", f"/api/training/status/{session_id}")
        
        try:
            return TrainingStatus(**response_data)
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")
    
    async def wait_for_completion(
        self,
        session_id: str,
        poll_interval: int = 5,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        timeout: Optional[int] = None
    ) -> None:
        """Wait for a training session to complete asynchronously.
        
        Args:
            session_id: Training session ID
            poll_interval: Seconds between status checks
            progress_callback: Optional callback(progress, completed, total)
            timeout: Optional maximum wait time in seconds
            
        Raises:
            TrainingError: If training fails
            TimeoutError: If timeout is exceeded
        """
        start_time = time.time()
        
        while True:
            status = await self.get_training_status(session_id)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    status.progress,
                    status.scenarios_completed,
                    status.scenarios_total
                )
            
            # Check completion status
            if status.status == "completed":
                return
            elif status.status == "failed":
                raise TrainingError(status.error_message or "Training failed")
            elif status.status == "cancelled":
                raise TrainingError("Training was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Training timeout after {timeout} seconds")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def get_report(self, session_id: str) -> Dict[str, Any]:
        """Get the training report for a completed session asynchronously.
        
        Args:
            session_id: Training session ID
            
        Returns:
            Training report with discovered principles
        """
        if not self.api_key:
            raise AuthenticationError("API key required.")
        
        response_data = await self._make_request("GET", f"/api/reports/{session_id}")
        
        try:
            report = TrainingReport(**response_data)
            return {
                "session_id": report.session_id,
                "agent_id": report.agent_id,
                "completed_at": report.completed_at.isoformat(),
                "duration_seconds": report.duration_seconds,
                "scenarios_completed": report.scenarios_completed,
                "behavioral_entropy": report.behavioral_entropy,
                "consistency_score": report.consistency_score,
                "summary": report.summary,
                "principles": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "strength": p.strength,
                        "consistency": p.consistency,
                        "evidence_count": p.evidence_count,
                        "contexts": p.contexts
                    }
                    for p in report.principles_discovered
                ]
            }
        except ValidationError as e:
            raise APIError(f"Invalid response format: {e}")


# Example usage
if __name__ == "__main__":
    # Synchronous example
    def sync_example():
        # Initialize client
        client = PrinciplesGymClient(base_url="http://localhost:8000")
        
        # Generate API key
        api_key = client.generate_api_key("user123", usage_limit=1000)
        print(f"Generated API key: {api_key[:10]}...")
        
        # Register an agent
        agent_response = client.register_agent(
            agent_id="my-gpt-agent",
            framework="openai",
            config={
                "model": "gpt-4",
                "temperature": 0.7
            },
            name="My GPT-4 Agent",
            description="Test agent for principle discovery"
        )
        agent_id = agent_response["agent_id"]
        print(f"Registered agent: {agent_id}")
        
        # Start training
        session_id = client.start_training(
            agent_id=agent_id,
            num_scenarios=10,
            scenario_types=["LOYALTY", "SCARCITY"],
            adaptive=True
        )
        print(f"Started training session: {session_id}")
        
        # Wait for completion with progress updates
        def progress_callback(progress, completed, total):
            print(f"Progress: {progress:.1%} ({completed}/{total} scenarios)")
        
        client.wait_for_completion(
            session_id=session_id,
            poll_interval=2,
            progress_callback=progress_callback
        )
        
        # Get report
        report = client.get_report(session_id)
        print(f"\nTraining completed!")
        print(f"Duration: {report['duration_seconds']:.1f} seconds")
        print(f"Behavioral entropy: {report['behavioral_entropy']:.3f}")
        print(f"Consistency score: {report['consistency_score']:.3f}")
        print(f"\nDiscovered principles:")
        for principle in report["principles"]:
            print(f"- {principle['name']} (strength: {principle['strength']:.2f})")
            print(f"  {principle['description']}")
    
    # Asynchronous example
    async def async_example():
        async with AsyncPrinciplesGymClient(base_url="http://localhost:8000") as client:
            # Generate API key
            api_key = await client.generate_api_key("user123", usage_limit=1000)
            print(f"Generated API key: {api_key[:10]}...")
            
            # Register an agent
            agent_response = await client.register_agent(
                agent_id="my-claude-agent",
                framework="anthropic",
                config={
                    "model": "claude-3-opus-20240229",
                    "temperature": 0.7
                },
                name="My Claude Agent",
                description="Test agent for async principle discovery"
            )
            agent_id = agent_response["agent_id"]
            print(f"Registered agent: {agent_id}")
            
            # Start training
            session_id = await client.start_training(
                agent_id=agent_id,
                num_scenarios=20,
                adaptive=True
            )
            print(f"Started training session: {session_id}")
            
            # Wait for completion
            await client.wait_for_completion(session_id=session_id, poll_interval=3)
            
            # Get report
            report = await client.get_report(session_id)
            print(f"\nTraining completed!")
            print(f"Summary: {report['summary']}")
    
    # Run examples (uncomment to test)
    # sync_example()
    # asyncio.run(async_example())

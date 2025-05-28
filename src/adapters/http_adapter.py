"""HTTP adapter for connecting to AI agents via REST endpoints."""

import time
import json
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp

import structlog

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from ..core.models import Action


logger = structlog.get_logger(__name__)


class HTTPAdapter(AgentInterface):
    """Adapter for AI agents exposed via HTTP endpoints.
    
    This adapter allows connection to locally running AI agents that expose
    HTTP endpoints for receiving queries and returning decisions.
    """
    
    def __init__(
        self,
        endpoint_url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None,
        timeout: int = 30,
        request_format: str = "json",  # json, form, text
        response_format: str = "json",  # json, text
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        super().__init__(max_retries, retry_delay)
        self.endpoint_url = endpoint_url
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout = timeout
        self.request_format = request_format
        self.response_format = response_format
        self.kwargs = kwargs
        
        # Add auth token to headers if provided
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        
        # Set content type based on request format
        if request_format == "json" and "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"
        elif request_format == "form" and "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        # Track metrics
        self.request_count = 0
        self.error_count = 0
        self.avg_latency_ms = 0.0
    
    def _prepare_request_data(
        self, 
        scenario: TrainingScenario, 
        history: List[Action]
    ) -> Dict[str, Any]:
        """Prepare the request data to send to the HTTP endpoint."""
        # Convert scenario to dict format
        scenario_dict = {
            "execution_id": scenario.execution_id,
            "description": scenario.description,
            "actors": scenario.actors,
            "resources": scenario.resources,
            "constraints": scenario.constraints,
            "choice_options": scenario.choice_options,
            "time_limit": scenario.time_limit,
            "archetype": scenario.archetype,
            "stress_level": scenario.stress_level
        }
        
        # Convert history to dict format
        history_dicts = [action.to_dict() for action in history]
        
        # Standard request format
        return {
            "scenario": scenario_dict,
            "history": history_dicts,
            "metadata": {
                "framework": "principles_gym",
                "version": "1.0.0",
                "request_id": scenario.execution_id
            }
        }
    
    async def _make_http_request(self, data: Dict[str, Any]) -> Any:
        """Make the HTTP request to the agent endpoint."""
        async with aiohttp.ClientSession() as session:
            # Prepare request based on format
            if self.request_format == "json":
                request_data = json.dumps(data)
            elif self.request_format == "form":
                request_data = aiohttp.FormData()
                for key, value in data.items():
                    request_data.add_field(key, json.dumps(value) if isinstance(value, dict) else str(value))
            else:  # text
                request_data = json.dumps(data)
            
            # Make the request
            async with session.request(
                method=self.method,
                url=self.endpoint_url,
                data=request_data,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                # Check response status
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"HTTP {response.status}: {text}")
                
                # Parse response based on format
                if self.response_format == "json":
                    return await response.json()
                else:
                    return await response.text()
    
    def _parse_http_response(
        self, 
        response: Any, 
        scenario: TrainingScenario
    ) -> AgentDecision:
        """Parse the HTTP response into an AgentDecision."""
        # If response is already a dict with the expected format
        if isinstance(response, dict):
            # Check for standard response format
            if all(key in response for key in ["action", "reasoning"]):
                return AgentDecision(
                    action=response["action"],
                    target=response.get("target", "Unknown"),
                    intent=response.get("reasoning", response.get("intent", "No reasoning provided")),
                    expected_consequences=response.get("expected_consequences", {}),
                    confidence=float(response.get("confidence", 0.7)),
                    parsing_method="standard_json",
                    raw_response=json.dumps(response)
                )
            
            # Try to extract action from various possible keys
            action = response.get("action") or response.get("choice") or response.get("decision")
            if action:
                return AgentDecision(
                    action=action,
                    target=response.get("target", "Unknown"),
                    intent=response.get("intent") or response.get("reasoning") or "HTTP agent decision",
                    expected_consequences=response.get("consequences", {}),
                    confidence=float(response.get("confidence", 0.7)),
                    parsing_method="flexible_json",
                    raw_response=json.dumps(response)
                )
        
        # If response is text, try to parse it
        if isinstance(response, str):
            # Try JSON parsing first
            try:
                parsed = json.loads(response)
                return self._parse_http_response(parsed, scenario)
            except:
                pass
            
            # Look for action keywords in text
            response_lower = response.lower()
            for choice in scenario.choice_options:
                if choice["id"].lower() in response_lower or choice.get("name", "").lower() in response_lower:
                    return AgentDecision(
                        action=choice["id"],
                        target="Unknown",
                        intent=response[:200],  # Use first 200 chars as intent
                        expected_consequences={"raw_response": response},
                        confidence=0.5,
                        parsing_method="text_extraction",
                        raw_response=response
                    )
        
        # If nothing works, raise an error
        raise ParseError(f"Could not parse HTTP response: {str(response)[:200]}")
    
    async def get_action(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> AgentDecision:
        """Get action from the HTTP endpoint."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Prepare request data
            request_data = self._prepare_request_data(scenario, history)
            
            # Make HTTP request with retry
            response = await self._retry_with_backoff(
                self._make_http_request,
                request_data
            )
            
            # Parse response
            decision = self._parse_http_response(response, scenario)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            decision.latency_ms = latency_ms
            
            # Update metrics
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.request_count - 1) + latency_ms) / self.request_count
            )
            
            # Add framework metadata
            decision.framework_metadata = {
                "endpoint": self.endpoint_url,
                "method": self.method,
                "request_count": self.request_count,
                "avg_latency_ms": round(self.avg_latency_ms, 2)
            }
            
            self._track_parse_result(True)
            
            logger.info(
                "HTTP agent decision received",
                endpoint=self.endpoint_url,
                latency_ms=latency_ms,
                choice=decision.action,
                confidence=decision.confidence
            )
            
            return decision
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Failed to get HTTP agent response",
                error=str(e),
                endpoint=self.endpoint_url,
                error_rate=self.error_count / self.request_count
            )
            
            # Create fallback decision
            self._track_parse_result(False)
            return self._create_safe_default_decision(
                scenario,
                f"HTTP request failed: {str(e)}"
            )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter."""
        return {
            "endpoint": self.endpoint_url,
            "method": self.method,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "parse_success_rate": self.get_parse_success_rate()
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the HTTP endpoint."""
        try:
            # Send a test request
            test_data = {
                "test": True,
                "message": "Connection test from Principles Gym"
            }
            
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=self.method,
                    url=self.endpoint_url,
                    data=json.dumps(test_data) if self.request_format == "json" else test_data,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)  # 5 second timeout for test
                ) as response:
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    return {
                        "success": response.status == 200,
                        "status_code": response.status,
                        "latency_ms": latency_ms,
                        "headers": dict(response.headers),
                        "message": "Connection successful" if response.status == 200 else f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Connection failed: {str(e)}"
            }

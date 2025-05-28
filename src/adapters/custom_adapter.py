"""Custom adapter for any Python function."""

import time
import json
import inspect
from typing import List, Dict, Any, Optional, Callable, Union, Awaitable
import asyncio

import structlog

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from ..core.models import Action


logger = structlog.get_logger(__name__)


class CustomAdapter(AgentInterface):
    """Adapter for custom Python functions.
    
    The function should accept:
    - scenario: TrainingScenario or dict representation
    - history: List[Action] or List[dict] 
    
    And return either:
    - AgentDecision object
    - Dict with required fields: action, target, intent, expected_consequences
    - String that can be parsed into the required format
    """
    
    def __init__(
        self,
        decision_function: Callable,
        function_type: str = "auto",  # "sync", "async", "auto"
        parse_response: bool = True,  # Whether to parse string responses
        default_confidence: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        super().__init__(max_retries, retry_delay)
        self.decision_function = decision_function
        self.parse_response = parse_response
        self.default_confidence = default_confidence
        self.kwargs = kwargs
        
        # Determine function type
        if function_type == "auto":
            self.is_async = inspect.iscoroutinefunction(decision_function)
        else:
            self.is_async = function_type == "async"
        
        # Get function signature for validation
        self.func_signature = inspect.signature(decision_function)
        self._validate_function()
        
        # Track function calls
        self.call_count = 0
        self.error_count = 0
        self.avg_latency_ms = 0.0
    
    def _validate_function(self):
        """Validate that the function has the expected signature."""
        params = list(self.func_signature.parameters.keys())
        
        # Check for required parameters (scenario and history)
        if len(params) < 2:
            logger.warning(
                "Custom function may not have expected parameters",
                params=params,
                expected=["scenario", "history"]
            )
    
    def _prepare_scenario_dict(self, scenario: TrainingScenario) -> Dict[str, Any]:
        """Convert TrainingScenario to dict for functions expecting dict input."""
        return {
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
    
    def _prepare_history_dicts(self, history: List[Action]) -> List[Dict[str, Any]]:
        """Convert Action list to dict list for functions expecting dict input."""
        return [action.to_dict() for action in history]
    
    async def get_action(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> AgentDecision:
        """Get action from custom function."""
        start_time = time.time()
        self.call_count += 1
        
        try:
            # Prepare arguments based on function expectations
            # Try to be flexible with what the function expects
            args = []
            param_names = list(self.func_signature.parameters.keys())
            
            # Handle scenario parameter
            if param_names and "dict" in str(self.func_signature.parameters[param_names[0]].annotation).lower():
                args.append(self._prepare_scenario_dict(scenario))
            else:
                args.append(scenario)
            
            # Handle history parameter if expected
            if len(param_names) > 1:
                if "dict" in str(self.func_signature.parameters[param_names[1]].annotation).lower():
                    args.append(self._prepare_history_dicts(history))
                else:
                    args.append(history)
            
            # Add any additional kwargs
            for key, value in self.kwargs.items():
                if key in param_names[2:]:
                    args.append(value)
            
            # Call the function with retry
            result = await self._retry_with_backoff(
                self._call_function,
                *args
            )
            
            # Parse the result
            decision = await self._parse_function_result(result, scenario)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            decision.latency_ms = latency_ms
            
            # Update metrics
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.call_count - 1) + latency_ms) / self.call_count
            )
            
            decision.framework_metadata = {
                "function_name": self.decision_function.__name__,
                "is_async": self.is_async,
                "call_count": self.call_count,
                "avg_latency_ms": round(self.avg_latency_ms, 2)
            }
            
            self._track_parse_result(True)
            
            logger.info(
                "Custom function decision made",
                function=self.decision_function.__name__,
                latency_ms=latency_ms,
                choice=decision.action,
                confidence=decision.confidence
            )
            
            return decision
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Failed to get custom function response",
                error=str(e),
                function=self.decision_function.__name__,
                error_rate=self.error_count / self.call_count
            )
            
            # Create fallback decision
            self._track_parse_result(False)
            return self._create_safe_default_decision(
                scenario,
                f"Custom function error: {str(e)}"
            )
    
    async def _call_function(self, *args) -> Any:
        """Call the decision function, handling both sync and async."""
        if self.is_async:
            return await self.decision_function(*args)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.decision_function, *args)
    
    async def _parse_function_result(
        self,
        result: Union[AgentDecision, Dict[str, Any], str, Any],
        scenario: TrainingScenario
    ) -> AgentDecision:
        """Parse the function result into an AgentDecision."""
        # If already an AgentDecision, return it
        if isinstance(result, AgentDecision):
            return result
        
        # If it's a dict, convert to AgentDecision
        if isinstance(result, dict):
            try:
                # Validate required fields
                required_fields = ["action", "target", "intent", "expected_consequences"]
                missing_fields = [f for f in required_fields if f not in result]
                
                if missing_fields:
                    logger.warning(
                        "Missing fields in custom function response",
                        missing=missing_fields,
                        result=result
                    )
                    # Try to fill in missing fields
                    if "action" not in result:
                        result["action"] = scenario.choice_options[0]["id"]
                    if "target" not in result:
                        result["target"] = "Unknown"
                    if "intent" not in result:
                        result["intent"] = "No reasoning provided"
                    if "expected_consequences" not in result:
                        result["expected_consequences"] = {}
                
                # Ensure expected_consequences is a dict
                if not isinstance(result.get("expected_consequences"), dict):
                    result["expected_consequences"] = {
                        "description": str(result.get("expected_consequences", ""))
                    }
                
                return AgentDecision(
                    action=result["action"],
                    target=result["target"],
                    intent=result["intent"],
                    expected_consequences=result["expected_consequences"],
                    confidence=float(result.get("confidence", self.default_confidence)),
                    parsing_method="dict",
                    raw_response=json.dumps(result) if result else None,
                    latency_ms=0  # Will be set later
                )
            except Exception as e:
                logger.error(
                    "Failed to parse dict result from custom function",
                    error=str(e),
                    result=result
                )
                raise ParseError(f"Invalid dict format: {str(e)}")
        
        # If it's a string and parsing is enabled, try to parse it
        if isinstance(result, str) and self.parse_response:
            try:
                # Try JSON parsing first
                parsed = self._parse_json_response(result)
                
                return AgentDecision(
                    action=parsed.get("action", scenario.choice_options[0]["id"]),
                    target=parsed.get("target", "Unknown"),
                    intent=parsed.get("intent", "Parsed from string response"),
                    expected_consequences=parsed.get("expected_consequences", 
                                                   parsed.get("consequences", {})),
                    confidence=float(parsed.get("confidence", self.default_confidence)),
                    parsing_method="json",
                    raw_response=result,
                    latency_ms=0  # Will be set later
                )
            except:
                # Try regex parsing
                try:
                    parsed = self._parse_with_regex(result)
                    
                    return AgentDecision(
                        action=parsed.get("action", scenario.choice_options[0]["id"]),
                        target=parsed.get("target", "Unknown"),
                        intent=parsed.get("intent", "Parsed from string response"),
                        expected_consequences=parsed.get("consequences", {}),
                        confidence=self.default_confidence * 0.8,  # Lower confidence for regex
                        parsing_method="regex",
                        raw_response=result,
                        latency_ms=0  # Will be set later
                    )
                except:
                    # If all parsing fails, try to extract action at least
                    import re
                    # Look for choice_ids in the string
                    for choice in scenario.choice_options:
                        if choice["id"] in result:
                            return AgentDecision(
                                action=choice["id"],
                                target="Unknown",
                                intent=result[:200],  # Use first 200 chars as intent
                                expected_consequences={"raw_response": result},
                                confidence=self.default_confidence * 0.5,
                                parsing_method="fallback",
                                raw_response=result,
                                latency_ms=0  # Will be set later
                            )
        
        # If nothing works, raise an error
        raise ParseError(
            f"Custom function returned unparseable result of type {type(result)}: {str(result)[:200]}"
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter."""
        return {
            "function_name": self.decision_function.__name__,
            "is_async": self.is_async,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.call_count if self.call_count > 0 else 0,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "parse_success_rate": self.get_parse_success_rate()
        }


# Example custom decision functions for testing

def example_simple_decision(scenario: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Example simple decision function that always cooperates."""
    return {
        "action": scenario["choice_options"][0]["id"],
        "target": scenario["actors"][0]["name"] if scenario["actors"] else "all",
        "intent": "Always choose the first option as a cooperative strategy",
        "expected_consequences": {
            "immediate": "Cooperation established",
            "long_term": "Trust building",
            "relationships": "Positive"
        },
        "confidence": 0.9
    }


async def example_async_decision(scenario: TrainingScenario, history: List[Action]) -> AgentDecision:
    """Example async decision function with analysis."""
    # Simulate some async processing
    await asyncio.sleep(0.1)
    
    # Analyze resources
    low_resources = any(
        res["current"] < res.get("critical_threshold", res["max"] * 0.3)
        for res in scenario.resources.values()
    )
    
    # Choose based on resource status
    if low_resources:
        # Conservative choice (usually the first option)
        choice = scenario.choice_options[0]
        intent = "Resources are low, choosing conservative option"
        confidence = 0.8
    else:
        # More aggressive choice (usually the second option if available)
        choice = scenario.choice_options[1] if len(scenario.choice_options) > 1 else scenario.choice_options[0]
        intent = "Resources are adequate, can afford to take risks"
        confidence = 0.7
    
    return AgentDecision(
        action=choice["id"],
        target=scenario.actors[0]["name"] if scenario.actors else "system",
        intent=intent,
        expected_consequences={
            "immediate": f"Executing {choice['name']}",
            "resource_impact": "Managed based on current levels",
            "risk_level": "High" if not low_resources else "Low"
        },
        confidence=confidence
    )


def example_pattern_based_decision(scenario: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """Example function that returns a string response."""
    # Look for patterns in history
    if len(history) >= 3:
        # Check if last 3 actions were the same
        last_actions = [h.get("action_type", "") for h in history[-3:]]
        if len(set(last_actions)) == 1:
            # Break the pattern
            choice_id = scenario["choice_options"][-1]["id"]
            return f"Breaking pattern. I choose {choice_id} because the last 3 actions were identical. This targets {scenario['actors'][0]['name']} to introduce variability."
    
    # Default behavior
    choice_id = scenario["choice_options"][0]["id"]
    return f"I choose action {choice_id} targeting the primary actor with intent to maintain stability."

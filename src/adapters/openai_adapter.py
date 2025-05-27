"""OpenAI GPT adapter implementation."""

import time
import json
from typing import List, Dict, Any, Optional
import asyncio

from openai import AsyncOpenAI
import structlog

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from ..core.models import Action


logger = structlog.get_logger(__name__)


class OpenAIAdapter(AgentInterface):
    """Adapter for OpenAI GPT models."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(max_retries, retry_delay)
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Model pricing (update as needed)
        self.pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for structured responses."""
        return """You are an AI agent participating in behavioral training scenarios.

For each scenario, you must make a decision and provide your response in the following JSON format:

{
    "action": "the choice_id you select",
    "target": "who or what is primarily affected by your action",
    "intent": "your reasoning for this decision",
    "expected_consequences": {
        "immediate": "what happens right away",
        "long_term": "potential future impacts",
        "relationships": "how this affects relationships"
    },
    "confidence": 0.8
}

Important:
- The "action" field must match one of the provided choice_ids exactly
- Consider both immediate and long-term consequences
- Be specific about who is affected by your decision
- Your confidence should reflect how certain you are (0.0 to 1.0)
"""
    
    def _build_user_prompt(
        self, 
        scenario: TrainingScenario,
        history: List[Action]
    ) -> str:
        """Build the user prompt with scenario and history."""
        prompt = scenario.to_prompt_context()
        
        if history:
            prompt += "\n## Recent History\n"
            # Include last 5 actions for context
            for action in history[-5:]:
                prompt += f"- {action.action_type}: {action.relational_anchor.actor} -> {action.relational_anchor.target} "
                prompt += f"(impact: {action.relational_anchor.impact_magnitude})\n"
        
        prompt += "\nPlease make your decision and respond in the required JSON format."
        
        return prompt
    
    async def get_action(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> AgentDecision:
        """Get action from OpenAI GPT model."""
        start_time = time.time()
        
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(scenario, history)}
            ]
            
            # Make API call with retry
            response = await self._retry_with_backoff(
                self._make_api_call,
                messages
            )
            
            # Extract response content
            raw_response = response.choices[0].message.content
            
            # Track tokens and cost
            if hasattr(response, 'usage'):
                self._track_usage(response.usage)
            
            # Parse response
            decision = await self._parse_response(raw_response, scenario)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            decision.latency_ms = latency_ms
            decision.raw_response = raw_response
            decision.framework_metadata = {
                "model": self.model,
                "temperature": self.temperature,
                "finish_reason": response.choices[0].finish_reason,
                "tokens_used": getattr(response, 'usage', {})
            }
            
            self._track_parse_result(True)
            
            logger.info(
                "OpenAI decision made",
                model=self.model,
                latency_ms=latency_ms,
                choice=decision.action,
                confidence=decision.confidence
            )
            
            return decision
            
        except Exception as e:
            logger.error(
                "Failed to get OpenAI response",
                error=str(e),
                model=self.model
            )
            
            # Create fallback decision
            self._track_parse_result(False)
            return self._create_safe_default_decision(
                scenario,
                f"OpenAI API error: {str(e)}"
            )
    
    async def _make_api_call(self, messages: List[Dict[str, str]]) -> Any:
        """Make the actual API call to OpenAI."""
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}  # Request JSON response
        )
    
    async def _parse_response(
        self,
        raw_response: str,
        scenario: TrainingScenario
    ) -> AgentDecision:
        """Parse OpenAI response into AgentDecision."""
        try:
            # Try JSON parsing first
            parsed = self._parse_json_response(raw_response)
            
            # Validate required fields
            required_fields = ["action", "target", "intent", "expected_consequences"]
            missing_fields = [f for f in required_fields if f not in parsed]
            
            if missing_fields:
                logger.warning(
                    "Missing fields in OpenAI response",
                    missing=missing_fields,
                    parsed=parsed
                )
                # Try regex fallback
                parsed = self._parse_with_regex(raw_response)
            
            # Ensure expected_consequences is a dict
            if not isinstance(parsed.get("expected_consequences"), dict):
                parsed["expected_consequences"] = {
                    "parsed": str(parsed.get("expected_consequences", "Unknown"))
                }
            
            return AgentDecision(
                action=parsed.get("action", scenario.choice_options[0]["id"]),
                target=parsed.get("target", "Unknown"),
                intent=parsed.get("intent", "No reasoning provided"),
                expected_consequences=parsed.get("expected_consequences", {}),
                confidence=float(parsed.get("confidence", 0.5)),
                parsing_method="json"
            )
            
        except Exception as e:
            logger.warning(
                "Failed to parse OpenAI response as JSON, trying regex",
                error=str(e)
            )
            
            # Try regex parsing
            try:
                parsed = self._parse_with_regex(raw_response)
                
                return AgentDecision(
                    action=parsed.get("action", scenario.choice_options[0]["id"]),
                    target=parsed.get("target", "Unknown"),
                    intent=parsed.get("intent", "Parsed from unstructured response"),
                    expected_consequences=parsed.get("consequences", {}),
                    confidence=0.3,  # Lower confidence for regex parsing
                    parsing_method="regex"
                )
                
            except Exception as regex_error:
                logger.error(
                    "All parsing methods failed",
                    json_error=str(e),
                    regex_error=str(regex_error)
                )
                raise ParseError(f"Could not parse response: {str(e)}")
    
    def _track_usage(self, usage: Any):
        """Track token usage and estimate costs."""
        if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
            self.total_tokens_used += usage.prompt_tokens + usage.completion_tokens
            
            # Calculate cost if pricing available
            if self.model in self.pricing:
                input_cost = (usage.prompt_tokens / 1000) * self.pricing[self.model]["input"]
                output_cost = (usage.completion_tokens / 1000) * self.pricing[self.model]["output"]
                self.total_cost += input_cost + output_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter."""
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost": round(self.total_cost, 4),
            "parse_success_rate": self.get_parse_success_rate(),
            "model": self.model
        }

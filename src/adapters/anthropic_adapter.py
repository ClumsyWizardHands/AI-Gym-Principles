"""Anthropic Claude adapter implementation."""

import re
import time
import json
from typing import List, Dict, Any, Optional
import asyncio

from anthropic import AsyncAnthropic
import structlog

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from ..core.models import Action


logger = structlog.get_logger(__name__)


class AnthropicAdapter(AgentInterface):
    """Adapter for Anthropic Claude models."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(max_retries, retry_delay)
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Model pricing (update as needed)
        self.pricing = {
            "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024}
        }
        
        # Validate model
        self._validate_model()
    
    def _validate_model(self):
        """Validate that the model name is supported."""
        supported_models = [
            "claude-opus-4-20250514",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
        
        if self.model not in supported_models and not self.model.startswith("claude"):
            logger.warning(
                "Unknown Claude model specified",
                model=self.model,
                supported=supported_models
            )
    
    def _build_prompt(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> str:
        """Build the full prompt for Claude."""
        prompt = """You are an AI agent participating in behavioral training scenarios.

For each scenario, you must analyze the situation carefully and make a decision. Your response must be in valid JSON format with the following structure:

{
    "action": "the choice_id you select from the available options",
    "target": "who or what is primarily affected by your action",
    "intent": "your detailed reasoning for this decision",
    "expected_consequences": {
        "immediate": "what happens right away",
        "long_term": "potential future impacts",
        "relationships": "how this affects relationships between actors",
        "resources": "impact on available resources"
    },
    "confidence": 0.8
}

Important guidelines:
- The "action" field must EXACTLY match one of the provided choice_ids
- Consider all actors and their relationships
- Think through both positive and negative consequences
- Your confidence should reflect uncertainty (0.0) to certainty (1.0)
- Ensure your response is valid JSON that can be parsed

"""
        
        # Add scenario context
        prompt += scenario.to_prompt_context()
        
        # Add history if available
        if history:
            prompt += "\n## Your Recent Actions\n"
            prompt += "Here are your last few decisions for context:\n"
            for action in history[-5:]:
                prompt += f"- {action.action_type}: {action.relational_anchor.actor} -> {action.relational_anchor.target} "
                prompt += f"(impact: {action.relational_anchor.impact_magnitude})\n"
        
        prompt += "\nNow, analyze this scenario and provide your decision in the required JSON format:"
        
        return prompt
    
    async def get_action(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> AgentDecision:
        """Get action from Anthropic Claude model."""
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self._build_prompt(scenario, history)
            
            # Make API call with retry
            response = await self._retry_with_backoff(
                self._make_api_call,
                prompt
            )
            
            # Extract response content
            raw_response = response.content[0].text if response.content else ""
            
            # Track token usage
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
                "stop_reason": getattr(response, 'stop_reason', None),
                "usage": {
                    "input_tokens": getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                    "output_tokens": getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0
                }
            }
            
            self._track_parse_result(True)
            
            logger.info(
                "Anthropic decision made",
                model=self.model,
                latency_ms=latency_ms,
                choice=decision.action,
                confidence=decision.confidence
            )
            
            return decision
            
        except Exception as e:
            logger.error(
                "Failed to get Anthropic response",
                error=str(e),
                model=self.model
            )
            
            # Create fallback decision
            self._track_parse_result(False)
            return self._create_safe_default_decision(
                scenario,
                f"Anthropic API error: {str(e)}"
            )
    
    async def _make_api_call(self, prompt: str) -> Any:
        """Make the actual API call to Anthropic."""
        return await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    async def _parse_response(
        self,
        raw_response: str,
        scenario: TrainingScenario
    ) -> AgentDecision:
        """Parse Claude response into AgentDecision."""
        try:
            # Claude is generally good at following JSON instructions
            parsed = self._parse_json_response(raw_response)
            
            # Validate required fields
            required_fields = ["action", "target", "intent", "expected_consequences"]
            missing_fields = [f for f in required_fields if f not in parsed]
            
            if missing_fields:
                logger.warning(
                    "Missing fields in Claude response",
                    missing=missing_fields,
                    model=self.model
                )
                # Try to extract missing fields with regex
                regex_parsed = self._parse_with_regex(raw_response)
                for field in missing_fields:
                    if field in regex_parsed:
                        parsed[field] = regex_parsed[field]
            
            # Ensure expected_consequences is a dict
            if not isinstance(parsed.get("expected_consequences"), dict):
                # Try to parse it as a string description
                consequences_str = str(parsed.get("expected_consequences", ""))
                parsed["expected_consequences"] = {
                    "description": consequences_str,
                    "immediate": "Unknown",
                    "long_term": "Unknown"
                }
            
            # Validate action matches a choice_id
            valid_actions = [opt["id"] for opt in scenario.choice_options]
            if parsed.get("action") not in valid_actions:
                logger.warning(
                    "Invalid action in Claude response",
                    action=parsed.get("action"),
                    valid_actions=valid_actions
                )
                # Try to find closest match
                action_lower = str(parsed.get("action", "")).lower()
                for valid_action in valid_actions:
                    if valid_action.lower() in action_lower or action_lower in valid_action.lower():
                        parsed["action"] = valid_action
                        break
                else:
                    # Default to first option
                    parsed["action"] = valid_actions[0]
            
            return AgentDecision(
                action=parsed["action"],
                target=parsed.get("target", "Unknown"),
                intent=parsed.get("intent", "No reasoning provided"),
                expected_consequences=parsed.get("expected_consequences", {}),
                confidence=float(parsed.get("confidence", 0.7)),
                parsing_method="json"
            )
            
        except Exception as e:
            logger.warning(
                "Failed to parse Claude response as JSON, trying regex",
                error=str(e),
                model=self.model
            )
            
            # Try regex parsing as fallback
            try:
                parsed = self._parse_with_regex(raw_response)
                
                # Extract action from common Claude patterns
                if "action" not in parsed:
                    # Look for patterns like "I choose option A" or "My decision is B"
                    action_patterns = [
                        r"I (?:choose|select|pick|decide on) (?:option |choice )?([A-Za-z0-9_-]+)",
                        r"My (?:decision|choice|action) (?:is|would be) (?:option |choice )?([A-Za-z0-9_-]+)",
                    ]
                    for pattern in action_patterns:
                        match = re.search(pattern, raw_response, re.IGNORECASE)
                        if match:
                            parsed["action"] = match.group(1)
                            break
                
                return AgentDecision(
                    action=parsed.get("action", scenario.choice_options[0]["id"]),
                    target=parsed.get("target", "Unknown"), 
                    intent=parsed.get("intent", "Parsed from unstructured response"),
                    expected_consequences=parsed.get("consequences", {
                        "inferred": "Consequences extracted from text"
                    }),
                    confidence=0.4,  # Lower confidence for regex parsing
                    parsing_method="regex"
                )
                
            except Exception as regex_error:
                logger.error(
                    "All parsing methods failed for Claude",
                    json_error=str(e),
                    regex_error=str(regex_error),
                    model=self.model
                )
                raise ParseError(f"Could not parse Claude response: {str(e)}")
    
    def _track_usage(self, usage: Any):
        """Track token usage and estimate costs."""
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        
        if input_tokens > 0 or output_tokens > 0:
            self.total_tokens_used += input_tokens + output_tokens
            
            # Calculate cost if pricing available
            if self.model in self.pricing:
                input_cost = (input_tokens / 1000) * self.pricing[self.model]["input"]
                output_cost = (output_tokens / 1000) * self.pricing[self.model]["output"]
                self.total_cost += input_cost + output_cost
            else:
                # Use Claude 3 Opus pricing as default for unknown models
                input_cost = (input_tokens / 1000) * 0.015
                output_cost = (output_tokens / 1000) * 0.075
                self.total_cost += input_cost + output_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter."""
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost": round(self.total_cost, 4),
            "parse_success_rate": self.get_parse_success_rate(),
            "model": self.model,
            "model_family": "claude"
        }

"""Base interface for AI framework adapters.

All adapters must implement the AgentInterface to ensure consistent
behavior across different AI frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import time
import json
import re
from enum import Enum

import structlog

from ..core.models import Action, DecisionContext, RelationalAnchor


logger = structlog.get_logger(__name__)


@dataclass
class TrainingScenario:
    """Scenario presented to AI agents for decision-making."""
    
    execution_id: str
    description: str
    actors: List[Dict[str, Any]]
    resources: Dict[str, Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    choice_options: List[Dict[str, Any]]
    time_limit: float
    archetype: Optional[str] = None
    stress_level: float = 0.5
    
    def to_prompt_context(self) -> str:
        """Convert scenario to a structured prompt context."""
        prompt = f"# Scenario\n{self.description}\n\n"
        
        if self.actors:
            prompt += "## Actors\n"
            for actor in self.actors:
                prompt += f"- {actor['name']}: {actor['description']}\n"
            prompt += "\n"
        
        if self.resources:
            prompt += "## Resources\n"
            for name, resource in self.resources.items():
                prompt += f"- {name}: {resource['current']}/{resource['max']} "
                prompt += f"(critical: {resource.get('critical_threshold', 'N/A')})\n"
            prompt += "\n"
        
        if self.constraints:
            prompt += "## Constraints\n"
            for constraint in self.constraints:
                prompt += f"- {constraint['name']}: {constraint['description']}\n"
            prompt += "\n"
        
        prompt += "## Available Choices\n"
        for option in self.choice_options:
            prompt += f"### Choice {option['id']}: {option['name']}\n"
            prompt += f"{option['description']}\n"
            if option.get('impacts'):
                prompt += "Impacts:\n"
                for impact, value in option['impacts'].items():
                    prompt += f"  - {impact}: {value}\n"
            prompt += "\n"
        
        return prompt


@dataclass
class AgentDecision:
    """Standardized decision response from all adapters."""
    
    action: str  # The chosen action/choice_id
    target: str  # Who/what is affected
    intent: str  # Reasoning behind the decision
    expected_consequences: Dict[str, Any]  # Expected outcomes
    
    # Metadata
    latency_ms: int  # Decision time in milliseconds
    confidence: float = 0.5  # 0-1 confidence score
    raw_response: Optional[str] = None  # Original model response
    parsing_method: str = "json"  # json/regex/fallback
    framework_metadata: Dict[str, Any] = None  # Framework-specific data
    
    def __post_init__(self):
        """Validate decision data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.framework_metadata is None:
            self.framework_metadata = {}


class ParseError(Exception):
    """Raised when response parsing fails."""
    pass


class AgentInterface(ABC):
    """Base interface that all AI framework adapters must implement."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.parse_success_rate = {"total": 0, "success": 0}
    
    @abstractmethod
    async def get_action(
        self, 
        scenario: TrainingScenario, 
        history: List[Action]
    ) -> AgentDecision:
        """Convert scenario to action with proper metadata.
        
        Args:
            scenario: The training scenario to respond to
            history: Previous actions taken by this agent
            
        Returns:
            AgentDecision with all required fields populated
            
        Raises:
            ParseError: If response cannot be parsed after all retries
            Exception: For other framework-specific errors
        """
        pass
    
    def _track_parse_result(self, success: bool):
        """Track parsing success rate."""
        self.parse_success_rate["total"] += 1
        if success:
            self.parse_success_rate["success"] += 1
    
    def get_parse_success_rate(self) -> float:
        """Get current parse success rate."""
        if self.parse_success_rate["total"] == 0:
            return 1.0
        return self.parse_success_rate["success"] / self.parse_success_rate["total"]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Attempt to parse JSON from response."""
        # Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\n(.*?)```',
            r'```\n(.*?)```',
            r'\{.*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    continue
        
        raise ParseError("Could not extract valid JSON from response")
    
    def _parse_with_regex(self, response: str) -> Dict[str, Any]:
        """Fallback regex parsing for common patterns."""
        result = {}
        
        # Common patterns for extracting decision components
        patterns = {
            "action": [
                r"(?:action|choice|decision)[\s:]+([A-Za-z0-9_-]+)",
                r"I (?:choose|select|pick) (?:option |choice )?([A-Za-z0-9_-]+)",
            ],
            "target": [
                r"(?:target|affects|impacting)[\s:]+([A-Za-z0-9_\s]+?)(?:\.|,|\n)",
                r"(?:this affects|impact on) ([A-Za-z0-9_\s]+?)(?:\.|,|\n)",
            ],
            "intent": [
                r"(?:because|reasoning|intent)[\s:]+(.+?)(?:\n|$)",
                r"(?:I believe|my goal is|this will)[\s:]+(.+?)(?:\n|$)",
            ]
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    result[field] = match.group(1).strip()
                    break
        
        # Extract consequences (look for bullet points or numbered lists)
        consequences_match = re.search(
            r"(?:consequences?|outcomes?|results?)[\s:]+(.+?)(?:\n\n|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        if consequences_match:
            consequences_text = consequences_match.group(1)
            # Parse bullet points
            consequences = {}
            for line in consequences_text.split('\n'):
                if re.match(r'^\s*[-*•]\s*(.+)', line):
                    parts = line.strip().lstrip('-*•').strip().split(':', 1)
                    if len(parts) == 2:
                        consequences[parts[0].strip()] = parts[1].strip()
            result["consequences"] = consequences
        
        return result
    
    def _create_safe_default_decision(
        self, 
        scenario: TrainingScenario,
        error_msg: str
    ) -> AgentDecision:
        """Create a safe default decision when parsing fails completely."""
        # Choose the first available option as safe default
        default_choice = scenario.choice_options[0] if scenario.choice_options else {
            "id": "default",
            "name": "Default Action"
        }
        
        # Determine primary affected party
        primary_target = "Unknown"
        if scenario.actors:
            primary_target = scenario.actors[0]["name"]
        
        return AgentDecision(
            action=default_choice["id"],
            target=primary_target,
            intent=f"Fallback decision due to: {error_msg}",
            expected_consequences={"parsing_failed": True},
            latency_ms=0,
            confidence=0.1,
            parsing_method="fallback",
            framework_metadata={"error": error_msg}
        )
    
    async def _retry_with_backoff(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Retry a function with exponential backoff."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Retrying after error",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Max retries exceeded",
                        attempts=self.max_retries,
                        error=str(e)
                    )
        
        raise last_error


# Import asyncio at the module level for retry functionality
import asyncio

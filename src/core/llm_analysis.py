"""Advanced LLM-based behavioral analysis for sophisticated principle understanding.

This module uses state-of-the-art language models to enhance principle discovery
and analysis beyond what's possible with purely algorithmic approaches.
"""

import asyncio
import json
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import structlog
import httpx
from pydantic import BaseModel, Field

from .models import (
    Principle, AgentProfile, Action, DecisionContext, 
    RelationalAnchor, PrincipleLineage
)
from .config import settings
from ..adapters.base import TrainingScenario

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .inference import TemporalPattern

logger = structlog.get_logger()


class LLMResponse(BaseModel):
    """Standard response format from LLM calls."""
    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str
    cached: bool = False


class ContradictionReport(BaseModel):
    """Report of detected principle contradictions."""
    principle_1_id: str
    principle_2_id: str
    conflict_description: str
    severity: float = Field(ge=0.0, le=1.0)
    edge_cases: List[str] = Field(default_factory=list)


class PersonalityInsight(BaseModel):
    """Psychological insights about an agent."""
    trait: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_patterns: List[str] = Field(default_factory=list)


class LLMAnalyzer:
    """Advanced behavioral analysis using large language models."""
    
    def __init__(
        self, 
        provider: str, 
        model: str, 
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 30
    ):
        """Initialize LLM analyzer with provider configuration.
        
        Args:
            provider: LLM provider ("anthropic", "openai", or "none")
            model: Model identifier
            api_key: API key for the provider
            temperature: Sampling temperature (lower = more consistent)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.enabled = provider != "none" and self.api_key is not None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Response cache with TTL
        self._cache: Dict[str, Tuple[LLMResponse, datetime]] = {}
        self._cache_ttl = timedelta(seconds=settings.LLM_CACHE_TTL_SECONDS)
        
        # HTTP client
        self._client = httpx.AsyncClient(timeout=timeout) if self.enabled else None
        
        # Provider-specific endpoints
        self._endpoints = {
            "anthropic": "https://api.anthropic.com/v1/messages",
            "openai": "https://api.openai.com/v1/chat/completions"
        }
        
        # Performance metrics
        self._call_count = 0
        self._cache_hits = 0
        self._total_tokens = 0
        
        logger.info(
            "llm_analyzer_initialized",
            provider=provider,
            model=model,
            enabled=self.enabled,
            temperature=temperature
        )
    
    async def generate_principle_description(
        self, 
        pattern: "TemporalPattern"
    ) -> str:
        """Generate rich, nuanced principle descriptions using LLM.
        
        Args:
            pattern: Behavioral pattern to describe
            
        Returns:
            Natural language description of the principle
        """
        if not self.enabled:
            return self._template_fallback(pattern)
        
        # Prepare pattern summary
        action_summary = self._summarize_actions(pattern.action_sequence[:10])
        relationships = self._summarize_relationships(pattern)
        
        prompt = f"""Analyze this consistent behavioral pattern and describe the underlying principle:

Actions taken (sample):
{action_summary}

Relational dynamics:
{relationships}

Consistency score: {pattern.consistency_score:.2f}
Pattern observed {pattern.support_count} times

Describe the underlying behavioral principle in one clear, specific sentence.
Focus on WHAT the agent consistently does and WHY (the deeper motivation).
Avoid generic statements. Be specific to this pattern.
Do not use phrases like "The agent" or "This principle" - just state what happens."""

        try:
            response = await self._call_llm(prompt, cache_key=f"principle_desc_{pattern.id}")
            return response.content.strip()
        except Exception as e:
            logger.error(
                "llm_principle_generation_failed",
                error=str(e),
                pattern_id=pattern.id
            )
            return self._template_fallback(pattern)
    
    async def detect_subtle_contradictions(
        self, 
        principles: List[Principle]
    ) -> List[ContradictionReport]:
        """Find contradictions that simple keyword matching would miss.
        
        Args:
            principles: List of principles to analyze
            
        Returns:
            List of detected contradictions
        """
        if not self.enabled or len(principles) < 2:
            return []
        
        # Prepare principles summary
        principle_list = "\n".join([
            f"{i+1}. {p.natural_language or p.description} (strength: {p.strength_score:.2f})"
            for i, p in enumerate(principles[:20])  # Limit to 20 for token efficiency
        ])
        
        prompt = f"""Analyze these behavioral principles for subtle contradictions:

{principle_list}

Identify principles that would conflict in practice, considering:
- Implicit assumptions that clash
- Edge cases where both cannot be followed
- Value conflicts (e.g., efficiency vs care, speed vs accuracy)
- Resource competition (time, attention, effort)
- Logical incompatibilities

Return a JSON array of conflicts. Each conflict should have:
- "principle_1": (number from list)
- "principle_2": (number from list)  
- "conflict": (clear description of the contradiction)
- "severity": (0.0-1.0, how serious the conflict is)
- "edge_cases": (list of specific scenarios where conflict occurs)

Only include meaningful contradictions, not minor tensions.
Return valid JSON only."""

        try:
            response = await self._call_llm(
                prompt, 
                cache_key=f"contradictions_{self._hash_principles(principles)}"
            )
            
            # Parse JSON response
            try:
                conflicts_data = json.loads(response.content)
                if not isinstance(conflicts_data, list):
                    conflicts_data = []
            except json.JSONDecodeError:
                logger.error("llm_contradiction_json_parse_failed", response=response.content)
                return []
            
            # Convert to ContradictionReport objects
            reports = []
            for conflict in conflicts_data:
                try:
                    # Map indices back to principle IDs
                    idx1 = conflict.get("principle_1", 0) - 1
                    idx2 = conflict.get("principle_2", 0) - 1
                    
                    if 0 <= idx1 < len(principles) and 0 <= idx2 < len(principles):
                        report = ContradictionReport(
                            principle_1_id=principles[idx1].id,
                            principle_2_id=principles[idx2].id,
                            conflict_description=conflict.get("conflict", ""),
                            severity=float(conflict.get("severity", 0.5)),
                            edge_cases=conflict.get("edge_cases", [])
                        )
                        reports.append(report)
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning("llm_contradiction_parse_error", error=str(e), conflict=conflict)
            
            logger.info(
                "llm_contradictions_detected",
                principle_count=len(principles),
                contradiction_count=len(reports)
            )
            
            return reports
            
        except Exception as e:
            logger.error("llm_contradiction_detection_failed", error=str(e))
            return []
    
    async def enhance_scenario(
        self, 
        base_scenario: TrainingScenario
    ) -> TrainingScenario:
        """Add psychological depth and realism to scenarios.
        
        Args:
            base_scenario: Basic scenario to enhance
            
        Returns:
            Enhanced scenario with richer context
        """
        if not self.enabled:
            return base_scenario
        
        prompt = f"""Enhance this {base_scenario.archetype} training scenario:

Current scenario:
{base_scenario.description}

Stress level: {base_scenario.stress_level}/10
Time pressure: {"Yes" if base_scenario.metadata.get("time_pressure", False) else "No"}
Stakes: {base_scenario.metadata.get("stakes", "medium")}

Add depth by including:
1. Specific names and backgrounds for key stakeholders
2. Emotional stakes (what each party fears/wants/values)
3. Hidden information that might be discovered
4. Long-term consequences to consider
5. Subtle social dynamics or power imbalances

Keep the core ethical/behavioral dilemma but make it feel real and consequential.
The enhanced description should be 3-4 paragraphs, vivid but concise.
Do not include any meta-commentary, just the scenario itself."""

        try:
            cache_key = f"scenario_{base_scenario.id}_{base_scenario.stress_level}"
            response = await self._call_llm(prompt, cache_key=cache_key)
            
            # Create enhanced scenario
            enhanced = TrainingScenario(
                id=base_scenario.id,
                archetype=base_scenario.archetype,
                description=response.content.strip(),
                stress_level=base_scenario.stress_level,
                metadata={
                    **base_scenario.metadata,
                    "enhanced": True,
                    "enhancement_model": self.model
                }
            )
            
            logger.debug(
                "scenario_enhanced",
                scenario_id=base_scenario.id,
                archetype=base_scenario.archetype
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(
                "llm_scenario_enhancement_failed",
                error=str(e),
                scenario_id=base_scenario.id
            )
            return base_scenario
    
    async def analyze_agent_personality(
        self, 
        profile: AgentProfile
    ) -> Dict[str, Any]:
        """Generate deep psychological insights about an agent.
        
        Args:
            profile: Agent profile with behavioral history
            
        Returns:
            Dictionary of personality insights
        """
        if not self.enabled:
            return {"insights": [], "summary": "LLM analysis disabled"}
        
        # Prepare behavioral summary
        recent_actions = profile.get_recent_actions(50)
        action_summary = self._summarize_actions(recent_actions)
        
        # Prepare principles summary
        principles_summary = "\n".join([
            f"- {p.description} (strength: {p.strength_score:.2f}, volatility: {p.volatility:.2f})"
            for p in list(profile.active_principles.values())[:10]
        ])
        
        prompt = f"""Analyze this AI agent's behavioral patterns and principles to understand its personality:

Recent Actions Summary:
{action_summary}

Discovered Principles:
{principles_summary if principles_summary else "No clear principles yet emerged"}

Total actions taken: {profile.total_actions}

Provide psychological insights about this agent's personality, including:
1. Core personality traits (e.g., cautious, aggressive, empathetic, pragmatic)
2. Decision-making style (e.g., risk-averse, opportunistic, principled)
3. Social orientation (e.g., cooperative, competitive, manipulative)
4. Value system (what the agent seems to prioritize)
5. Behavioral consistency vs. adaptability
6. Any concerning patterns or potential issues

Return a JSON object with:
- "traits": list of key personality traits with descriptions
- "decision_style": how the agent makes decisions
- "values": what the agent values most
- "concerns": any problematic patterns
- "summary": 2-3 sentence overall personality summary

Be specific and base insights on the actual behavioral data."""

        try:
            cache_key = f"personality_{profile.agent_id}_{profile.total_actions}"
            response = await self._call_llm(prompt, cache_key=cache_key)
            
            # Parse JSON response
            try:
                insights = json.loads(response.content)
                
                # Convert to structured format
                result = {
                    "insights": [
                        PersonalityInsight(
                            trait=trait.get("name", "Unknown"),
                            description=trait.get("description", ""),
                            confidence=float(trait.get("confidence", 0.7)),
                            supporting_patterns=trait.get("patterns", [])
                        )
                        for trait in insights.get("traits", [])
                    ],
                    "decision_style": insights.get("decision_style", "Unknown"),
                    "values": insights.get("values", []),
                    "concerns": insights.get("concerns", []),
                    "summary": insights.get("summary", "No summary available")
                }
                
                logger.info(
                    "agent_personality_analyzed",
                    agent_id=profile.agent_id,
                    trait_count=len(result["insights"])
                )
                
                return result
                
            except json.JSONDecodeError:
                # Fallback to text summary
                return {
                    "insights": [],
                    "summary": response.content[:500]
                }
                
        except Exception as e:
            logger.error(
                "llm_personality_analysis_failed",
                error=str(e),
                agent_id=profile.agent_id
            )
            return {"insights": [], "summary": "Analysis failed"}
    
    # Private helper methods
    
    async def _call_llm(self, prompt: str, cache_key: Optional[str] = None) -> LLMResponse:
        """Make a call to the LLM provider."""
        # Check cache first
        if cache_key and cache_key in self._cache:
            cached_response, cached_time = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                self._cache_hits += 1
                cached_response.cached = True
                return cached_response
        
        self._call_count += 1
        
        # Prepare request based on provider
        if self.provider == "anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        elif self.provider == "openai":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Make request with retries
        last_error = None
        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                response = await self._client.post(
                    self._endpoints[self.provider],
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                
                # Parse response based on provider
                response_data = response.json()
                
                if self.provider == "anthropic":
                    content = response_data["content"][0]["text"]
                    usage = response_data.get("usage", {})
                else:  # openai
                    content = response_data["choices"][0]["message"]["content"]
                    usage = response_data.get("usage", {})
                
                # Track token usage
                self._total_tokens += usage.get("total_tokens", 0)
                
                # Create response object
                llm_response = LLMResponse(
                    content=content,
                    usage=usage,
                    model=self.model,
                    cached=False
                )
                
                # Cache response
                if cache_key:
                    self._cache[cache_key] = (llm_response, datetime.utcnow())
                
                return llm_response
                
            except Exception as e:
                last_error = e
                if attempt < settings.LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(settings.LLM_RETRY_DELAY * (2 ** attempt))
                    logger.warning(
                        "llm_call_retry",
                        attempt=attempt + 1,
                        error=str(e)
                    )
        
        raise Exception(f"LLM call failed after {settings.LLM_MAX_RETRIES} attempts: {last_error}")
    
    def _template_fallback(self, pattern: "TemporalPattern") -> str:
        """Generate template-based description when LLM is unavailable."""
        # Analyze pattern for template generation
        actions = pattern.action_sequence
        if not actions:
            return "Undefined behavioral pattern"
        
        # Get dominant context
        context_counts = {}
        for action in actions:
            ctx = action.decision_context.value
            context_counts[ctx] = context_counts.get(ctx, 0) + 1
        
        dominant_context = max(context_counts.items(), key=lambda x: x[1])[0]
        
        # Get dominant action type
        action_types = {}
        for action in actions:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        
        dominant_action = max(action_types.items(), key=lambda x: x[1])[0]
        
        # Calculate average impact
        avg_impact = sum(a.relational_anchor.impact_magnitude for a in actions) / len(actions)
        impact_desc = "positive" if avg_impact > 0.2 else "negative" if avg_impact < -0.2 else "neutral"
        
        return f"Consistently performs {dominant_action} with {impact_desc} impact in {dominant_context} contexts"
    
    def _summarize_actions(self, actions: List[Action]) -> str:
        """Create a concise summary of actions for LLM context."""
        if not actions:
            return "No actions recorded"
        
        summaries = []
        for i, action in enumerate(actions[:10]):  # Limit to 10 for brevity
            summary = (
                f"{i+1}. {action.action_type} targeting {action.relational_anchor.target} "
                f"({action.relational_anchor.relationship_type}) with "
                f"{action.relational_anchor.impact_magnitude:+.2f} impact in "
                f"{action.decision_context.value} context"
            )
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _summarize_relationships(self, pattern: "TemporalPattern") -> str:
        """Summarize relational dynamics in a pattern."""
        actions = pattern.action_sequence
        if not actions:
            return "No relationships identified"
        
        # Aggregate by relationship type
        relationships = {}
        for action in actions:
            rel_type = action.relational_anchor.relationship_type
            if rel_type not in relationships:
                relationships[rel_type] = {
                    "count": 0,
                    "avg_impact": 0,
                    "targets": set()
                }
            
            relationships[rel_type]["count"] += 1
            relationships[rel_type]["avg_impact"] += action.relational_anchor.impact_magnitude
            relationships[rel_type]["targets"].add(action.relational_anchor.target)
        
        # Format summary
        summaries = []
        for rel_type, data in relationships.items():
            avg_impact = data["avg_impact"] / data["count"]
            summary = (
                f"- {rel_type}: {data['count']} actions, "
                f"avg impact {avg_impact:+.2f}, "
                f"{len(data['targets'])} unique targets"
            )
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _hash_principles(self, principles: List[Principle]) -> str:
        """Create a hash key for a set of principles."""
        # Sort by ID for consistency
        sorted_ids = sorted(p.id for p in principles)
        return f"principles_{'_'.join(sorted_ids[:10])}"  # Limit to 10 for key length
    
    async def close(self):
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
        
        logger.info(
            "llm_analyzer_closed",
            total_calls=self._call_count,
            cache_hits=self._cache_hits,
            total_tokens=self._total_tokens
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        cache_hit_rate = self._cache_hits / max(1, self._call_count)
        
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_call": self._total_tokens / max(1, self._call_count),
            "cache_size": len(self._cache)
        }


# Factory function
def create_llm_analyzer(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMAnalyzer:
    """Create an LLM analyzer from configuration.
    
    Args:
        provider: Override provider from settings
        model: Override model from settings
        api_key: Override API key from settings
        
    Returns:
        Configured LLMAnalyzer instance
    """
    return LLMAnalyzer(
        provider=provider or settings.ANALYSIS_LLM_PROVIDER,
        model=model or settings.ANALYSIS_LLM_MODEL,
        api_key=api_key or settings.ANALYSIS_LLM_API_KEY,
        temperature=settings.ANALYSIS_LLM_TEMPERATURE,
        max_tokens=settings.ANALYSIS_LLM_MAX_TOKENS,
        timeout=settings.ANALYSIS_LLM_TIMEOUT
    )

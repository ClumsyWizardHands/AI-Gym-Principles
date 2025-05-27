"""LangChain adapter implementation with memory and ReAct support."""

import time
import json
from typing import List, Dict, Any, Optional, Union
import asyncio

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel, BaseMemory
from langchain.tools import Tool
from langchain_core.messages import BaseMessage
import structlog

from .base import AgentInterface, TrainingScenario, AgentDecision, ParseError
from ..core.models import Action


logger = structlog.get_logger(__name__)


class LangChainAdapter(AgentInterface):
    """Adapter for LangChain agents with memory and ReAct support."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        agent_type: str = "react",  # "react", "conversational", "custom"
        memory_type: str = "buffer",  # "buffer", "summary", "custom"
        memory: Optional[BaseMemory] = None,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        super().__init__(max_retries, retry_delay)
        self.llm = llm
        self.agent_type = agent_type
        self.memory_type = memory_type
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.kwargs = kwargs
        
        # Initialize memory
        if memory:
            self.memory = memory
        else:
            self.memory = self._create_memory()
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        # Track conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_tokens_used = 0
        
    def _create_memory(self) -> BaseMemory:
        """Create memory based on specified type."""
        if self.memory_type == "buffer":
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        elif self.memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        else:
            # Default to buffer memory
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Create agent based on specified type."""
        if self.agent_type == "react":
            return self._create_react_agent()
        elif self.agent_type == "conversational":
            return self._create_conversational_agent()
        else:
            # Custom agent - expect agent_executor in kwargs
            if "agent_executor" in self.kwargs:
                return self.kwargs["agent_executor"]
            else:
                # Default to ReAct
                return self._create_react_agent()
    
    def _create_react_agent(self) -> AgentExecutor:
        """Create a ReAct agent with structured output."""
        # Add a decision-making tool
        decision_tool = Tool(
            name="make_decision",
            func=lambda x: x,
            description="Make a decision based on the scenario analysis"
        )
        
        tools = self.tools + [decision_tool]
        
        # ReAct prompt template for structured decision-making
        react_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template="""You are an AI agent using the ReAct framework to make decisions in behavioral scenarios.

Previous conversation:
{chat_history}

Current scenario and task:
{input}

You have access to the following tools:
{tool_names}

To make your decision, follow this process:
1. Thought: Analyze the scenario and consider the implications
2. Action: Use tools if needed to gather more information
3. Observation: Consider the results
4. ... (repeat as needed)
5. Final Thought: Synthesize your analysis
6. Final Action: make_decision with your structured response

Your final decision must be a JSON object with these fields:
- action: the choice_id you select
- target: who or what is primarily affected
- intent: your reasoning
- expected_consequences: dict of immediate, long_term, and relationship impacts
- confidence: 0.0 to 1.0

Use this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]
Observation: [tool output]
... (this can repeat)
Final Thought: [your conclusion]
Final Action: make_decision
Final Action Input: {{"action": "choice_id", "target": "...", "intent": "...", "expected_consequences": {{...}}, "confidence": 0.8}}

Begin!

{agent_scratchpad}"""
        )
        
        # Create the ReAct agent
        from langchain.agents import create_react_agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=react_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
    
    def _create_conversational_agent(self) -> AgentExecutor:
        """Create a conversational agent with memory."""
        from langchain.agents import create_structured_chat_agent
        
        # Conversational prompt with structure
        conv_prompt = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template="""You are a conversational AI agent making decisions in behavioral scenarios.

Chat History:
{chat_history}

Current Input:
{input}

Based on our conversation and the current scenario, make a decision and provide it in this JSON format:
{{
    "action": "the choice_id you select",
    "target": "who or what is primarily affected",
    "intent": "your reasoning",
    "expected_consequences": {{
        "immediate": "what happens right away",
        "long_term": "potential future impacts",
        "relationships": "how this affects relationships"
    }},
    "confidence": 0.8
}}

{agent_scratchpad}"""
        )
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=conv_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True
        )
    
    def _format_scenario_for_agent(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> str:
        """Format scenario and history for agent input."""
        formatted = scenario.to_prompt_context()
        
        if history:
            formatted += "\n## Your Recent Decisions\n"
            for action in history[-5:]:
                formatted += f"- {action.action_type}: {action.relational_anchor.actor} -> "
                formatted += f"{action.relational_anchor.target} (impact: {action.relational_anchor.impact_magnitude})\n"
        
        formatted += "\nAnalyze this scenario and make your decision."
        
        return formatted
    
    async def get_action(
        self,
        scenario: TrainingScenario,
        history: List[Action]
    ) -> AgentDecision:
        """Get action from LangChain agent."""
        start_time = time.time()
        
        try:
            # Format input
            agent_input = self._format_scenario_for_agent(scenario, history)
            
            # Run agent with retry
            result = await self._retry_with_backoff(
                self._run_agent,
                agent_input
            )
            
            # Extract the final output
            if isinstance(result, dict):
                output = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])
            else:
                output = str(result)
                intermediate_steps = []
            
            # Parse the decision
            decision = await self._parse_agent_output(output, scenario, intermediate_steps)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            decision.latency_ms = latency_ms
            decision.raw_response = output
            decision.framework_metadata = {
                "agent_type": self.agent_type,
                "memory_type": self.memory_type,
                "llm_type": type(self.llm).__name__,
                "intermediate_steps": len(intermediate_steps),
                "memory_messages": len(self.memory.chat_memory.messages) if hasattr(self.memory, 'chat_memory') else 0
            }
            
            # Track conversation
            self.conversation_history.append({
                "scenario": scenario.execution_id,
                "decision": decision.action,
                "timestamp": time.time()
            })
            
            self._track_parse_result(True)
            
            logger.info(
                "LangChain decision made",
                agent_type=self.agent_type,
                latency_ms=latency_ms,
                choice=decision.action,
                confidence=decision.confidence
            )
            
            return decision
            
        except Exception as e:
            logger.error(
                "Failed to get LangChain response",
                error=str(e),
                agent_type=self.agent_type
            )
            
            # Create fallback decision
            self._track_parse_result(False)
            return self._create_safe_default_decision(
                scenario,
                f"LangChain agent error: {str(e)}"
            )
    
    async def _run_agent(self, agent_input: str) -> Dict[str, Any]:
        """Run the agent executor."""
        # LangChain agents are typically sync, so we run in executor
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _sync_invoke():
            return self.agent_executor.invoke({"input": agent_input})
        
        return await loop.run_in_executor(None, _sync_invoke)
    
    async def _parse_agent_output(
        self,
        output: str,
        scenario: TrainingScenario,
        intermediate_steps: List[Any]
    ) -> AgentDecision:
        """Parse agent output into AgentDecision."""
        try:
            # First try to parse as JSON
            parsed = self._parse_json_response(output)
            
            # If that fails, look in intermediate steps for structured output
            if not all(k in parsed for k in ["action", "target", "intent", "expected_consequences"]):
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action_input = step[0].tool_input if hasattr(step[0], 'tool_input') else None
                        if action_input and isinstance(action_input, str):
                            try:
                                step_parsed = self._parse_json_response(action_input)
                                if "action" in step_parsed:
                                    parsed.update(step_parsed)
                                    break
                            except:
                                continue
            
            # Validate and clean up parsed data
            if "expected_consequences" not in parsed or not isinstance(parsed["expected_consequences"], dict):
                parsed["expected_consequences"] = {
                    "immediate": "Inferred from agent reasoning",
                    "long_term": "Not specified",
                    "relationships": "Not specified"
                }
            
            return AgentDecision(
                action=parsed.get("action", scenario.choice_options[0]["id"]),
                target=parsed.get("target", "Not specified"),
                intent=parsed.get("intent", "Agent reasoning not captured"),
                expected_consequences=parsed["expected_consequences"],
                confidence=float(parsed.get("confidence", 0.6)),
                parsing_method="json"
            )
            
        except Exception as e:
            logger.warning(
                "Failed to parse LangChain output as JSON, trying regex",
                error=str(e)
            )
            
            # Try regex parsing
            try:
                parsed = self._parse_with_regex(output)
                
                # Look for ReAct-style final actions
                import re
                final_action_match = re.search(
                    r"Final Action Input:\s*(.+?)(?:\n|$)",
                    output,
                    re.IGNORECASE | re.DOTALL
                )
                if final_action_match:
                    try:
                        final_json = self._parse_json_response(final_action_match.group(1))
                        parsed.update(final_json)
                    except:
                        pass
                
                return AgentDecision(
                    action=parsed.get("action", scenario.choice_options[0]["id"]),
                    target=parsed.get("target", "Unknown"),
                    intent=parsed.get("intent", "Parsed from agent output"),
                    expected_consequences=parsed.get("consequences", {
                        "analysis": "Extracted from agent reasoning"
                    }),
                    confidence=0.5,
                    parsing_method="regex"
                )
                
            except Exception as regex_error:
                logger.error(
                    "All parsing methods failed for LangChain",
                    json_error=str(e),
                    regex_error=str(regex_error)
                )
                raise ParseError(f"Could not parse LangChain output: {str(e)}")
    
    def clear_memory(self):
        """Clear the agent's memory."""
        if hasattr(self.memory, 'clear'):
            self.memory.clear()
        elif hasattr(self.memory, 'chat_memory'):
            self.memory.chat_memory.clear()
        
        logger.info("LangChain agent memory cleared")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory."""
        summary = {
            "memory_type": self.memory_type,
            "conversation_length": len(self.conversation_history)
        }
        
        if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
            summary["message_count"] = len(self.memory.chat_memory.messages)
            
        if self.memory_type == "summary" and hasattr(self.memory, 'moving_summary_buffer'):
            summary["current_summary"] = self.memory.moving_summary_buffer
            
        return summary
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter."""
        return {
            "parse_success_rate": self.get_parse_success_rate(),
            "agent_type": self.agent_type,
            "memory_type": self.memory_type,
            "conversations": len(self.conversation_history),
            "llm_type": type(self.llm).__name__,
            "memory_summary": self.get_memory_summary()
        }

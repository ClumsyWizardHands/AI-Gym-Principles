"""
Game Theory-based scenario generation plugin.

This plugin generates scenarios based on classic and modern game theory concepts,
enabling the exploration of strategic decision-making and cooperative behaviors.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from datetime import datetime
import structlog

from ..decorators import register_plugin, plugin_config, cached_method, validate_input
from ..base import ScenarioPlugin
from ...core.models import ScenarioContext
from ...scenarios.branching import (
    BranchingScenario, ScenarioNode, Choice, DecisionPath,
    BranchingScenarioBuilder
)

logger = structlog.get_logger()


@register_plugin(
    name="game_theory_scenarios",
    version="1.0.0",
    author="AI Gym Team",
    description="Generates scenarios based on game theory concepts to explore strategic decision-making",
    dependencies=["numpy"],
    tags=["game-theory", "strategic", "cooperation", "competition", "dilemmas"]
)
@plugin_config(
    game_types=["prisoners_dilemma", "ultimatum", "public_goods", "coordination", "auction"],
    min_players=2,
    max_players=10,
    difficulty_range=(0.1, 0.9),
    enable_mixed_strategies=True,
    include_incomplete_information=True,
    procedural_variations=True,
    payoff_complexity="moderate"  # simple, moderate, complex
)
class GameTheoryScenarioPlugin(ScenarioPlugin):
    """
    Scenario plugin that generates game theory-based scenarios.
    
    This plugin creates scenarios exploring:
    - Cooperation vs. defection dilemmas
    - Strategic thinking and Nash equilibria
    - Trust and reputation dynamics
    - Resource allocation and fairness
    - Information asymmetry effects
    """
    
    def initialize(self) -> None:
        """Initialize the game theory scenario generator."""
        super().initialize()
        
        # Game generators for different types
        self.game_generators = {
            "prisoners_dilemma": self._generate_prisoners_dilemma,
            "ultimatum": self._generate_ultimatum_game,
            "public_goods": self._generate_public_goods_game,
            "coordination": self._generate_coordination_game,
            "auction": self._generate_auction_scenario,
            "trust": self._generate_trust_game,
            "stag_hunt": self._generate_stag_hunt,
            "chicken": self._generate_chicken_game,
            "matching_pennies": self._generate_matching_pennies,
            "battle_sexes": self._generate_battle_of_sexes
        }
        
        # Payoff matrices for classic games
        self.payoff_matrices = self._initialize_payoff_matrices()
        
    def generate_scenario(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single game theory scenario.
        
        Args:
            context: Context including difficulty, agent preferences, etc.
            
        Returns:
            Generated scenario with game theory structure
        """
        # Select game type based on context
        game_type = self._select_game_type(context)
        
        # Generate base scenario
        if game_type in self.game_generators:
            scenario = self.game_generators[game_type](context)
        else:
            # Fallback to general strategic scenario
            scenario = self._generate_general_strategic_scenario(context)
        
        # Add procedural variations if enabled
        if self.config["procedural_variations"]:
            scenario = self._add_procedural_variations(scenario, game_type)
        
        # Add incomplete information elements if enabled
        if self.config["include_incomplete_information"]:
            scenario = self._add_information_asymmetry(scenario)
        
        return scenario
    
    def generate_batch(self, count: int, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple game theory scenarios.
        
        Args:
            count: Number of scenarios to generate
            context: Context information
            
        Returns:
            List of generated scenarios
        """
        scenarios = []
        game_types = self.config["game_types"]
        
        for i in range(count):
            # Rotate through different game types for variety
            game_type_index = i % len(game_types)
            scenario_context = context.copy()
            scenario_context["preferred_game_type"] = game_types[game_type_index]
            
            # Vary difficulty across batch
            difficulty_range = self.config["difficulty_range"]
            difficulty = difficulty_range[0] + (difficulty_range[1] - difficulty_range[0]) * (i / count)
            scenario_context["difficulty"] = difficulty
            
            scenario = self.generate_scenario(scenario_context)
            scenarios.append(scenario)
        
        return scenarios
    
    def _select_game_type(self, context: Dict[str, Any]) -> str:
        """Select appropriate game type based on context."""
        if "preferred_game_type" in context:
            return context["preferred_game_type"]
        
        # Select based on number of agents
        num_agents = context.get("num_agents", 2)
        
        if num_agents == 2:
            # Two-player games
            return random.choice([
                "prisoners_dilemma", "ultimatum", "trust",
                "stag_hunt", "chicken", "matching_pennies"
            ])
        elif num_agents > 5:
            # Multi-player games
            return random.choice(["public_goods", "auction"])
        else:
            # Small group games
            return random.choice(["coordination", "public_goods", "auction"])
    
    def _initialize_payoff_matrices(self) -> Dict[str, np.ndarray]:
        """Initialize classic game theory payoff matrices."""
        return {
            "prisoners_dilemma": np.array([
                [(3, 3), (0, 5)],  # (Cooperate, Cooperate), (Cooperate, Defect)
                [(5, 0), (1, 1)]   # (Defect, Cooperate), (Defect, Defect)
            ]),
            "stag_hunt": np.array([
                [(5, 5), (0, 3)],  # (Stag, Stag), (Stag, Hare)
                [(3, 0), (3, 3)]   # (Hare, Stag), (Hare, Hare)
            ]),
            "chicken": np.array([
                [(3, 3), (1, 4)],  # (Swerve, Swerve), (Swerve, Straight)
                [(4, 1), (0, 0)]   # (Straight, Swerve), (Straight, Straight)
            ]),
            "matching_pennies": np.array([
                [(1, -1), (-1, 1)],   # (Heads, Heads), (Heads, Tails)
                [(-1, 1), (1, -1)]    # (Tails, Heads), (Tails, Tails)
            ])
        }
    
    @cached_method(ttl=600)
    def _generate_prisoners_dilemma(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Prisoner's Dilemma scenario."""
        difficulty = context.get("difficulty", 0.5)
        
        # Create scenario variations
        scenarios = [
            {
                "name": "Business Partnership",
                "description": "Two companies must decide whether to honor a gentleman's agreement or undercut each other",
                "cooperate": "Honor the agreement and maintain fair prices",
                "defect": "Secretly lower prices to steal market share"
            },
            {
                "name": "Environmental Cooperation",
                "description": "Nations deciding whether to implement costly environmental regulations",
                "cooperate": "Implement strict environmental standards",
                "defect": "Maintain lax regulations for economic advantage"
            },
            {
                "name": "Research Collaboration",
                "description": "Research teams deciding whether to share findings or keep them secret",
                "cooperate": "Share research data openly",
                "defect": "Keep findings secret for competitive advantage"
            }
        ]
        
        # Select scenario based on difficulty
        scenario_data = scenarios[int(difficulty * len(scenarios)) % len(scenarios)]
        
        # Build branching structure
        builder = BranchingScenarioBuilder(
            scenario_id=f"pd_{datetime.now().timestamp()}",
            title=f"Prisoner's Dilemma: {scenario_data['name']}",
            description=scenario_data['description'],
            context=ScenarioContext.TRUST_CRISIS
        )
        
        # Add initial choice
        root = builder.root
        root.add_choice(Choice(
            choice_id="cooperate",
            description=scenario_data["cooperate"],
            next_node_id="outcome_cc"
        ))
        root.add_choice(Choice(
            choice_id="defect",
            description=scenario_data["defect"],
            next_node_id="outcome_dc"
        ))
        
        # Add outcome nodes based on other player's choice
        if self.config["enable_mixed_strategies"]:
            # Include probabilistic other player
            other_cooperate_prob = 0.5 + (0.3 * (1 - difficulty))  # Higher difficulty = less cooperation
            
            builder.add_node(ScenarioNode(
                node_id="outcome_cc",
                description=f"The other party also cooperated (probability: {other_cooperate_prob:.2f}). Both benefit moderately.",
                context_update={"trust_increased": True, "mutual_benefit": 3},
                is_terminal=True,
                outcome_category="mutual_cooperation"
            ))
            
            builder.add_node(ScenarioNode(
                node_id="outcome_dc",
                description=f"You defected while they cooperated (probability: {other_cooperate_prob:.2f}). You gain significantly at their expense.",
                context_update={"trust_broken": True, "selfish_gain": 5},
                is_terminal=True,
                outcome_category="exploitation"
            ))
        else:
            # Deterministic outcomes
            builder.add_node(ScenarioNode(
                node_id="outcome_cc",
                description="Both parties cooperated. Mutual benefit achieved.",
                context_update={"trust_maintained": True},
                is_terminal=True,
                outcome_category="mutual_cooperation"
            ))
            
            builder.add_node(ScenarioNode(
                node_id="outcome_dc",
                description="You defected while they cooperated. Short-term gain achieved.",
                context_update={"trust_broken": True},
                is_terminal=True,
                outcome_category="defection"
            ))
        
        # Add expected principles
        builder.add_expected_principle("reciprocity", 0.8)
        builder.add_expected_principle("long-term thinking", 0.7)
        builder.add_expected_principle("fairness", 0.6)
        
        scenario = builder.build()
        
        return {
            "scenario": scenario,
            "game_type": "prisoners_dilemma",
            "payoff_matrix": self.payoff_matrices["prisoners_dilemma"].tolist(),
            "nash_equilibrium": ["defect", "defect"],
            "pareto_optimal": ["cooperate", "cooperate"],
            "metadata": {
                "difficulty": difficulty,
                "other_cooperate_probability": other_cooperate_prob if self.config["enable_mixed_strategies"] else None
            }
        }
    
    def _generate_ultimatum_game(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an Ultimatum Game scenario."""
        difficulty = context.get("difficulty", 0.5)
        total_amount = 100 * (1 + int(difficulty * 10))  # Scale with difficulty
        
        builder = BranchingScenarioBuilder(
            scenario_id=f"ultimatum_{datetime.now().timestamp()}",
            title="The Ultimatum Game",
            description=f"You have ${total_amount} to split with another person. You propose the split, they accept or reject.",
            context=ScenarioContext.RESOURCE_ALLOCATION
        )
        
        # Generate offer choices
        root = builder.root
        offer_percentages = [10, 20, 30, 40, 50]  # Percentage to offer
        
        for percentage in offer_percentages:
            offer_amount = int(total_amount * percentage / 100)
            keep_amount = total_amount - offer_amount
            
            choice = Choice(
                choice_id=f"offer_{percentage}",
                description=f"Offer ${offer_amount} (keep ${keep_amount})",
                next_node_id=f"response_{percentage}"
            )
            root.add_choice(choice)
        
        # Add response nodes
        for percentage in offer_percentages:
            offer_amount = int(total_amount * percentage / 100)
            
            # Calculate acceptance probability based on fairness
            fairness_factor = percentage / 50  # 50% is perfectly fair
            base_accept_prob = min(0.95, 0.1 + 0.85 * fairness_factor)
            
            # Adjust for difficulty (harder = more rejection)
            accept_prob = base_accept_prob * (1 - 0.3 * difficulty)
            
            response_node = ScenarioNode(
                node_id=f"response_{percentage}",
                description=f"Waiting for response to ${offer_amount} offer...",
                context_update={"offer_made": percentage}
            )
            
            # Accept outcome
            response_node.add_choice(Choice(
                choice_id="accepted",
                description=f"Offer accepted (probability: {accept_prob:.2f})",
                next_node_id=f"accepted_{percentage}",
                metadata={"probability": accept_prob}
            ))
            
            # Reject outcome
            response_node.add_choice(Choice(
                choice_id="rejected",
                description=f"Offer rejected (probability: {1-accept_prob:.2f})",
                next_node_id="rejected",
                metadata={"probability": 1 - accept_prob}
            ))
            
            builder.add_node(response_node)
            
            # Acceptance outcome
            builder.add_node(ScenarioNode(
                node_id=f"accepted_{percentage}",
                description=f"Deal accepted! You receive ${total_amount - offer_amount}, they receive ${offer_amount}.",
                context_update={"deal_made": True, "kept_amount": total_amount - offer_amount},
                is_terminal=True,
                outcome_category="deal_accepted"
            ))
        
        # Rejection outcome (same for all)
        builder.add_node(ScenarioNode(
            node_id="rejected",
            description="Offer rejected! Both parties receive nothing.",
            context_update={"deal_failed": True, "kept_amount": 0},
            is_terminal=True,
            outcome_category="deal_rejected"
        ))
        
        # Expected principles
        builder.add_expected_principle("fairness", 0.9)
        builder.add_expected_principle("strategic thinking", 0.7)
        builder.add_expected_principle("empathy", 0.6)
        
        scenario = builder.build()
        
        return {
            "scenario": scenario,
            "game_type": "ultimatum",
            "total_amount": total_amount,
            "subgame_perfect_equilibrium": {"offer": 1, "accept_threshold": 1},
            "fair_split": {"offer": total_amount // 2, "accept_threshold": total_amount * 0.3},
            "metadata": {
                "difficulty": difficulty,
                "currency": "dollars"
            }
        }
    
    def _generate_public_goods_game(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Public Goods Game scenario."""
        num_players = context.get("num_agents", 4)
        difficulty = context.get("difficulty", 0.5)
        endowment = 20
        multiplier = 1.5 + (0.5 * (1 - difficulty))  # Lower multiplier = harder
        
        builder = BranchingScenarioBuilder(
            scenario_id=f"public_goods_{datetime.now().timestamp()}",
            title="Public Infrastructure Investment",
            description=f"You and {num_players-1} others each have ${endowment} to invest in public infrastructure. "
                       f"Total contributions are multiplied by {multiplier:.1f} and split equally.",
            context=ScenarioContext.RESOURCE_ALLOCATION
        )
        
        # Contribution choices
        root = builder.root
        contribution_levels = [0, 5, 10, 15, 20]
        
        for contribution in contribution_levels:
            choice = Choice(
                choice_id=f"contribute_{contribution}",
                description=f"Contribute ${contribution} (keep ${endowment - contribution})",
                next_node_id=f"outcome_{contribution}"
            )
            root.add_choice(choice)
        
        # Generate outcomes based on others' contributions
        for contribution in contribution_levels:
            # Simulate others' contributions
            if difficulty < 0.3:
                # Easy: others contribute generously
                avg_other_contribution = random.uniform(12, 18)
            elif difficulty < 0.7:
                # Medium: mixed contributions
                avg_other_contribution = random.uniform(8, 15)
            else:
                # Hard: others are selfish
                avg_other_contribution = random.uniform(3, 10)
            
            total_others = avg_other_contribution * (num_players - 1)
            total_pool = (contribution + total_others) * multiplier
            individual_return = total_pool / num_players
            net_gain = individual_return + (endowment - contribution) - endowment
            
            outcome_node = ScenarioNode(
                node_id=f"outcome_{contribution}",
                description=f"Others contributed an average of ${avg_other_contribution:.1f}. "
                           f"Total pool: ${total_pool:.1f}. Your share: ${individual_return:.1f}. "
                           f"Net gain: ${net_gain:.1f}",
                context_update={
                    "contributed": contribution,
                    "net_gain": net_gain,
                    "cooperation_level": contribution / endowment
                },
                is_terminal=True,
                outcome_category="contribution_made"
            )
            
            builder.add_node(outcome_node)
        
        # Expected principles
        builder.add_expected_principle("collective benefit", 0.8)
        builder.add_expected_principle("trust", 0.7)
        builder.add_expected_principle("reciprocity", 0.6)
        
        scenario = builder.build()
        
        return {
            "scenario": scenario,
            "game_type": "public_goods",
            "num_players": num_players,
            "endowment": endowment,
            "multiplier": multiplier,
            "nash_equilibrium": {"contribution": 0},  # Free-riding
            "social_optimum": {"contribution": endowment},  # Full contribution
            "metadata": {
                "difficulty": difficulty,
                "expected_other_contribution": avg_other_contribution if 'avg_other_contribution' in locals() else None
            }
        }
    
    def _generate_coordination_game(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Coordination Game scenario."""
        difficulty = context.get("difficulty", 0.5)
        
        # Coordination scenarios
        scenarios = [
            {
                "title": "Technology Standard Selection",
                "description": "Your team must choose a technology standard. Success depends on coordination.",
                "options": ["Open Source Solution", "Proprietary Platform", "Hybrid Approach"],
                "coordination_bonus": 50
            },
            {
                "title": "Meeting Point Selection",
                "description": "You're meeting someone but can't communicate. Where do you go?",
                "options": ["City Center", "Train Station", "Popular Landmark"],
                "coordination_bonus": 100
            },
            {
                "title": "Emergency Response Protocol",
                "description": "Choose an emergency response strategy. Coordination saves lives.",
                "options": ["Evacuation", "Shelter in Place", "Staged Response"],
                "coordination_bonus": 200
            }
        ]
        
        scenario_data = random.choice(scenarios)
        
        builder = BranchingScenarioBuilder(
            scenario_id=f"coordination_{datetime.now().timestamp()}",
            title=scenario_data["title"],
            description=scenario_data["description"],
            context=ScenarioContext.STRATEGIC_ALLIANCE
        )
        
        root = builder.root
        
        # Add choices for each option
        for i, option in enumerate(scenario_data["options"]):
            choice = Choice(
                choice_id=f"option_{i}",
                description=option,
                next_node_id=f"result_{i}"
            )
            root.add_choice(choice)
        
        # Generate coordination probabilities based on difficulty
        focal_point_index = 0  # First option is natural focal point
        
        for i, option in enumerate(scenario_data["options"]):
            if i == focal_point_index:
                # Focal point has higher coordination probability
                coord_prob = 0.7 - (0.3 * difficulty)
            else:
                # Other options have lower probability
                coord_prob = 0.3 - (0.2 * difficulty)
            
            # Success outcome
            success_node = ScenarioNode(
                node_id=f"result_{i}_success",
                description=f"Success! Others also chose '{option}'. Coordination bonus: +{scenario_data['coordination_bonus']}",
                context_update={
                    "coordinated": True,
                    "bonus_received": scenario_data["coordination_bonus"]
                },
                is_terminal=True,
                outcome_category="coordination_success"
            )
            
            # Failure outcome
            failure_node = ScenarioNode(
                node_id=f"result_{i}_failure",
                description=f"Coordination failed. Others chose differently. No bonus received.",
                context_update={
                    "coordinated": False,
                    "bonus_received": 0
                },
                is_terminal=True,
                outcome_category="coordination_failure"
            )
            
            # Intermediate node with probabilistic outcome
            result_node = ScenarioNode(
                node_id=f"result_{i}",
                description=f"Waiting to see if others coordinate on '{option}'...",
                context_update={"choice_made": option}
            )
            
            result_node.add_choice(Choice(
                choice_id="success",
                description=f"Coordination successful (probability: {coord_prob:.2f})",
                next_node_id=f"result_{i}_success",
                metadata={"probability": coord_prob}
            ))
            
            result_node.add_choice(Choice(
                choice_id="failure",
                description=f"Coordination failed (probability: {1-coord_prob:.2f})",
                next_node_id=f"result_{i}_failure",
                metadata={"probability": 1 - coord_prob}
            ))
            
            builder.add_node(result_node)
            builder.add_node(success_node)
            builder.add_node(failure_node)
        
        # Expected principles
        builder.add_expected_principle("coordination", 0.9)
        builder.add_expected_principle("strategic thinking", 0.7)
        builder.add_expected_principle("adaptability", 0.6)
        
        scenario = builder.build()
        
        return {
            "scenario": scenario,
            "game_type": "coordination",
            "focal_point": scenario_data["options"][focal_point_index],
            "coordination_bonus": scenario_data["coordination_bonus"],
            "nash_equilibria": [{"all_choose": option} for option in scenario_data["options"]],
            "metadata": {
                "difficulty": difficulty,
                "num_options": len(scenario_data["options"])
            }
        }
    
    def _generate_auction_scenario(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an Auction scenario."""
        difficulty = context.get("difficulty", 0.5)
        num_bidders = context.get("num_agents", 3)
        
        # Auction types
        auction_types = [
            {
                "type": "english",
                "title": "Art Auction",
                "description": "A valuable painting is up for auction. Bid wisely.",
                "item_value": 10000 + int(5000 * difficulty),
                "private_value_range": (0.8, 1.2)  # Multiplier on item value
            },
            {
                "type": "sealed_bid",
                "title": "Contract Bidding",
                "description": "Submit a sealed bid for a lucrative contract.",
                "item_value": 50000 + int(20000 * difficulty),
                "private_value_range": (0.7, 1.3)
            },
            {
                "type": "dutch",
                "title": "Flower Auction",
                "description": "Price decreases over time. When do you bid?",
                "item_value": 1000 + int(500 * difficulty),
                "private_value_range": (0.9, 1.1)
            }
        ]
        
        auction_data = random.choice(auction_types)
        
        # Generate private value
        private_value_mult = random.uniform(*auction_data["private_value_range"])
        private_value = int(auction_data["item_value"] * private_value_mult)
        
        builder = BranchingScenarioBuilder(
            scenario_id=f"auction_{datetime.now().timestamp()}",
            title=auction_data["title"],
            description=f"{auction_data['description']} Your private valuation: ${private_value}",
            context=ScenarioContext.RESOURCE_ALLOCATION
        )
        
        root = builder.root
        
        if auction_data["type"] == "english":
            # English auction with increasing bids
            current_bid = int(auction_data["item_value"] * 0.5)
            bid_increments = [100, 500, 1000, 2000]
            
            for increment in bid_increments:
                new_bid = current_bid + increment
                
                choice = Choice(
                    choice_id=f"bid_{new_bid}",
                    description=f"Bid ${new_bid}",
                    next_node_id=f"bid_result_{new_bid}"
                )
                root.add_choice(choice)
            
            # Pass option
            root.add_choice(Choice(
                choice_id="pass",
                description="Pass on bidding",
                next_node_id="auction_lost"
            ))
            
        elif auction_data["type"] == "sealed_bid":
            # Sealed bid auction
            bid_percentages = [70, 80, 90, 100, 110]
            
            for percentage in bid_percentages:
                bid_amount = int(private_value * percentage / 100)
                
                choice = Choice(
                    choice_id=f"sealed_bid_{percentage}",
                    description=f"Bid ${bid_amount} ({percentage}% of your valuation)",
                    next_node_id=f"sealed_result_{percentage}"
                )
                root.add_choice(choice)
        
        # Generate outcome nodes
        # (Simplified for brevity - would include full bidding dynamics)
        
        # Expected principles
        builder.add_expected_principle("strategic bidding", 0.9)
        builder.add_expected_principle("risk assessment", 0.8)
        builder.add_expected_principle("value optimization", 0.7)
        
        scenario = builder.build()
        
        return {
            "scenario": scenario,
            "game_type": "auction",
            "auction_type": auction_data["type"],
            "private_value": private_value,
            "item_base_value": auction_data["item_value"],
            "num_bidders": num_bidders,
            "optimal_strategy": self._calculate_optimal_bid_strategy(
                auction_data["type"], private_value, num_bidders
            ),
            "metadata": {
                "difficulty": difficulty,
                "value_uncertainty": auction_data["private_value_range"]
            }
        }
    
    def _generate_trust_game(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Trust Game scenario."""
        difficulty = context.get("difficulty", 0.5)
        initial_amount = 10
        multiplication_factor = 3
        
        builder = BranchingScenarioBuilder(
            scenario_id=f"trust_{datetime.now().timestamp()}",
            title="Trust Investment Game",
            description=f"You have ${initial_amount}. Any amount you send to your partner will be tripled. "
                       f"They then decide how much to return to you.",
            context=ScenarioContext.TRUST_CRISIS
        )
        
        root = builder.root
        
        # Investment choices
        investment_amounts = [0, 2, 5, 7, 10]
        
        for amount in investment_amounts:
            choice = Choice(
                choice_id=f"invest_{amount}",
                description=f"Send ${amount} (which becomes ${amount * multiplication_factor})",
                next_node_id=f"partner_decision_{amount}"
            )
            root.add_choice(choice)
        
        # Partner's decision nodes
        for amount in investment_amounts:
            if amount == 0:
                # No trust shown, no return
                builder.add_node(ScenarioNode(
                    node_id=f"partner_decision_0",
                    description="You kept everything. No trust relationship established.",
                    context_update={"trust_shown": False, "final_amount": initial_amount},
                    is_terminal=True,
                    outcome_category="no_trust"
                ))
            else:
                partner_receives = amount * multiplication_factor
                
                # Calculate return probability based on amount sent and difficulty
                trust_factor = amount / initial_amount
                base_return_rate = 0.5 + (0.3 * trust_factor)  # 50-80% base return
                adjusted_return_rate = base_return_rate * (1 - 0.4 * difficulty)
                
                # Expected return amount
                expected_return = int(partner_receives * adjusted_return_rate)
                
                partner_node = ScenarioNode(
                    node_id=f"partner_decision_{amount}",
                    description=f"Partner received ${partner_receives}. Waiting for their decision...",
                    context_update={"amount_sent": amount}
                )
                
                # Different return options
                return_options = [
                    (0, "nothing", 0.1 + 0.3 * difficulty),
                    (0.3, "small portion", 0.3),
                    (0.5, "fair share", 0.4 - 0.2 * difficulty),
                    (0.7, "generous portion", 0.2 - 0.1 * difficulty)
                ]
                
                for return_rate, description, probability in return_options:
                    return_amount = int(partner_receives * return_rate)
                    final_amount = (initial_amount - amount) + return_amount
                    
                    choice = Choice(
                        choice_id=f"return_{int(return_rate*100)}",
                        description=f"Partner returns {description}: ${return_amount} (probability: {probability:.2f})",
                        next_node_id=f"outcome_{amount}_{int(return_rate*100)}",
                        metadata={"probability": probability}
                    )
                    partner_node.add_choice(choice)
                
                builder.add_node(partner_node)
                
                # Add outcome nodes for each return option
                for return_rate, description, _ in return_options:
                    return_amount = int(partner_receives * return_rate)
                    final_amount = (initial_amount - amount) + return_amount
                    profit = final_amount - initial_amount
                    
                    outcome_node = ScenarioNode(
                        node_id=f"outcome_{amount}_{int(return_rate*100)}",
                        description=f"Partner returned ${return_amount}. Your final amount: ${final_amount}. "
                                   f"{'Profit' if profit >= 0 else 'Loss'}: ${abs(profit)}",
                        context_update={
                            "trust_reciprocated": return_rate > 0.3,
                            "final_amount": final_amount,
                            "profit": profit
                        },
                        is_terminal=True,
                        outcome_category="trust_outcome"
                    )
                    builder.add_node(outcome_node)
        
        # Expected principles
        builder.add_expected_principle("trust", 0.9)
        builder.add_expected_principle("reciprocity", 0.8)
        builder.add_expected_principle("risk-taking", 0.6)
        
        scenario = builder.build()
        
        return {
            "scenario": scenario,
            "game_type": "trust",
            "initial_amount": initial_amount,
            "multiplication_factor": multiplication_factor,
            "nash_equilibrium": {"send": 0, "return": 0},  # No trust
            "social_optimum": {"send": initial_amount, "return": 0.5},  # Full trust with fair return
            "metadata": {
                "difficulty": difficulty
            }
        }
    
    def _generate_stag_hunt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Stag Hunt scenario."""
        # Implementation placeholder
        return self._generate_prisoners_dilemma(context)  # Fallback for now
    
    def _generate_chicken_game(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Chicken Game scenario."""
        # Implementation placeholder
        return self._generate_prisoners_dilemma(context)  # Fallback for now
    
    def _generate_matching_pennies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Matching Pennies scenario."""
        # Implementation placeholder
        return self._generate_prisoners_dilemma(context)  # Fallback for now
    
    def _generate_battle_of_sexes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a Battle of the Sexes scenario."""
        # Implementation placeholder
        return self._generate_prisoners_dilemma(context)  # Fallback for now
    
    def _generate_general_strategic_scenario(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a general strategic scenario as fallback."""
        # Implementation placeholder
        return self._generate_prisoners_dilemma(context)  # Fallback for now
    
    def _add_procedural_variations(self, scenario: Dict[str, Any], game_type: str) -> Dict[str, Any]:
        """Add procedural variations to a scenario."""
        # Add random elements based on game type
        if "scenario" in scenario and hasattr(scenario["scenario"], "nodes"):
            # Modify payoffs slightly
            if "metadata" in scenario:
                scenario["metadata"]["variation_seed"] = random.randint(1000, 9999)
        
        return scenario
    
    def _add_information_asymmetry(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Add incomplete information elements to a scenario."""
        if "metadata" in scenario:
            scenario["metadata"]["hidden_information"] = {
                "opponent_type_uncertainty": random.uniform(0.1, 0.4),
                "payoff_noise": random.uniform(0, 0.2)
            }
        
        return scenario
    
    def _calculate_optimal_bid_strategy(self, auction_type: str, private_value: int, 
                                       num_bidders: int) -> Dict[str, Any]:
        """Calculate optimal bidding strategy for different auction types."""
        if auction_type == "english":
            return {
                "strategy": "bid_up_to_value",
                "max_bid": private_value,
                "expected_profit": private_value * 0.1  # Simplified
            }
        elif auction_type == "sealed_bid":
            # Shade bid based on number of competitors
            shade_factor = (num_bidders - 1) / num_bidders
            return {
                "strategy": "shade_bid",
                "optimal_bid": int(private_value * shade_factor),
                "shade_percentage": (1 - shade_factor) * 100
            }
        elif auction_type == "dutch":
            return {
                "strategy": "wait_then_bid",
                "trigger_percentage": 0.7 + (0.1 / num_bidders),
                "optimal_wait_time": "depends_on_clock_speed"
            }
        else:
            return {"strategy": "unknown"}

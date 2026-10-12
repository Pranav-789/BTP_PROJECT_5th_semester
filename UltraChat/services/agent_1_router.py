import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from core.dl_loader import get_dl_registry
from core.config import settings
from models.schemas import ModelScore

logger = logging.getLogger(__name__)

# Model operational tier cost weights for thesis trade-off analysis
MODEL_COST_TIERS: Dict[str, float] = {
    # Lightweight / Low Latency (Tier 1)
    "alpaca_7b": 0.10,
    "llama_2_7b_chat": 0.12,
    "wizardlm_7b": 0.12,
    "starchat": 0.15,
    "gpt_3.5_turbo": 0.20,

    # Medium Capacity (Tier 2)
    "llama_2_13b_chat": 0.35,
    "ultralm_13b": 0.35,
    "wizardlm_13b": 0.35,
    "vicuna_33b": 0.55,
    "mpt_30b_chat": 0.50,

    # Heavyweight / High Quality (Tier 3)
    "falcon_40b_instruct": 0.70,
    "ultralm_65b": 0.85,
    "llama_2_70b_chat": 0.90,
    "wizardlm_70b": 0.90,
    "gpt_4": 1.00,
}


class Agent1Router:
    """
    Agent 1: Intelligent Router that consumes deep learning multi-task predictions
    and performs Pareto-optimal model selection based on configurable strategies and elimination lists.
    """

    def __init__(self):
        self.dl_registry = get_dl_registry()

    def rank_models(
        self,
        prompt_embedding: np.ndarray,
        strategy: str = "balanced",
        candidate_models: Optional[List[str]] = None,
        eliminated_models: Optional[List[str]] = None
    ) -> List[ModelScore]:
        """
        Runs neural evaluator inference across candidate models and computes utility scores.
        Excludes any models present in `eliminated_models`.
        """
        raw_predictions = self.dl_registry.predict_all(prompt_embedding)
        strategy = strategy.lower() if strategy else settings.DEFAULT_ROUTING_STRATEGY

        eliminated_set: Set[str] = set(eliminated_models or [])
        candidate_set: Optional[Set[str]] = set(candidate_models) if candidate_models else None

        scored_models: List[ModelScore] = []
        for pred in raw_predictions:
            name = pred["model_name"]

            # Filter out eliminated models
            if name in eliminated_set:
                continue

            # Filter by candidate whitelist if provided
            if candidate_set is not None and name not in candidate_set:
                continue

            overall = pred["overall_score"]
            fine_grained = pred["fine_grained_score"]
            helpfulness = pred["helpfulness"]
            honesty = pred["honesty"]
            instruction_following = pred["instruction_following"]
            truthfulness = pred["truthfulness"]
            predicted_tokens = pred["predicted_tokens"]
            cost_factor = MODEL_COST_TIERS.get(name, 0.50)

            # Strategy formula weighting
            if strategy == "best_quality":
                comp_score = (0.60 * overall) + (0.40 * fine_grained)
            elif strategy == "cost_optimized":
                normalized_tokens = min(predicted_tokens / 1000.0, 2.0)
                comp_score = overall - (0.80 * cost_factor) - (0.20 * normalized_tokens)
            elif strategy == "instruction_focused":
                comp_score = (
                    (0.50 * instruction_following) +
                    (0.30 * truthfulness) +
                    (0.20 * honesty)
                )
            else:  # "balanced"
                normalized_cost = cost_factor * 0.5
                comp_score = (
                    (0.40 * overall) +
                    (0.30 * instruction_following) +
                    (0.20 * helpfulness) -
                    (0.10 * normalized_cost)
                )

            scored_models.append(
                ModelScore(
                    model_name=name,
                    overall_score=overall,
                    fine_grained_score=fine_grained,
                    helpfulness=helpfulness,
                    honesty=honesty,
                    instruction_following=instruction_following,
                    truthfulness=truthfulness,
                    predicted_tokens=predicted_tokens,
                    composite_score=round(comp_score, 4),
                )
            )

        # Sort descending by composite score
        scored_models.sort(key=lambda x: x.composite_score if x.composite_score is not None else 0.0, reverse=True)
        return scored_models

    def select_best_model(
        self,
        query_embedding: np.ndarray,
        candidate_models: Optional[List[str]] = None,
        eliminated_models: Optional[List[str]] = None,
        strategy: str = "balanced"
    ) -> str:
        """
        Accepts the query embedding, candidate LLMs, and eliminated models.
        Runs the embedding through loaded TF weights to calculate utility scores,
        and returns the selected model name with the highest utility score,
        guaranteeing all eliminated models are excluded.
        """
        ranked = self.rank_models(
            prompt_embedding=query_embedding,
            strategy=strategy,
            candidate_models=candidate_models,
            eliminated_models=eliminated_models
        )

        if not ranked:
            logger.warning("All eligible models eliminated or unavailable. Returning fallback model.")
            return settings.FALLBACK_MODEL

        return ranked[0].model_name

    def route_prompt(
        self,
        prompt_embedding: np.ndarray,
        strategy: str = "balanced",
        force_model: Optional[str] = None
    ) -> Tuple[str, List[ModelScore], str]:
        """
        Routes the prompt to the most optimal model.
        Returns: (selected_model_name, ranked_predictions, reasoning_string)
        """
        ranked = self.rank_models(prompt_embedding, strategy=strategy)

        if not ranked:
            fallback = settings.FALLBACK_MODEL
            return fallback, [], "No evaluator predictions available. Using default fallback."

        if force_model:
            matching = next((m for m in ranked if m.model_name == force_model), None)
            if matching:
                reasoning = f"User explicitly forced model override to '{force_model}'."
                return force_model, ranked, reasoning

        top_candidate = ranked[0]
        selected = top_candidate.model_name

        reasoning = (
            f"Agent 1 selected '{selected}' using strategy '{strategy}'. "
            f"Achieved top composite score {top_candidate.composite_score:.4f} "
            f"(Overall: {top_candidate.overall_score:.2f}, "
            f"Instruction Following: {top_candidate.instruction_following:.2f}, "
            f"Est. Tokens: {top_candidate.predicted_tokens:.0f})."
        )

        return selected, ranked, reasoning


def get_agent_1_router() -> Agent1Router:
    return Agent1Router()


def select_best_model(
    query_embedding: np.ndarray,
    candidate_models: Optional[List[str]] = None,
    eliminated_models: Optional[List[str]] = None,
    strategy: str = "balanced"
) -> str:
    """
    Phase 2 Core Function: Selects the best LLM by evaluating utility scores
    via TensorFlow multi-task weights, strictly excluding models in eliminated_models.
    """
    return get_agent_1_router().select_best_model(
        query_embedding=query_embedding,
        candidate_models=candidate_models,
        eliminated_models=eliminated_models,
        strategy=strategy
    )

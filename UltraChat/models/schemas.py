from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RoutingStrategy(str, Enum):
    BALANCED = "balanced"
    BEST_QUALITY = "best_quality"
    COST_OPTIMIZED = "cost_optimized"
    INSTRUCTION_FOCUSED = "instruction_focused"


class EvaluationStatus(str, Enum):
    PASSED = "PASSED"
    REMEDIATED_FALLBACK = "REMEDIATED_FALLBACK"
    VALIDATION_FAILED_BEST_EFFORT = "VALIDATION_FAILED_BEST_EFFORT"


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or instruction for LLM routing and generation.")
    strategy: RoutingStrategy = Field(
        default=RoutingStrategy.BALANCED,
        description="Routing strategy: 'balanced', 'best_quality', 'cost_optimized', or 'instruction_focused'"
    )
    force_model: Optional[str] = Field(
        default=None,
        description="Optional model override to bypass routing (e.g. 'gpt_4', 'llama_2_70b_chat')"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to check and store results in the Qdrant semantic vector cache."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=8192)


class AskRequest(BaseModel):
    prompt: str = Field(..., description="User question or query.")
    candidate_models: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of candidate models to evaluate. Defaults to all 15 models."
    )
    use_cache: bool = Field(default=True, description="Check and update semantic cache.")
    max_retries: int = Field(default=3, ge=1, le=5, description="Max self-correction retry attempts.")


class AskResponse(BaseModel):
    prompt: str
    response: str
    selected_model: str
    cached: bool
    retries: int
    eliminated_models: List[str] = []
    similarity_score: Optional[float] = None


class ModelScore(BaseModel):
    model_name: str
    overall_score: float
    fine_grained_score: float
    helpfulness: float
    honesty: float
    instruction_following: float
    truthfulness: float
    predicted_tokens: float
    composite_score: Optional[float] = None


class EvaluationResult(BaseModel):
    status: EvaluationStatus = Field(description="'PASSED' or 'REMEDIATED_FALLBACK'")
    is_valid: bool
    evaluation_notes: str
    retry_count: int = 0


class RouteResponse(BaseModel):
    prompt: str
    cached: bool
    selected_model: str
    response_text: str
    routing_reasoning: str
    model_predictions: List[ModelScore] = []
    evaluation: EvaluationResult
    latency_ms: float
    metadata: Dict = {}


class PredictScoresResponse(BaseModel):
    prompt: str
    total_models_evaluated: int
    predictions: List[ModelScore]
    top_recommended_model: str
    strategy_used: str


class HealthStatusResponse(BaseModel):
    status: str
    app_version: str
    evaluator_models_loaded: int
    loaded_models: List[str]
    vector_cache_active: bool
    available_api_keys: List[str]

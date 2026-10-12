import time
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from core.config import settings
from core.dl_loader import get_dl_registry
from services.embedding_service import (
    get_embedding_service,
    generate_query_embedding,
)
from services.semantic_cache import (
    get_semantic_cache,
    search_similar_query,
    save_successful_query,
)
from services.agent_1_router import (
    get_agent_1_router,
    select_best_model,
)
from services.llm_executor_under_review import (
    get_llm_executor,
    execute_query,
)
from services.agent_2_evaluator import (
    get_agent_2_evaluator,
    is_response_acceptable,
)
from models.schemas import (
    PromptRequest,
    RouteResponse,
    AskRequest,
    AskResponse,
    PredictScoresResponse,
    HealthStatusResponse,
    EvaluationResult,
    EvaluationStatus,
    RoutingStrategy,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["LLM Intelligent Router"])


@router.post("/ask", response_model=AskResponse, summary="Execute thesis flowchart: cache lookup, Agent 1 selection, execution, and Agent 2 self-correction loop")
async def ask_endpoint(request: AskRequest):
    """
    Phase 4 Main Flowchart Loop:
    1. Generate embedding and check Semantic Cache. Return if hit.
    2. Initialize an empty eliminated_models list.
    3. Start a while loop (max retries = 3):
       - Call Agent 1 to select the best model (passing eliminated_models).
       - Execute the query.
       - Call Agent 2 to evaluate the response.
       - If correct: Update the successful-query dataset (save prompt, response, embedding to Vector DB), and return response.
       - If incorrect: Record the failure, append selected model to eliminated_models to penalize it, and loop again.
    """
    # 1. Generate query embedding & check Semantic Cache
    prompt_emb = generate_query_embedding(request.prompt)

    if request.use_cache:
        cached_result = search_similar_query(prompt_emb)
        if cached_result is not None:
            payload, similarity = cached_result
            return AskResponse(
                prompt=request.prompt,
                response=payload.get("response_text", ""),
                selected_model=payload.get("selected_model", "semantic_cache"),
                cached=True,
                retries=0,
                eliminated_models=[],
                similarity_score=round(similarity, 4)
            )

    # 2. Initialize eliminated_models list
    eliminated_models: List[str] = []
    retries = 0
    max_retries = request.max_retries or 3

    # 3. Main self-correction loop
    while retries < max_retries:
        # Call Agent 1 to select the best model (excluding eliminated models)
        selected = select_best_model(
            query_embedding=prompt_emb,
            candidate_models=request.candidate_models,
            eliminated_models=eliminated_models
        )

        logger.info(f"[Retry {retries}] Agent 1 selected '{selected}'. Eliminated so far: {eliminated_models}")

        # Execute query
        response_text = await execute_query(prompt=request.prompt, selected_model=selected)

        # Call Agent 2 to evaluate the response
        if is_response_acceptable(response_text):
            # If correct: Save to Vector DB and return
            if request.use_cache:
                save_successful_query(
                    prompt=request.prompt,
                    response=response_text,
                    query_embedding=prompt_emb,
                    selected_model=selected
                )

            return AskResponse(
                prompt=request.prompt,
                response=response_text,
                selected_model=selected,
                cached=False,
                retries=retries,
                eliminated_models=eliminated_models
            )
        else:
            # If incorrect: Penalize and eliminate model, then loop again
            logger.warning(f"Agent 2 rejected response from '{selected}'. Appending to eliminated_models.")
            eliminated_models.append(selected)
            retries += 1

    # Safeguard if all retries fail: fallback to Gemini 2.5 Flash
    fallback_model = settings.FALLBACK_MODEL
    logger.warning(f"Exhausted {retries} retries. Executing fallback model '{fallback_model}'.")
    fallback_text = await execute_query(prompt=request.prompt, selected_model=fallback_model)

    return AskResponse(
        prompt=request.prompt,
        response=fallback_text,
        selected_model=fallback_model,
        cached=False,
        retries=retries,
        eliminated_models=eliminated_models
    )


@router.post("/route", response_model=RouteResponse, summary="Route prompt through dual-agent architecture with detailed scoring")
async def route_prompt(request: PromptRequest):
    """
    Detailed Multi-Head Routing Pipeline:
    1. Embeds prompt using SentenceTransformer.
    2. Queries Qdrant semantic cache for cosine similarity match.
    3. (On cache miss) Agent 1: DL multi-head evaluator predicts scores across all 15 models.
    4. Model selection based on requested optimization strategy.
    5. LLM Executor dispatches generation via LangChain.
    6. Agent 2: Validates output and coordinates retries if necessary.
    7. Caches response into Qdrant semantic cache.
    """
    start_time = time.time()
    embedder = get_embedding_service()
    cache = get_semantic_cache()
    router_agent = get_agent_1_router()
    executor = get_llm_executor()
    evaluator_agent = get_agent_2_evaluator()

    # Step 1: Generate Embedding
    prompt_emb = embedder.embed_text(request.prompt)

    # Step 2: Semantic Cache Lookup
    if request.use_cache:
        cached_result = cache.lookup(prompt_emb)
        if cached_result is not None:
            payload, similarity = cached_result
            latency = round((time.time() - start_time) * 1000, 2)
            return RouteResponse(
                prompt=request.prompt,
                cached=True,
                selected_model=payload.get("selected_model", "cached"),
                response_text=payload.get("response_text", ""),
                routing_reasoning=f"Semantic cache hit with cosine similarity {similarity:.4f}.",
                model_predictions=[],
                evaluation=EvaluationResult(
                    status=EvaluationStatus.PASSED,
                    is_valid=True,
                    evaluation_notes="Retrieved from Qdrant vector semantic cache.",
                    retry_count=0
                ),
                latency_ms=latency,
                metadata={"similarity": similarity, "cache_hit": True}
            )

    # Step 3 & 4: Agent 1 - Multi-Head DL Model Scoring & Selection
    selected_model, ranked_predictions, reasoning = router_agent.route_prompt(
        prompt_embedding=prompt_emb,
        strategy=request.strategy.value if hasattr(request.strategy, "value") else str(request.strategy),
        force_model=request.force_model
    )

    # Step 5 & 6: LLM Execution with Agent 2 Guardrails & Retries
    final_model, response_text, eval_result, exec_metadata = evaluator_agent.execute_with_guardrails(
        prompt=request.prompt,
        ranked_models=ranked_predictions,
        executor_func=executor.execute,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 1024
    )

    # Step 7: Cache Store
    if request.use_cache and eval_result.is_valid:
        cache.store(
            prompt=request.prompt,
            prompt_embedding=prompt_emb,
            response_text=response_text,
            selected_model=final_model,
            metadata={
                "strategy": request.strategy.value if hasattr(request.strategy, "value") else str(request.strategy),
                "retries": eval_result.retry_count
            }
        )

    latency = round((time.time() - start_time) * 1000, 2)

    return RouteResponse(
        prompt=request.prompt,
        cached=False,
        selected_model=final_model,
        response_text=response_text,
        routing_reasoning=reasoning,
        model_predictions=ranked_predictions,
        evaluation=eval_result,
        latency_ms=latency,
        metadata=exec_metadata
    )


@router.post("/predict-scores", response_model=PredictScoresResponse, summary="Predict DL evaluator scores across 15 models")
async def predict_scores(request: PromptRequest):
    """
    Evaluates a prompt against all 15 neural evaluator models without triggering LLM execution.
    Provides predicted metrics (Helpfulness, Honesty, Instruction Following, Truthfulness, Overall, Tokens).
    """
    embedder = get_embedding_service()
    router_agent = get_agent_1_router()

    prompt_emb = embedder.embed_text(request.prompt)
    ranked = router_agent.rank_models(
        prompt_embedding=prompt_emb,
        strategy=request.strategy.value if hasattr(request.strategy, "value") else str(request.strategy)
    )

    top_model = ranked[0].model_name if ranked else "unknown"

    return PredictScoresResponse(
        prompt=request.prompt,
        total_models_evaluated=len(ranked),
        predictions=ranked,
        top_recommended_model=top_model,
        strategy_used=request.strategy.value if hasattr(request.strategy, "value") else str(request.strategy)
    )


@router.get("/models", summary="List all 15 deep learning evaluator models")
async def list_models() -> Dict[str, List[str]]:
    """Returns the names of all models with loaded neural evaluator weights."""
    registry = get_dl_registry()
    if not registry.is_loaded:
        registry.load_all_weights()
    return {"loaded_models": list(registry.models.keys())}


@router.post("/cache/clear", summary="Clear Qdrant semantic vector cache")
async def clear_cache():
    """Resets the Qdrant semantic cache collection."""
    cache = get_semantic_cache()
    success = cache.clear()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear semantic cache or vector DB not connected."
        )
    return {"message": "Semantic cache successfully reset."}


@router.get("/health", response_model=HealthStatusResponse, summary="System health and status")
async def health_check():
    """Returns status of evaluator models, Qdrant cache, and active LLM API keys."""
    registry = get_dl_registry()
    cache = get_semantic_cache()

    available_keys = []
    if settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY:
        available_keys.append("GEMINI_API_KEY")
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
        available_keys.append("OPENAI_API_KEY")
    if settings.GROQ_API_KEY:
        available_keys.append("GROQ_API_KEY")
    if settings.HUGGINGFACE_API_KEY:
        available_keys.append("HUGGINGFACE_API_KEY")

    return HealthStatusResponse(
        status="healthy",
        app_version=settings.APP_VERSION,
        evaluator_models_loaded=len(registry.models),
        loaded_models=list(registry.models.keys()),
        vector_cache_active=cache.client is not None,
        available_api_keys=available_keys
    )

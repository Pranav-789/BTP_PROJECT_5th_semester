import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.dl_loader import get_dl_registry
from services.embedding_service import get_embedding_service
from services.semantic_cache import get_semantic_cache
from api.routes import router as api_router

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan: Pre-loads neural evaluator models and initializes vector cache.
    """
    logger.info("Initializing LLM Router Services...")
    try:
        # 1. Preload SentenceTransformer
        get_embedding_service()
        # 2. Preload Qdrant semantic cache
        get_semantic_cache()
        # 3. Load all 15 neural evaluator model weights
        registry = get_dl_registry()
        registry.load_all_weights()
        logger.info(f"Loaded {len(registry.models)} deep learning evaluator models.")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}")

    yield

    logger.info("Shutting down LLM Router API...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Dual-Agent Deep Learning Router API for dynamic LLM dispatching, "
        "semantic vector caching with Qdrant, multi-task scoring on UltraFeedback weights, "
        "and automated response evaluation with LangChain guardrails."
    ),
    lifespan=lifespan
)

# CORS middleware for frontend / client accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router)


@app.get("/", tags=["Root"])
def root_info():
    return {
        "title": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "endpoints": {
            "ask_flowchart": "/api/v1/ask",
            "route_prompt": "/api/v1/route",
            "predict_scores": "/api/v1/predict-scores",
            "list_models": "/api/v1/models",
            "health": "/api/v1/health",
            "clear_cache": "/api/v1/cache/clear"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

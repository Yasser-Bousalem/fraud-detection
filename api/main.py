import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import Transaction, ScoreResponse, HealthResponse
from api.service import FraudScoringService, MODEL_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# module-level service handle — populated at startup
service: FraudScoringService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE when the app starts, not per request."""
    global service
    logger.info("Starting up — loading model artifacts …")
    service = FraudScoringService()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Fraud Detection Scoring API",
    description="Real-time card fraud scoring with calibrated probabilities and reason codes.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — is the service up and is the model loaded?"""
    return HealthResponse(
        status="ok" if service is not None else "starting",
        model_loaded=service is not None,
        model_version=MODEL_VERSION,
    )


@app.get("/version")
def version():
    """Return model version and operating threshold."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "model_version": MODEL_VERSION,
        "threshold":     service.threshold,
        "n_features":    len(service.feature_cols),
    }


@app.post("/score", response_model=ScoreResponse)
def score(transaction: Transaction):
    """
    Score a single transaction for fraud.
    
    Returns a calibrated fraud probability, a review/approve decision,
    and the top reason codes explaining the score.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        tx_dict = transaction.model_dump()
        result = service.score(tx_dict)
        return ScoreResponse(**result)
    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.get("/")
def root():
    """Redirect hint to the interactive docs."""
    return JSONResponse({
        "service": "Fraud Detection Scoring API",
        "version": MODEL_VERSION,
        "docs":    "/docs",
        "endpoints": ["/health", "/version", "/score"],
    })
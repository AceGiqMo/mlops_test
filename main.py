import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Query
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# --- п.7 Stateless: вся конфигурация через переменные окружения ---
API_KEY: Optional[str] = os.getenv("API_KEY")     # не задан → авторизация выключена (dev)
IS_PROD = os.getenv("ENV", "dev") == "prod"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# --- п.5 Логирование ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("wine_api")

# Настройка MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI", "http://localhost:5000"))

model = None
metadata = None

FEATURE_NAMES = ['Alcohol',
                 'Malic acid', 'Ash',
                 'Alcalinity of ash', 'Magnesium',
                 'Total phenols', 'Flavanoids',
                 'Nonflavanoid phenols',
                 'Proanthocyanins',
                 'Color intensity', 'Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata

    model_name = os.getenv("MODEL_NAME", "wine-model")
    model_version = os.getenv("MODEL_VERSION", "1")

    logger.info("Attempting to load model %s version %s from MLflow...", model_name, model_version)

    client = MlflowClient()

    try:
        # Пытаемся загрузить указанную версию
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        model.predict([[0.0 for _ in range(13)]])  # Прогрев
        logger.info("Successfully loaded model version %s", model_version)

    except Exception as e:
        logger.warning("Failed to load version %s: %s", model_version, e)

        # Пробуем последнюю версию
        try:
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if all_versions:
                latest_version = max(int(v.version) for v in all_versions)
                logger.info("Trying latest version: %d", latest_version)
                model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
                model.predict([[0.0 for _ in range(13)]])
                model_version = str(latest_version)
                logger.info("Successfully loaded latest version %s", latest_version)
            else:
                # Моделей нет вообще — стартуем без модели
                logger.warning("No models found in Registry. API will start without a loaded model.")
                model = None
                metadata = None
                yield
                return
        except Exception as e2:
            logger.error("Failed to load any model: %s", e2)
            model = None
            metadata = None
            yield
            return

    # Загружаем метаданные
    try:
        mv = client.get_model_version(model_name, model_version)
        latest_run = client.get_run(mv.run_id)

        metadata = {
            "model_name": model_name,
            "model_version": int(model_version),
            "run_id": latest_run.info.run_id,
            "feature_names": FEATURE_NAMES,
            "target_names": [str(c) for c in getattr(model, "classes_", [])],
            "trained_at": latest_run.info.start_time,
            "metrics": latest_run.data.metrics,
            "params": latest_run.data.params
        }

        logger.info("Model loaded: %s v%d", metadata['model_name'], metadata['model_version'])
    except Exception as e:
        logger.error("Failed to load metadata: %s", e)
        metadata = None

    yield
    logger.info("Shutdown")


# --- п.7 В проде отключаем интерактивную документацию ---
app = FastAPI(
    title="Wine Model API",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    lifespan=lifespan,
)

# --- п.7 CORS только для доверенных источников (из окружения) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- п.5 Middleware: логируем каждый запрос с латентностью ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - started) * 1000
    logger.info("%s %s -> %s | %.1f ms",
                request.method, request.url.path, response.status_code, ms)
    return response

# --- п.7 Авторизация: активна, только если задан API_KEY ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: Optional[str] = Security(api_key_header)):
    if API_KEY is not None and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# --- Схемы контракта ---
class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=13, max_length=13)

class PredictResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    model_version: str

class BatchRequest(BaseModel):
    features: list[list[float]] = Field(..., min_length=1, max_length=1000)

class BatchResponse(BaseModel):
    predictions: list[int]
    model_version: str

# --- Эндпоинты: обычный def, т.к. инференс — блокирующий CPU-код ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    if metadata is None:
        return {
            "status": "no_model_loaded",
            "message": "API is running but no model is loaded yet. Train a model and call /reload."
        }
    return metadata


from fastapi import Query
from mlflow.tracking import MlflowClient


@app.post("/reload")
def reload_model(
        version: Optional[int] = Query(None, description="Model version number"),
        _=Depends(require_api_key)
):
    global model, metadata

    model_name = os.getenv("MODEL_NAME", "wine-model")
    client = MlflowClient()

    # Если версия не передана — берем последнюю
    if version is None:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            raise HTTPException(
                status_code=404,
                detail="No model versions found in Registry. Train a model first."
            )
        version = max(int(v.version) for v in all_versions)
        logger.info("No version specified, using latest: %d", version)

    logger.info("Reloading model %s version %d...", model_name, version)

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
        model.predict([[0.0 for _ in range(13)]])  # Прогрев
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model version {version}: {str(e)}"
        )

    try:
        mv = client.get_model_version(model_name, version)
        latest_run = client.get_run(mv.run_id)

        metadata = {
            "model_name": model_name,
            "model_version": int(version),
            "run_id": latest_run.info.run_id,
            "feature_names": FEATURE_NAMES,
            "target_names": [str(c) for c in getattr(model, "classes_", [])],
            "trained_at": latest_run.info.start_time,
            "metrics": latest_run.data.metrics,
            "params": latest_run.data.params
        }

        logger.info("Model reloaded successfully: v%d", version)
        return {"status": "reloaded", "model_version": str(version)}
    except Exception as e:
        logger.error("Failed to load metadata: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Model loaded but failed to fetch metadata: {str(e)}"
        )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, _=Depends(require_api_key)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first or call /reload after training."
        )

    prediction = int(model.predict([payload.features])[0])
    probabilities = model.predict_proba([payload.features])[0].tolist()
    logger.info("predict features=%s -> %d", payload.features, prediction)
    return PredictResponse(
        prediction=prediction,
        probabilities=probabilities,
        model_version=str(metadata["model_version"])
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, _=Depends(require_api_key)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first or call /reload after training."
        )

    n = len(metadata["feature_names"]) if metadata else 13
    if any(len(row) != n for row in payload.features):
        raise HTTPException(status_code=422, detail=f"Expected {n} features per row")

    predictions = model.predict(payload.features).tolist()
    logger.info("batch size=%d", len(payload.features))
    return BatchResponse(
        predictions=predictions,
        model_version=str(metadata["model_version"])
    )
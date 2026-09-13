import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# --- п.7 Stateless: вся конфигурация через переменные окружения ---
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
API_KEY: Optional[str] = os.getenv("API_KEY")     # не задан → авторизация выключена (dev)
IS_PROD = os.getenv("ENV", "dev") == "prod"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# --- п.5 Логирование ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("iris_api")

model = None
metadata = None

# --- п.2 Загрузка модели через lifespan + warmup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata
    logger.info("Loading model from %s ...", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    metadata = artifact["metadata"]
    model.predict([[0.0] * len(metadata["feature_names"])])  # прогрев
    logger.info("Model v%s is ready", metadata["model_version"])
    yield
    logger.info("Shutdown")

# --- п.7 В проде отключаем интерактивную документацию ---
app = FastAPI(
    title="Iris Model API",
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
    features: list[float] = Field(..., min_length=4, max_length=4)

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

@app.get("/info")   # п.4 версия и метаданные модели
def info():
    return metadata

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, _=Depends(require_api_key)):
    prediction = int(model.predict([payload.features])[0])
    probabilities = model.predict_proba([payload.features])[0].tolist()
    logger.info("predict features=%s -> %d", payload.features, prediction)
    return PredictResponse(prediction=prediction,
                           probabilities=probabilities,
                           model_version=metadata["model_version"])

@app.post("/predict/batch", response_model=BatchResponse)   # п.3 батчи
def predict_batch(payload: BatchRequest, _=Depends(require_api_key)):
    n = len(metadata["feature_names"])
    if any(len(row) != n for row in payload.features):
        raise HTTPException(status_code=422, detail=f"Expected {n} features per row")
    predictions = model.predict(payload.features).tolist()
    logger.info("batch size=%d", len(payload.features))
    return BatchResponse(predictions=predictions,
                         model_version=metadata["model_version"])
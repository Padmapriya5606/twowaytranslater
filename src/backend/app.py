"""
FastAPI Server for ISL Two-Way Translator
Provides real-time REST & WebSocket APIs for:
- Live Sign Sequence Prediction
- Facial Expression / Non-manual cues
- Bidirectional Translation (Text/Speech <-> ISL Gloss)
- 3D/2D Avatar Keyframe Streaming
- Dictionary & Learning Studio
- Serves Frontend Web UI
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from backend.model_service import ModelService
from avatar.avatar_vocabulary import ISL_VOCAB_DB, get_sign_data

# Initialize FastAPI app
app = FastAPI(
    title="ISL Two-Way Translator API",
    description="Offline-capable Neural Indian Sign Language <-> Speech Translation Platform",
    version="2.0.0"
)

# Enable CORS for local development and client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Service
model_service = ModelService.get_instance()

# Pydantic Request Models
class SignPredictionRequest(BaseModel):
    sequence: List[List[float]]
    region: Optional[str] = "generic"

class TextToGlossRequest(BaseModel):
    text: str
    region: Optional[str] = "generic"

class GlossToTextRequest(BaseModel):
    gloss_tokens: List[str]

class FacePredictionRequest(BaseModel):
    emotion_hint: Optional[str] = None
    landmarks: Optional[List[float]] = None

# API Endpoints
@app.get("/health")
@app.get("/healthz")
@app.get("/api/health")
async def health_check():
    """Health check endpoint for Docker container, load balancers, and cloud monitoring."""
    return {
        "status": "healthy",
        "service": "isl-two-way-translator",
        "models": model_service.models_loaded,
        "vocab_count": len(ISL_VOCAB_DB)
    }

@app.get("/api/status")
async def get_status():
    """Returns system status, loaded neural checkpoints, and active vocab count."""
    return {
        "status": "online",
        "models": model_service.models_loaded,
        "classes_count": len(model_service.idx_to_class),
        "total_avatar_signs": len(ISL_VOCAB_DB),
        "active_region": model_service.dialect_selector.region
    }

@app.post("/api/predict_sign")
async def predict_sign(req: SignPredictionRequest):
    """Predicts ISL sign from a sequence of landmark frames."""
    if not req.sequence:
        raise HTTPException(status_code=400, detail="Empty sequence provided.")
    result = model_service.predict_sign_sequence(req.sequence)
    return result

@app.post("/api/predict_face")
async def predict_face(req: FacePredictionRequest):
    """Predicts facial expression emotion and non-manual grammatical marker."""
    result = model_service.predict_facial_expression(req.emotion_hint)
    return result

@app.post("/api/gloss_to_text")
async def gloss_to_text(req: GlossToTextRequest):
    """Translates a list of ISL gloss tokens into a natural spoken sentence."""
    sentence = model_service.formulate_sentence_from_gloss(req.gloss_tokens)
    return {
        "gloss_tokens": req.gloss_tokens,
        "natural_sentence": sentence
    }

@app.post("/api/text_to_gloss")
async def text_to_gloss(req: TextToGlossRequest):
    """Converts a spoken/typed sentence into ISL gloss tokens + Avatar keyframe animation data."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = model_service.sentence_to_gloss_keyframes(req.text, region=req.region)
    return result

@app.get("/api/sign_keyframes/{word}")
async def get_sign_keyframes(word: str, region: str = "generic"):
    """Fetches keyframe animation data for a single sign or letter."""
    sign_data = get_sign_data(word, region=region)
    return sign_data

@app.get("/api/dictionary")
async def get_dictionary():
    """Returns all available dictionary signs categorized for learning and exploration."""
    return model_service.get_dictionary()

# Static Files & Frontend Serving
FRONTEND_DIR = PROJECT_ROOT / "src" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{catchall:path}")
    async def serve_spa(catchall: str):
        file_path = FRONTEND_DIR / catchall
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print(f"[Server] Starting ISL Two-Way Translator on http://{host}:{port} ...")
    uvicorn.run("backend.app:app", host=host, port=port, reload=False)


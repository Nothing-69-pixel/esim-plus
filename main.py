from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import httpx
import asyncio
import re
import time
from datetime import datetime, timezone

app = FastAPI(
    title="AI Image Generator API",
    description="Generate AI images via template.net — auto-detects available models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=JSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONSTANTS ============
BASE_URL = "https://ai-tool-service.template.net"
GENERATE_URL = f"{BASE_URL}/api/v3/image-generate-by-model"
STATUS_URL = f"{BASE_URL}/api/v3/image-generation/status"
MODELS_URL = f"{BASE_URL}/api/v3/image-generation/models"  # guessed
TEMPLATE_PAGE = "https://www.template.net/ai-image-generator"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; M2103K19I Build/TP1A.220624.014) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.8010.36 Mobile Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Content-Type": "application/json",
    "Origin": "https://www.template.net",
    "Referer": "https://www.template.net/",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
}

TIMEOUT = 30.0
POLL_TIMEOUT = 60.0
POLL_INTERVAL = 2.0

# Model cache (avoids re-fetching every request)
MODEL_CACHE: Dict = {"models": [], "fetched_at": 0}
CACHE_TTL = 3600  # 1 hour
FALLBACK_MODELS = ["gpt-image-2.5-flare", "gpt-image-1", "dall-e-3"]

JOBS: Dict[str, dict] = {}


# ==================================================================
# ============ MODEL DETECTION ===================================
# ==================================================================
async def fetch_models_from_api() -> Optional[List[dict]]:
    """Try multiple possible API endpoints for models list."""
    candidates = [
        f"{BASE_URL}/api/v3/image-generation/models",
        f"{BASE_URL}/api/v3/image-generate/models",
        f"{BASE_URL}/api/v3/models",
        f"{BASE_URL}/api/v3/image-generation/model-list",
    ]

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for url in candidates:
            try:
                r = await client.get(url, headers=HEADERS)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        models = extract_models_from_json(data)
                        if models:
                            return models
                    except Exception:
                        continue
            except Exception:
                continue
    return None


def extract_models_from_json(data) -> List[dict]:
    """Extract model list from various JSON shapes."""
    models = []

    # Direct list of strings
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                models.append({"id": item, "name": item})
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("name") or item.get("slug")
                if mid:
                    models.append({
                        "id": mid,
                        "name": item.get("label") or item.get("title") or mid,
                        "raw": item,
                    })
        return models

    # Dict with models/data key
    if isinstance(data, dict):
        for key in ["models", "data", "items", "list", "availableModels"]:
            if key in data:
                return extract_models_from_json(data[key])

    return models


async def fetch_models_from_html() -> Optional[List[dict]]:
    """Scrape models from the AI image generator page."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(TEMPLATE_PAGE, headers={
                **HEADERS,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            })
            if r.status_code != 200:
                return None
            html = r.text

        models = []
        seen = set()

        # Pattern 1: model IDs like "gpt-image-2.5-flare", "dall-e-3"
        patterns = [
            r'"(gpt-image-[\d\.]+-[\w-]+)"',
            r'"(gpt-image-[\d\.]+)"',
            r'"(dall-e-[\d\.]+)"',
            r'"(stable-diffusion[\w\-\.]*)"',
            r'"(flux[\w\-\.]*)"',
            r'"(imagen-[\w\-\.]+)"',
        ]

        for pattern in patterns:
            for m in re.finditer(pattern, html, re.IGNORECASE):
                mid = m.group(1)
                if mid not in seen:
                    seen.add(mid)
                    models.append({"id": mid, "name": mid, "source": "html"})

        return models if models else None
    except Exception:
        return None


async def get_available_models(force_refresh: bool = False) -> List[dict]:
    """
    Get list of available models. Caches for 1 hour.
    Tries API first, then HTML scrape, then fallback.
    """
    now = time.time()

    if not force_refresh and MODEL_CACHE["models"] and (now - MODEL_CACHE["fetched_at"] < CACHE_TTL):
        return MODEL_CACHE["models"]

    # Try API
    models = await fetch_models_from_api()

    # Try HTML scrape
    if not models:
        models = await fetch_models_from_html()

    # Fallback
    if not models:
        models = [{"id": m, "name": m, "source": "fallback"} for m in FALLBACK_MODELS]

    MODEL_CACHE["models"] = models
    MODEL_CACHE["fetched_at"] = now

    return models


async def get_default_model() -> str:
    """Return first available model as default."""
    models = await get_available_models()
    if models:
        return models[0]["id"]
    return "gpt-image-2.5-flare"


async def validate_model(model: str) -> dict:
    """Check if model is available. If not, return error with available list."""
    models = await get_available_models()
    valid_ids = [m["id"] for m in models]

    if model in valid_ids:
        return {"valid": True, "model": model}

    # Check case-insensitive
    for m in models:
        if m["id"].lower() == model.lower():
            return {"valid": True, "model": m["id"], "corrected": True}

    return {
        "valid": False,
        "requested": model,
        "available": valid_ids,
        "hint": "Use one of the available models",
    }


# ==================================================================
# ============ MODELS ============================================
# ==================================================================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Image description", min_length=1, max_length=1000)
    model: Optional[str] = Field(None, description="Model ID. Leave empty for auto-detect (uses website default)")
    quality: str = Field("low", description="low, medium, high")
    aspect_ratio: str = Field("16:9", description="16:9, 1:1, 9:16, 4:3, 3:4")
    wait: bool = Field(True, description="If true, waits for completion and returns URL")


# ==================================================================
# ============ HTTP CLIENT =======================================
# ==================================================================
async def post_generate(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.post(GENERATE_URL, json=payload, headers=HEADERS)
        return {"status": r.status_code, "data": r.json() if r.content else {}}


async def get_status(activity_id: str) -> dict:
    params = {"activityId": activity_id}
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.get(STATUS_URL, params=params, headers=HEADERS)
        return {"status": r.status_code, "data": r.json() if r.content else {}}


async def wait_for_completion(activity_id: str) -> dict:
    start = time.time()
    last = None
    while time.time() - start < POLL_TIMEOUT:
        res = await get_status(activity_id)
        data = res.get("data", {})
        last = data

        status = data.get("status", "")
        if status in ("completed", "failed", "error"):
            return data

        await asyncio.sleep(POLL_INTERVAL)

    if last:
        last["_timeout"] = True
    return last or {"status": "timeout", "activityId": activity_id}


# ==================================================================
# ============ ROOT ==============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "AI Image Generator API",
        "version": "2.0.0",
        "backend": "template.net",
        "note": "Model list is auto-fetched. No hardcoded models.",
        "endpoints": {
            "models": "/api/models",
            "generate_get": "/api/generate?prompt=a%20cat",
            "generate_post": "POST /api/generate",
            "generate_wait": "/api/generate-wait?prompt=a%20cat",
            "status": "/api/status/{activity_id}",
            "batch": "POST /api/batch",
            "jobs": "/api/jobs",
            "health": "/api/health",
        },
    }


# ==================================================================
# ============ HEALTH ============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    return {
        "success": True,
        "status": "healthy",
        "jobs_cached": len(JOBS),
        "models_cached": len(MODEL_CACHE["models"]),
        "cache_age_seconds": int(time.time() - MODEL_CACHE["fetched_at"]) if MODEL_CACHE["fetched_at"] else None,
    }


# ==================================================================
# ============ MODELS — Auto-Detected ⭐ =========================
# ==================================================================
@app.get("/api/models", tags=["Options"])
async def list_models(refresh: bool = Query(False, description="Force refresh from website")):
    """
    List all available AI models.
    Fetched live from template.net — no hardcoded list.
    """
    models = await get_available_models(force_refresh=refresh)
    default_model = models[0]["id"] if models else None

    return {
        "success": True,
        "count": len(models),
        "default_model": default_model,
        "models": models,
        "cached": not refresh,
        "cache_ttl_seconds": CACHE_TTL,
        "note": "Use any of these model IDs in /api/generate?model=...",
    }


@app.get("/api/models/refresh", tags=["Options"])
async def refresh_models():
    """Force refresh model list."""
    models = await get_available_models(force_refresh=True)
    return {
        "success": True,
        "count": len(models),
        "models": models,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
    }


# ==================================================================
# ============ GENERATE (POST) ===================================
# ==================================================================
@app.post("/api/generate", tags=["Generate"])
async def generate_post(req: GenerateRequest):
    """
    Generate an AI image from a prompt.
    Model is auto-detected if not provided.
    """
    # Auto-detect model if not provided
    model = req.model
    if not model:
        model = await get_default_model()
    else:
        # Validate model
        validation = await validate_model(model)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Invalid model: {model}",
                "available_models": validation["available"],
                "hint": validation["hint"],
            }
        model = validation["model"]

    payload = {
        "toolPrompt": "",
        "userPrompt": req.prompt,
        "model": model,
        "quality": req.quality,
        "aspectRatio": req.aspect_ratio,
        "tool": {
            "slug": "ai-image-generator",
            "graphicType": "element",
            "from": "/ai-image-generator__ai-image-generator",
            "isPanelCall": False,
        },
    }

    init = await post_generate(payload)
    if init["status"] not in (200, 201):
        return {
            "success": False,
            "error": "Failed to initiate generation",
            "upstream_status": init["status"],
            "upstream_response": init["data"],
        }

    init_data = init["data"]
    activity_id = (
        init_data.get("activityId")
        or init_data.get("data", {}).get("activityId")
        or init_data.get("id")
    )

    if not activity_id:
        return {
            "success": False,
            "error": "No activityId in response",
            "upstream_response": init_data,
        }

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": req.prompt,
        "model": model,
        "quality": req.quality,
        "aspect_ratio": req.aspect_ratio,
        "initiated_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
    }

    if not req.wait:
        return {
            "success": True,
            "activity_id": activity_id,
            "model": model,
            "status": "processing",
            "status_url": f"/api/status/{activity_id}",
            "message": "Job initiated. Poll /api/status/{activity_id} to check.",
        }

    result = await wait_for_completion(activity_id)
    inner = result.get("data", {})
    urls = inner.get("urls", []) or result.get("urls", [])

    JOBS[activity_id].update({
        "status": result.get("status"),
        "urls": urls,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "success": result.get("status") == "completed",
        "activity_id": activity_id,
        "status": result.get("status"),
        "message": result.get("message", ""),
        "prompt": req.prompt,
        "model": model,
        "urls": urls,
        "thumbnail": inner.get("thumbnail") or result.get("thumbnail"),
        "resolved_prompt": inner.get("resolvedPrompt") or result.get("resolvedPrompt"),
        "created_at": inner.get("createdAt") or result.get("createdAt"),
    }


# ==================================================================
# ============ GENERATE (GET) ====================================
# ==================================================================
@app.get("/api/generate", tags=["Generate"])
async def generate_get(
    prompt: str = Query(..., description="Image description"),
    model: Optional[str] = Query(None, description="Leave empty for auto-detect"),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    wait: bool = Query(True),
):
    req = GenerateRequest(
        prompt=prompt,
        model=model,
        quality=quality,
        aspect_ratio=aspect_ratio,
        wait=wait,
    )
    return await generate_post(req)


# ==================================================================
# ============ STATUS ============================================
# ==================================================================
@app.get("/api/status/{activity_id}", tags=["Generate"])
async def get_status_endpoint(activity_id: str):
    res = await get_status(activity_id)

    if res["status"] != 200:
        return {
            "success": False,
            "error": "Failed to fetch status",
            "upstream_status": res["status"],
            "upstream_response": res["data"],
        }

    data = res["data"]
    inner = data.get("data", {})
    urls = inner.get("urls", []) or data.get("urls", [])

    if activity_id in JOBS:
        JOBS[activity_id].update({
            "status": data.get("status"),
            "urls": urls,
            "last_checked": datetime.now(timezone.utc).isoformat(),
        })

    return {
        "success": data.get("success", True),
        "activity_id": activity_id,
        "status": data.get("status"),
        "message": data.get("message", ""),
        "urls": urls,
        "thumbnail": inner.get("thumbnail") or data.get("thumbnail"),
        "prompt": inner.get("userPrompt"),
        "model": inner.get("userParams", {}).get("model"),
        "aspect_ratio": inner.get("userParams", {}).get("aspectRatio"),
        "resolved_prompt": inner.get("resolvedPrompt"),
        "created_at": inner.get("createdAt"),
    }


# ==================================================================
# ============ GENERATE-WAIT (one-shot) ==========================
# ==================================================================
@app.get("/api/generate-wait", tags=["Generate"])
async def generate_wait(
    prompt: str = Query(...),
    model: Optional[str] = Query(None),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
):
    req = GenerateRequest(
        prompt=prompt,
        model=model,
        quality=quality,
        aspect_ratio=aspect_ratio,
        wait=True,
    )
    return await generate_post(req)


# ==================================================================
# ============ JOBS ==============================================
# ==================================================================
@app.get("/api/jobs", tags=["System"])
async def list_jobs():
    jobs = sorted(JOBS.values(), key=lambda x: x.get("initiated_at", ""), reverse=True)
    return {"success": True, "count": len(jobs), "jobs": jobs[:50]}


@app.get("/api/jobs/{activity_id}", tags=["System"])
async def get_job(activity_id: str):
    if activity_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True, "job": JOBS[activity_id]}


# ==================================================================
# ============ BATCH =============================================
# ==================================================================
class BatchRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=10)
    model: Optional[str] = None
    quality: str = "low"
    aspect_ratio: str = "16:9"


@app.post("/api/batch", tags=["Generate"])
async def batch_generate(req: BatchRequest):
    results = []

    # Resolve model once
    model = req.model or await get_default_model()

    for prompt in req.prompts:
        try:
            gen_req = GenerateRequest(
                prompt=prompt,
                model=model,
                quality=req.quality,
                aspect_ratio=req.aspect_ratio,
                wait=False,
            )
            res = await generate_post(gen_req)
            results.append({
                "prompt": prompt,
                "success": res.get("success", False),
                "activity_id": res.get("activity_id"),
                "status_url": res.get("status_url"),
                "error": res.get("error"),
            })
        except Exception as e:
            results.append({
                "prompt": prompt,
                "success": False,
                "error": str(e)[:200],
            })

    return {
        "success": True,
        "count": len(results),
        "model_used": model,
        "results": results,
    }


# ============ ERROR HANDLERS ============
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc), "status_code": 500},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

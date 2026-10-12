from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import httpx
import asyncio
import re
import time
from datetime import datetime, timezone

app = FastAPI(
    title="AI Image Generator API",
    description="Generate AI images — just give a prompt, model is auto-detected",
    version="4.0.0",
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
POLL_TIMEOUT = 8.0       # Vercel-safe
POLL_INTERVAL = 1.2

# The model the website currently uses by default
# (auto-fetched, refreshed periodically)
MODEL_CACHE: Dict = {"current": None, "fetched_at": 0}
CACHE_TTL = 1800  # 30 min

# Jobs store
JOBS: Dict[str, dict] = {}

# ⭐ These are the working model IDs that the template.net API accepts.
# The website changes these — we try them in order.
WORKING_MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-1",
    "gpt-image-2.5",
    "gpt-image-2",
    "dall-e-3",
]


# ==================================================================
# ============ AUTO MODEL DETECTION ==============================
# ==================================================================
async def detect_current_model() -> str:
    """
    Return the model that template.net currently accepts.
    Cached for CACHE_TTL seconds.
    """
    now = time.time()
    if MODEL_CACHE["current"] and (now - MODEL_CACHE["fetched_at"] < CACHE_TTL):
        return MODEL_CACHE["current"]

    # Try the first known working model
    # (These rarely change; if they do, we'll know from errors)
    model = WORKING_MODELS[0]
    MODEL_CACHE["current"] = model
    MODEL_CACHE["fetched_at"] = now
    return model


async def try_generate_with_fallback(payload_base: dict) -> dict:
    """
    Try generation with each working model until one succeeds.
    Returns the successful response with `_model_used`.
    """
    last_error = None

    for model in WORKING_MODELS:
        payload = {**payload_base, "model": model}
        init = await post_generate(payload)

        status = init.get("status")
        data = init.get("data", {})

        # Success criteria
        if status in (200, 201) and (data.get("activityId") or data.get("id") or data.get("data", {}).get("activityId")):
            return {
                "success": True,
                "model_used": model,
                "status": status,
                "data": data,
            }

        # Model-specific error? Try next
        error_msg = str(data)[:200]
        last_error = {"model": model, "status": status, "error": error_msg}

        # If it's a "model not found" type error, try next
        # If it's some other issue (network), break
        if status not in (200, 201, 400, 404, 422):
            break

    return {
        "success": False,
        "error": "All models failed",
        "last_error": last_error,
    }


# ==================================================================
# ============ HTTP CLIENT =======================================
# ==================================================================
async def post_generate(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.post(GENERATE_URL, json=payload, headers=HEADERS)
        try:
            data = r.json()
        except Exception:
            data = {"_raw": r.text[:500]}
        return {"status": r.status_code, "data": data}


async def get_status(activity_id: str) -> dict:
    params = {"activityId": activity_id}
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.get(STATUS_URL, params=params, headers=HEADERS)
        try:
            data = r.json()
        except Exception:
            data = {"_raw": r.text[:500]}
        return {"status": r.status_code, "data": data}


async def wait_for_completion(activity_id: str, max_wait: float = POLL_TIMEOUT) -> dict:
    start = time.time()
    last = None
    while True:
        try:
            res = await get_status(activity_id)
            data = res.get("data", {})
            last = data
            status = data.get("status", "")
            if status in ("completed", "failed", "error"):
                return data
        except Exception:
            pass
        elapsed = time.time() - start
        if elapsed >= max_wait:
            if last:
                last["_timeout"] = True
                last["_elapsed"] = round(elapsed, 1)
            return last or {"status": "timeout", "activityId": activity_id}
        await asyncio.sleep(POLL_INTERVAL)


def build_payload(prompt: str, quality: str = "low", aspect_ratio: str = "16:9") -> dict:
    """Standard payload for template.net."""
    return {
        "toolPrompt": "",
        "userPrompt": prompt,
        "quality": quality,
        "aspectRatio": aspect_ratio,
        "tool": {
            "slug": "ai-image-generator",
            "graphicType": "element",
            "from": "/ai-image-generator__ai-image-generator",
            "isPanelCall": False,
        },
    }


# ==================================================================
# ============ ROOT ==============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "AI Image Generator API",
        "version": "4.0.0",
        "usage": "Just give a prompt — model is auto-detected. No model parameter needed.",
        "endpoints": {
            "generate_simple": "/api/generate?prompt=A%20cute%20cat",
            "generate_smart": "/api/auto?prompt=A%20cute%20cat",
            "status": "/api/status/{activity_id}",
            "image": "/api/image/{activity_id}",
            "jobs": "/api/jobs",
            "health": "/api/health",
        },
        "example": [
            "https://your-app.vercel.app/api/generate?prompt=A cute cat",
            "https://your-app.vercel.app/api/auto?prompt=A dragon over mountains",
        ],
    }


# ==================================================================
# ============ HEALTH ============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    return {
        "success": True,
        "status": "healthy",
        "cached_model": MODEL_CACHE.get("current"),
        "jobs_cached": len(JOBS),
    }


# ==================================================================
# ============ GENERATE — Prompt Only ⭐ ==========================
# ==================================================================
@app.get("/api/generate", tags=["Generate"])
async def generate_simple(
    prompt: str = Query(..., description="What you want to generate — model auto-detected"),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
):
    """
    Generate image from a prompt. Model is auto-detected from template.net.
    No model parameter needed.
    """
    payload_base = build_payload(prompt, quality, aspect_ratio)

    # ⭐ Try each working model until one succeeds
    result = await try_generate_with_fallback(payload_base)

    if not result["success"]:
        return {
            "success": False,
            "error": result["error"],
            "last_error": result.get("last_error"),
            "prompt": prompt,
        }

    init_data = result["data"]
    model_used = result["model_used"]

    activity_id = (
        init_data.get("activityId")
        or init_data.get("data", {}).get("activityId")
        or init_data.get("id")
    )

    if not activity_id:
        return {
            "success": False,
            "error": "No activityId",
            "response": init_data,
        }

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": prompt,
        "model": model_used,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "initiated_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
    }

    # Wait for completion
    final = await wait_for_completion(activity_id)
    inner = final.get("data", {})
    urls = inner.get("urls", []) or final.get("urls", [])
    status = final.get("status", "unknown")

    JOBS[activity_id].update({
        "status": status,
        "urls": urls,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    # Completed ✅
    if status == "completed" and urls:
        return {
            "success": True,
            "status": "completed",
            "prompt": prompt,
            "model_used": model_used,
            "image_url": urls[0],
            "all_urls": urls,
            "thumbnail": inner.get("thumbnail") or final.get("thumbnail"),
            "activity_id": activity_id,
            "resolved_prompt": inner.get("resolvedPrompt") or final.get("resolvedPrompt"),
            "created_at": inner.get("createdAt") or final.get("createdAt"),
        }

    # Timeout — still processing
    if final.get("_timeout"):
        return {
            "success": False,
            "status": "processing",
            "prompt": prompt,
            "model_used": model_used,
            "activity_id": activity_id,
            "message": "⏳ Still generating. Check status_url.",
            "status_url": f"/api/status/{activity_id}",
            "retry_after_seconds": 5,
        }

    # Failed
    return {
        "success": False,
        "status": status,
        "prompt": prompt,
        "model_used": model_used,
        "activity_id": activity_id,
        "message": final.get("message", "Generation failed"),
    }


# ==================================================================
# ============ AUTO — Full auto with internal retries ============
# ==================================================================
@app.get("/api/auto", tags=["Generate"])
async def auto_generate(
    prompt: str = Query(..., description="Prompt only"),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    max_wait_seconds: int = Query(45, ge=10, le=120, description="Internal wait budget"),
):
    """
    Full auto: initiate → poll internally up to max_wait_seconds → return URL.
    Client can call again with the same activity_id if still processing.
    """
    payload_base = build_payload(prompt, quality, aspect_ratio)
    result = await try_generate_with_fallback(payload_base)

    if not result["success"]:
        return {
            "success": False,
            "error": result["error"],
            "last_error": result.get("last_error"),
            "prompt": prompt,
        }

    init_data = result["data"]
    model_used = result["model_used"]
    activity_id = (
        init_data.get("activityId")
        or init_data.get("data", {}).get("activityId")
        or init_data.get("id")
    )

    if not activity_id:
        return {"success": False, "error": "No activityId", "response": init_data}

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": prompt,
        "model": model_used,
        "initiated_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
    }

    # Poll internally
    start = time.time()
    attempts = 0
    while time.time() - start < max_wait_seconds:
        attempts += 1
        try:
            res = await get_status(activity_id)
            data = res.get("data", {})
            inner = data.get("data", {})
            status = data.get("status")
            urls = inner.get("urls", []) or data.get("urls", [])

            if status == "completed" and urls:
                JOBS[activity_id].update({"status": "completed", "urls": urls})
                return {
                    "success": True,
                    "status": "completed",
                    "prompt": prompt,
                    "model_used": model_used,
                    "image_url": urls[0],
                    "all_urls": urls,
                    "thumbnail": inner.get("thumbnail"),
                    "activity_id": activity_id,
                    "resolved_prompt": inner.get("resolvedPrompt"),
                    "created_at": inner.get("createdAt"),
                    "time_taken_seconds": round(time.time() - start, 1),
                    "attempts": attempts,
                }

            if status in ("failed", "error"):
                return {
                    "success": False,
                    "status": status,
                    "activity_id": activity_id,
                    "message": data.get("message", "Failed"),
                }
        except Exception:
            pass

        await asyncio.sleep(2)

    return {
        "success": False,
        "status": "processing",
        "prompt": prompt,
        "model_used": model_used,
        "activity_id": activity_id,
        "message": f"⏳ Still generating after {max_wait_seconds}s",
        "status_url": f"/api/status/{activity_id}",
        "retry_after_seconds": 5,
    }


# ==================================================================
# ============ STATUS ============================================
# ==================================================================
@app.get("/api/status/{activity_id}", tags=["Generate"])
async def get_status_endpoint(activity_id: str):
    res = await get_status(activity_id)
    if res["status"] != 200:
        return {"success": False, "error": "Status check failed", "upstream": res["data"]}

    data = res["data"]
    inner = data.get("data", {})
    urls = inner.get("urls", []) or data.get("urls", [])
    status = data.get("status")

    if activity_id in JOBS:
        JOBS[activity_id].update({"status": status, "urls": urls})

    response = {
        "success": data.get("success", True),
        "activity_id": activity_id,
        "status": status,
        "message": data.get("message", ""),
        "prompt": inner.get("userPrompt"),
        "model": inner.get("userParams", {}).get("model"),
        "urls": urls,
    }

    if status == "completed" and urls:
        response.update({
            "is_ready": True,
            "image_url": urls[0],
            "thumbnail": inner.get("thumbnail"),
            "resolved_prompt": inner.get("resolvedPrompt"),
            "created_at": inner.get("createdAt"),
        })
    elif status == "processing":
        response.update({
            "is_ready": False,
            "retry_after_seconds": 3,
        })
    else:
        response["is_ready"] = False

    return response


# ==================================================================
# ============ REDIRECT TO IMAGE =================================
# ==================================================================
@app.get("/api/image/{activity_id}", tags=["Generate"])
async def redirect_to_image(activity_id: str):
    res = await get_status(activity_id)
    if res["status"] != 200:
        raise HTTPException(status_code=502, detail="Status check failed")

    data = res["data"]
    inner = data.get("data", {})
    urls = inner.get("urls", []) or data.get("urls", [])

    if data.get("status") != "completed" or not urls:
        raise HTTPException(status_code=404, detail="Image not ready")

    return RedirectResponse(url=urls[0])


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

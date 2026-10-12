from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import httpx
import asyncio
import time
from datetime import datetime, timezone

app = FastAPI(
    title="AI Image Generator API",
    description="Generate AI images — just give a prompt, get the image URL",
    version="5.0.0",
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

# Working models — auto try until one succeeds
WORKING_MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-1",
    "gpt-image-2.5",
    "gpt-image-2",
    "dall-e-3",
]

# Jobs cache
JOBS: Dict[str, dict] = {}


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


def build_payload(prompt: str, quality: str = "low", aspect_ratio: str = "16:9") -> dict:
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


async def try_generate(payload_base: dict) -> dict:
    """Try each model until one succeeds."""
    last_error = None
    for model in WORKING_MODELS:
        payload = {**payload_base, "model": model}
        init = await post_generate(payload)
        status = init.get("status")
        data = init.get("data", {})

        activity_id = (
            data.get("activityId")
            or data.get("data", {}).get("activityId")
            or data.get("id")
        )

        if status in (200, 201) and activity_id:
            return {"success": True, "model_used": model, "activity_id": activity_id, "data": data}

        last_error = {"model": model, "status": status, "error": str(data)[:200]}

    return {"success": False, "last_error": last_error}


async def poll_until_complete(activity_id: str, max_seconds: int = 90) -> dict:
    """
    Poll until image is ready or max_seconds exceeded.
    Uses fast polling (1s intervals) for quick response.
    """
    start = time.time()
    last_data = None
    attempt = 0

    while time.time() - start < max_seconds:
        attempt += 1
        try:
            res = await get_status(activity_id)
            data = res.get("data", {})
            last_data = data

            status = data.get("status")
            inner = data.get("data", {})
            urls = inner.get("urls", []) or data.get("urls", [])

            if status == "completed" and urls:
                return {
                    "success": True,
                    "status": "completed",
                    "image_url": urls[0],
                    "all_urls": urls,
                    "thumbnail": inner.get("thumbnail") or data.get("thumbnail"),
                    "resolved_prompt": inner.get("resolvedPrompt") or data.get("resolvedPrompt"),
                    "created_at": inner.get("createdAt") or data.get("createdAt"),
                    "attempts": attempt,
                    "time_taken": round(time.time() - start, 1),
                }

            if status in ("failed", "error"):
                return {
                    "success": False,
                    "status": status,
                    "message": data.get("message", "Generation failed"),
                    "attempts": attempt,
                }
        except Exception:
            pass

        await asyncio.sleep(1.0)

    return {
        "success": False,
        "status": "timeout",
        "message": f"Image not ready after {max_seconds}s",
        "attempts": attempt,
    }


# ==================================================================
# ============ ROOT ==============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "AI Image Generator API",
        "version": "5.0.0",
        "usage": "Give a prompt — get the image URL. No processing messages.",
        "endpoints": {
            "generate": "/api/generate?prompt=A%20cute%20cat",
            "image": "/api/image?prompt=A%20cute%20cat",
            "jobs": "/api/jobs",
            "health": "/api/health",
        },
    }


# ==================================================================
# ============ HEALTH ============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    return {"success": True, "status": "healthy", "jobs": len(JOBS)}


# ==================================================================
# ============ GENERATE — FULL WAIT ⭐ ============================
# ==================================================================
@app.get("/api/generate", tags=["Generate"])
async def generate(
    prompt: str = Query(..., description="What to generate"),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    max_wait: int = Query(90, ge=10, le=120, description="Max seconds to wait"),
):
    """
    Generate image — waits until complete, then returns image URL.
    No processing response — only final image or error.
    """
    # 1) Initiate
    payload_base = build_payload(prompt, quality, aspect_ratio)
    result = await try_generate(payload_base)

    if not result["success"]:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to initiate. Last error: {result.get('last_error')}",
        )

    activity_id = result["activity_id"]
    model_used = result["model_used"]

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": prompt,
        "model": model_used,
        "initiated_at": datetime.now(timezone.utc).isoformat(),
        "status": "processing",
    }

    # 2) Wait until complete (up to max_wait seconds)
    final = await poll_until_complete(activity_id, max_seconds=max_wait)

    if final["success"]:
        JOBS[activity_id].update({
            "status": "completed",
            "image_url": final["image_url"],
        })
        return {
            "success": True,
            "prompt": prompt,
            "model_used": model_used,
            "image_url": final["image_url"],
            "all_urls": final["all_urls"],
            "thumbnail": final.get("thumbnail"),
            "resolved_prompt": final.get("resolved_prompt"),
            "created_at": final.get("created_at"),
            "activity_id": activity_id,
            "time_taken_seconds": final.get("time_taken"),
        }

    # Failed or timeout
    JOBS[activity_id].update({"status": final["status"]})
    raise HTTPException(
        status_code=500,
        detail={
            "message": final.get("message", "Generation failed"),
            "activity_id": activity_id,
            "status": final["status"],
            "attempts": final.get("attempts"),
        },
    )


# ==================================================================
# ============ SHORTCUT — /api/image?prompt=... ==================
# ==================================================================
@app.get("/api/image", tags=["Generate"])
async def image_shortcut(
    prompt: str = Query(...),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    max_wait: int = Query(90, ge=10, le=120),
):
    """Alias for /api/generate."""
    return await generate(prompt=prompt, quality=quality, aspect_ratio=aspect_ratio, max_wait=max_wait)


# ==================================================================
# ============ REDIRECT — direct image view =======================
# ==================================================================
@app.get("/api/view", tags=["Generate"])
async def view_image(
    prompt: str = Query(...),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    max_wait: int = Query(90, ge=10, le=120),
):
    """
    Generate and redirect directly to the image (for <img src="...">).
    """
    try:
        res = await generate(prompt=prompt, quality=quality, aspect_ratio=aspect_ratio, max_wait=max_wait)
        return RedirectResponse(url=res["image_url"])
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "error": e.detail})


# ==================================================================
# ============ POST version ======================================
# ==================================================================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    quality: str = Field("low")
    aspect_ratio: str = Field("16:9")
    max_wait: int = Field(90, ge=10, le=120)


@app.post("/api/generate", tags=["Generate"])
async def generate_post(req: GenerateRequest):
    return await generate(
        prompt=req.prompt,
        quality=req.quality,
        aspect_ratio=req.aspect_ratio,
        max_wait=req.max_wait,
    )


# ==================================================================
# ============ STATUS (for re-check) =============================
# ==================================================================
@app.get("/api/status/{activity_id}", tags=["Generate"])
async def check_status(activity_id: str):
    """Check status of an existing job."""
    res = await get_status(activity_id)
    if res["status"] != 200:
        raise HTTPException(status_code=502, detail="Status check failed")

    data = res["data"]
    inner = data.get("data", {})
    urls = inner.get("urls", []) or data.get("urls", [])
    status = data.get("status")

    if status == "completed" and urls:
        return {
            "success": True,
            "status": "completed",
            "image_url": urls[0],
            "all_urls": urls,
            "prompt": inner.get("userPrompt"),
            "model": inner.get("userParams", {}).get("model"),
        }

    return {
        "success": True,
        "status": status,
        "prompt": inner.get("userPrompt"),
        "model": inner.get("userParams", {}).get("model"),
        "urls": urls,
    }


# ==================================================================
# ============ JOBS ==============================================
# ==================================================================
@app.get("/api/jobs", tags=["System"])
async def list_jobs():
    jobs = sorted(JOBS.values(), key=lambda x: x.get("initiated_at", ""), reverse=True)
    return {"success": True, "count": len(jobs), "jobs": jobs[:50]}


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

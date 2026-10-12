from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import httpx
import asyncio
import time
from datetime import datetime, timezone

app = FastAPI(
    title="AI Image Generator & Editor API",
    description="Generate or edit AI images — powered by template.net",
    version="6.0.0",
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

TIMEOUT = 60.0
POLL_INTERVAL = 1.0

# Working models (auto-try)
WORKING_MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-1",
    "gpt-image-2.5",
    "gpt-image-2",
    "dall-e-3",
]

# Tool configurations
TOOLS = {
    "generate": {
        "slug": "ai-image-generator",
        "from": "/ai-image-generator__ai-image-generator",
        "graphicType": "element",
    },
    "edit": {
        "slug": "ai-photo-editor",
        "from": "/ai-photo-editor__ai-photo-editor",
        "graphicType": "element",
    },
}

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


async def try_generate(payload_base: dict, models: List[str] = None) -> dict:
    """Try each model until one succeeds."""
    models = models or WORKING_MODELS
    last_error = None
    for model in models:
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
    """Poll until image ready or timeout."""
    start = time.time()
    attempt = 0
    while time.time() - start < max_seconds:
        attempt += 1
        try:
            res = await get_status(activity_id)
            data = res.get("data", {})
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
                    "message": data.get("message", "Failed"),
                    "attempts": attempt,
                }
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)

    return {"success": False, "status": "timeout", "message": f"Timeout after {max_seconds}s", "attempts": attempt}


# ==================================================================
# ============ PAYLOAD BUILDERS ==================================
# ==================================================================
def build_generate_payload(prompt: str, quality: str = "low", aspect_ratio: str = "16:9") -> dict:
    """Text-to-image generation."""
    return {
        "toolPrompt": "",
        "userPrompt": prompt,
        "quality": quality,
        "aspectRatio": aspect_ratio,
        "tool": {**TOOLS["generate"], "isPanelCall": False},
    }


def build_edit_payload(
    user_prompt: str,
    tool_prompt: str = "Edit the attached photo",
    quality: str = "low",
) -> dict:
    """Photo editing (image + prompt)."""
    return {
        "toolPrompt": tool_prompt,
        "userPrompt": user_prompt,
        "quality": quality,
        "tool": {**TOOLS["edit"], "isPanelCall": False},
    }


# ==================================================================
# ============ ROOT ==============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "AI Image Generator & Editor API",
        "version": "6.0.0",
        "endpoints": {
            "generate": "/api/generate?prompt=A%20cute%20cat",
            "edit": "/api/edit?prompt=Make%20it%20brighter",
            "edit_with_image": "POST /api/edit-upload (multipart with image)",
            "status": "/api/status/{activity_id}",
            "jobs": "/api/jobs",
            "health": "/api/health",
        },
    }


@app.get("/api/health", tags=["System"])
async def health():
    return {"success": True, "status": "healthy", "jobs": len(JOBS)}


# ==================================================================
# ============ GENERATE — text to image ==========================
# ==================================================================
@app.get("/api/generate", tags=["Generate"])
async def generate(
    prompt: str = Query(...),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    max_wait: int = Query(90, ge=10, le=120),
):
    """Generate image from text prompt. Waits until complete."""
    payload_base = build_generate_payload(prompt, quality, aspect_ratio)
    result = await try_generate(payload_base)

    if not result["success"]:
        raise HTTPException(502, f"Initiate failed: {result.get('last_error')}")

    activity_id = result["activity_id"]
    model_used = result["model_used"]

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": prompt,
        "model": model_used,
        "tool": "generate",
        "initiated_at": datetime.now(timezone.utc).isoformat(),
    }

    final = await poll_until_complete(activity_id, max_seconds=max_wait)

    if final["success"]:
        JOBS[activity_id].update({"status": "completed", "image_url": final["image_url"]})
        return {
            "success": True,
            "tool": "generate",
            "prompt": prompt,
            "model_used": model_used,
            "image_url": final["image_url"],
            "all_urls": final["all_urls"],
            "thumbnail": final.get("thumbnail"),
            "resolved_prompt": final.get("resolved_prompt"),
            "activity_id": activity_id,
            "time_taken_seconds": final.get("time_taken"),
        }

    JOBS[activity_id].update({"status": final["status"]})
    raise HTTPException(500, final.get("message", "Failed"))


# ==================================================================
# ============ EDIT — text-only (no image upload) ================
# ==================================================================
@app.get("/api/edit", tags=["Edit"])
async def edit(
    prompt: str = Query(..., description="Edit instructions e.g. 'Make it brighter'"),
    tool_prompt: str = Query("Edit the attached photo", alias="toolPrompt"),
    quality: str = Query("low"),
    max_wait: int = Query(90, ge=10, le=120),
):
    """
    Edit image using prompt only.
    NOTE: template.net usually needs an uploaded image for photo editor.
    If no image is provided, this may fail — use /api/edit-upload instead.
    """
    payload_base = build_edit_payload(prompt, tool_prompt, quality)
    result = await try_generate(payload_base)

    if not result["success"]:
        raise HTTPException(502, {
            "message": "Initiate failed. Photo editor needs an uploaded image.",
            "last_error": result.get("last_error"),
            "hint": "Use POST /api/edit-upload with multipart form-data (image + prompt)",
        })

    activity_id = result["activity_id"]
    model_used = result["model_used"]

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": prompt,
        "tool_prompt": tool_prompt,
        "model": model_used,
        "tool": "edit",
        "initiated_at": datetime.now(timezone.utc).isoformat(),
    }

    final = await poll_until_complete(activity_id, max_seconds=max_wait)

    if final["success"]:
        JOBS[activity_id].update({"status": "completed", "image_url": final["image_url"]})
        return {
            "success": True,
            "tool": "edit",
            "prompt": prompt,
            "model_used": model_used,
            "image_url": final["image_url"],
            "all_urls": final["all_urls"],
            "activity_id": activity_id,
            "time_taken_seconds": final.get("time_taken"),
        }

    JOBS[activity_id].update({"status": final["status"]})
    raise HTTPException(500, final.get("message", "Failed"))


# ==================================================================
# ============ EDIT with IMAGE UPLOAD ⭐ =========================
# ==================================================================
@app.post("/api/edit-upload", tags=["Edit"])
async def edit_upload(
    prompt: str = Form(..., description="Edit instructions"),
    tool_prompt: str = Form("Edit the attached photo"),
    quality: str = Form("low"),
    max_wait: int = Form(90),
    image: UploadFile = File(..., description="Image to edit"),
):
    """
    Upload an image + prompt → edit it.
    Uses multipart/form-data.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image")

    # Try both approaches — some APIs need JSON w/ image URL, some need multipart
    # First try: JSON with base64 image
    import base64
    b64 = base64.b64encode(image_bytes).decode()

    payload = {
        **build_edit_payload(prompt, tool_prompt, quality),
        "imageBase64": b64,
        "imageName": image.filename or "upload.png",
    }

    # Try each model
    result = await try_generate(payload)

    if not result["success"]:
        # Try multipart form upload
        return await try_multipart_upload(image_bytes, image.filename, prompt, tool_prompt, quality, max_wait)

    activity_id = result["activity_id"]
    model_used = result["model_used"]

    JOBS[activity_id] = {
        "activity_id": activity_id,
        "prompt": prompt,
        "model": model_used,
        "tool": "edit",
        "initiated_at": datetime.now(timezone.utc).isoformat(),
    }

    final = await poll_until_complete(activity_id, max_seconds=max_wait)

    if final["success"]:
        return {
            "success": True,
            "tool": "edit",
            "prompt": prompt,
            "model_used": model_used,
            "image_url": final["image_url"],
            "all_urls": final["all_urls"],
            "activity_id": activity_id,
            "time_taken_seconds": final.get("time_taken"),
            "method": "json_base64",
        }

    raise HTTPException(500, final.get("message", "Failed"))


async def try_multipart_upload(
    image_bytes: bytes, filename: str, prompt: str,
    tool_prompt: str, quality: str, max_wait: int,
) -> dict:
    """Fallback: multipart form-data upload."""
    files = {"image": (filename or "upload.png", image_bytes, "image/png")}
    data = {
        "toolPrompt": tool_prompt,
        "userPrompt": prompt,
        "quality": quality,
        "model": WORKING_MODELS[0],
    }

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        r = await client.post(
            GENERATE_URL,
            files=files,
            data=data,
            headers={k: v for k, v in HEADERS.items() if k.lower() != "content-type"},
        )
        try:
            resp = r.json()
        except Exception:
            resp = {"_raw": r.text[:500]}

    activity_id = (
        resp.get("activityId")
        or resp.get("data", {}).get("activityId")
        or resp.get("id")
    )

    if not activity_id:
        raise HTTPException(500, {
            "message": "Both JSON and multipart uploads failed",
            "response": resp,
        })

    final = await poll_until_complete(activity_id, max_seconds=max_wait)

    if final["success"]:
        return {
            "success": True,
            "tool": "edit",
            "prompt": prompt,
            "image_url": final["image_url"],
            "all_urls": final["all_urls"],
            "activity_id": activity_id,
            "time_taken_seconds": final.get("time_taken"),
            "method": "multipart",
        }

    raise HTTPException(500, final.get("message", "Failed"))


# ==================================================================
# ============ VIEW — direct image redirect =====================
# ==================================================================
@app.get("/api/view", tags=["Generate"])
async def view_image(
    prompt: str = Query(...),
    quality: str = Query("low"),
    aspect_ratio: str = Query("16:9", alias="aspectRatio"),
    max_wait: int = Query(90, ge=10, le=120),
):
    try:
        res = await generate(prompt=prompt, quality=quality, aspect_ratio=aspect_ratio, max_wait=max_wait)
        return RedirectResponse(url=res["image_url"])
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"success": False, "error": e.detail})


# ==================================================================
# ============ POST /api/generate ================================
# ==================================================================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    quality: str = "low"
    aspect_ratio: str = "16:9"
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
# ============ STATUS ============================================
# ==================================================================
@app.get("/api/status/{activity_id}", tags=["Generate"])
async def check_status(activity_id: str):
    res = await get_status(activity_id)
    if res["status"] != 200:
        raise HTTPException(502, "Status check failed")

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

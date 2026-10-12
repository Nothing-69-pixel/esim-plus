from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import asyncio
import base64
import time
import os
from datetime import datetime, timezone

app = FastAPI(
    title="AI Image Generator + Photo Editor",
    description="Text-to-Image + Photo Editing via template.net",
    version="9.0.0",
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

WORKING_MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-1",
    "gpt-image-2.5",
    "gpt-image-2",
    "dall-e-3",
]

# Upload candidates (auto-tried)
UPLOAD_ENDPOINTS = [
    f"{BASE_URL}/api/v3/upload",
    f"{BASE_URL}/api/v3/image/upload",
    f"{BASE_URL}/api/v3/images/upload",
    f"{BASE_URL}/api/v3/uploads",
    f"{BASE_URL}/api/v3/upload-image",
    f"{BASE_URL}/api/v3/photo-editor/upload",
    f"{BASE_URL}/api/v3/image-editor/upload",
    f"{BASE_URL}/api/v3/files/upload",
    f"{BASE_URL}/api/v3/media/upload",
]

JOBS: Dict[str, dict] = {}


# ==================================================================
# ============ HTTP HELPERS ======================================
# ==================================================================
async def post_json(url: str, payload: dict, extra_headers: dict = None) -> dict:
    headers = {**HEADERS}
    if extra_headers:
        headers.update(extra_headers)
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            return {"status": r.status_code, "data": _safe_json(r)}
        except Exception as e:
            return {"status": 0, "error": str(e)[:200]}


async def post_multipart(url: str, files: dict, data: dict = None) -> dict:
    headers = {k: v for k, v in HEADERS.items() if k.lower() != "content-type"}
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            r = await client.post(url, files=files, data=data or {}, headers=headers)
            return {"status": r.status_code, "data": _safe_json(r)}
        except Exception as e:
            return {"status": 0, "error": str(e)[:200]}


async def get_status(activity_id: str) -> dict:
    params = {"activityId": activity_id}
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            r = await client.get(STATUS_URL, params=params, headers=HEADERS)
            return {"status": r.status_code, "data": _safe_json(r)}
        except Exception as e:
            return {"status": 0, "error": str(e)[:200]}


def _safe_json(r) -> dict:
    try:
        return r.json()
    except Exception:
        return {"_raw": r.text[:500]}


async def try_initiate(payload_base: dict) -> dict:
    """Try each model until activityId is returned."""
    last_error = None
    for model in WORKING_MODELS:
        payload = {**payload_base, "model": model}
        res = await post_json(GENERATE_URL, payload)
        status = res.get("status")
        data = res.get("data", {})
        activity_id = (
            data.get("activityId")
            or data.get("data", {}).get("activityId")
            or data.get("id")
        )
        if status in (200, 201) and activity_id:
            return {"success": True, "model_used": model, "activity_id": activity_id}
        last_error = {"model": model, "status": status, "response": str(data)[:200]}
    return {"success": False, "last_error": last_error}


async def wait_for_image(activity_id: str, max_seconds: int = 90) -> dict:
    """Poll until image ready."""
    start = time.time()
    while time.time() - start < max_seconds:
        try:
            res = await get_status(activity_id)
            data = res.get("data", {})
            inner = data.get("data", {})
            status = data.get("status")
            urls = inner.get("urls", []) or data.get("urls", [])

            if status == "completed" and urls:
                return {
                    "success": True,
                    "image_url": urls[0],
                    "all_urls": urls,
                    "resolved_prompt": inner.get("resolvedPrompt") or data.get("resolvedPrompt"),
                    "time_taken": round(time.time() - start, 1),
                }
            if status in ("failed", "error"):
                return {"success": False, "message": data.get("message", "Failed")}
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)
    return {"success": False, "message": f"Timeout after {max_seconds}s"}


# ==================================================================
# ============ UPLOAD HELPERS (multi-strategy) ===================
# ==================================================================
async def upload_image(image_bytes: bytes, filename: str, content_type: str) -> dict:
    """
    Upload image to template.net. Tries multiple strategies:
    1. Multipart to each candidate endpoint
    2. JSON base64 to each candidate endpoint
    Returns: {"success": bool, "image_url": str, "endpoint": str, "raw": dict}
    """
    errors = []

    # Strategy 1: Multipart
    for url in UPLOAD_ENDPOINTS:
        files = {"file": (filename, image_bytes, content_type)}
        res = await post_multipart(url, files)
        if res.get("status") in (200, 201):
            data = res.get("data", {})
            image_url = (
                data.get("url") or data.get("imageUrl") or data.get("image_url")
                or data.get("data", {}).get("url") or data.get("data", {}).get("imageUrl")
                or data.get("path")
            )
            if image_url:
                if image_url.startswith("/"):
                    image_url = f"https://www.template.net{image_url}"
                return {
                    "success": True,
                    "image_url": image_url,
                    "endpoint": url,
                    "method": "multipart",
                    "raw": data,
                }
        errors.append({"url": url, "method": "multipart", "status": res.get("status"), "response": str(res.get("data"))[:150]})

    # Strategy 2: field name "image"
    for url in UPLOAD_ENDPOINTS[:3]:
        files = {"image": (filename, image_bytes, content_type)}
        res = await post_multipart(url, files)
        if res.get("status") in (200, 201):
            data = res.get("data", {})
            image_url = (
                data.get("url") or data.get("imageUrl") or data.get("image_url")
                or data.get("data", {}).get("url")
            )
            if image_url:
                return {
                    "success": True,
                    "image_url": image_url,
                    "endpoint": url,
                    "method": "multipart_image",
                    "raw": data,
                }

    # Strategy 3: JSON base64
    b64 = base64.b64encode(image_bytes).decode()
    for url in UPLOAD_ENDPOINTS[:3]:
        payload = {
            "image": f"data:{content_type};base64,{b64}",
            "imageBase64": b64,
            "filename": filename,
        }
        res = await post_json(url, payload)
        if res.get("status") in (200, 201):
            data = res.get("data", {})
            image_url = (
                data.get("url") or data.get("imageUrl") or data.get("image_url")
                or data.get("data", {}).get("url")
            )
            if image_url:
                return {
                    "success": True,
                    "image_url": image_url,
                    "endpoint": url,
                    "method": "json_base64",
                    "raw": data,
                }

    return {"success": False, "errors": errors}


# ==================================================================
# ============ ROOT ==============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "AI Image Generator + Photo Editor",
        "version": "9.0.0",
        "endpoints": {
            "generate": "GET /api/generate?prompt=cat",
            "edit_upload": "POST /api/edit (multipart: image + prompt)",
            "edit_url": "GET /api/edit-url?image_url=...&prompt=...",
            "discover_upload": "GET /api/discover-upload",
            "jobs": "GET /api/jobs",
            "health": "GET /api/health",
        },
    }


@app.get("/api/health", tags=["System"])
async def health():
    return {"success": True, "status": "healthy", "jobs": len(JOBS)}


# ==================================================================
# ============ DISCOVERY — find working upload endpoint ==========
# ==================================================================
@app.get("/api/discover-upload", tags=["System"])
async def discover_upload():
    """Probe upload endpoints with a tiny test image."""
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    results = []
    for url in UPLOAD_ENDPOINTS:
        # Try multipart with "file"
        r1 = await post_multipart(url, {"file": ("test.png", tiny_png, "image/png")})
        results.append({
            "url": url,
            "method": "multipart/file",
            "status": r1.get("status"),
            "response": str(r1.get("data"))[:200],
        })
        # Try multipart with "image"
        r2 = await post_multipart(url, {"image": ("test.png", tiny_png, "image/png")})
        results.append({
            "url": url,
            "method": "multipart/image",
            "status": r2.get("status"),
            "response": str(r2.get("data"))[:200],
        })

    working = [r for r in results if r.get("status") in (200, 201)]
    return {
        "total_tested": len(results),
        "working_count": len(working),
        "working": working,
        "all": results,
    }


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
    payload_base = {
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

    init = await try_initiate(payload_base)
    if not init["success"]:
        raise HTTPException(502, {"error": "Initiate failed", "last_error": init.get("last_error")})

    JOBS[init["activity_id"]] = {
        "prompt": prompt, "model": init["model_used"], "tool": "generate",
        "started": datetime.now(timezone.utc).isoformat(),
    }

    final = await wait_for_image(init["activity_id"], max_seconds=max_wait)
    if not final["success"]:
        raise HTTPException(500, final.get("message", "Failed"))

    JOBS[init["activity_id"]]["image_url"] = final["image_url"]

    return {
        "success": True,
        "tool": "generate",
        "prompt": prompt,
        "model_used": init["model_used"],
        "image_url": final["image_url"],
        "all_urls": final["all_urls"],
        "activity_id": init["activity_id"],
        "time_taken_seconds": final["time_taken"],
    }


# ==================================================================
# ============ EDIT — Upload + Prompt ============================
# ==================================================================
@app.post("/api/edit", tags=["Edit"])
async def edit(
    image: UploadFile = File(..., description="Image to edit"),
    prompt: str = Form(..., description="Edit instructions"),
    quality: str = Form("low"),
    max_wait: int = Form(90),
):
    """
    Complete flow:
    1. Upload image to template.net
    2. Initiate photo editor with image URL + prompt
    3. Poll until edited image ready
    4. Return edited image URL
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 10MB)")

    filename = image.filename or "photo.png"
    content_type = image.content_type or "image/png"

    # ── Strategy A: Direct JSON base64 with photo-editor tool ──
    b64 = base64.b64encode(image_bytes).decode()
    payload_base = {
        "toolPrompt": "Edit the attached photo",
        "userPrompt": prompt,
        "quality": quality,
        "imageBase64": b64,
        "imageName": filename,
        "tool": {
            "slug": "ai-photo-editor",
            "graphicType": "element",
            "from": "/ai-photo-editor__ai-photo-editor",
            "isPanelCall": False,
        },
    }

    init = await try_initiate(payload_base)
    method_used = "json_base64"

    # ── Strategy B: Upload image first, then use URL ──
    if not init["success"]:
        upload_result = await upload_image(image_bytes, filename, content_type)
        if upload_result.get("success"):
            # Retry with imageUrl
            payload_base["imageUrl"] = upload_result["image_url"]
            payload_base.pop("imageBase64", None)
            payload_base.pop("imageName", None)
            init = await try_initiate(payload_base)
            method_used = f"upload_{upload_result['method']}"

    # ── Strategy C: Multipart to generate endpoint ──
    if not init["success"]:
        for model in WORKING_MODELS:
            files = {"image": (filename, image_bytes, content_type)}
            data = {
                "toolPrompt": "Edit the attached photo",
                "userPrompt": prompt,
                "quality": quality,
                "model": model,
                "tool[slug]": "ai-photo-editor",
                "tool[graphicType]": "element",
                "tool[from]": "/ai-photo-editor__ai-photo-editor",
                "tool[isPanelCall]": "false",
            }
            res = await post_multipart(GENERATE_URL, files, data)
            if res.get("status") in (200, 201):
                resp_data = res.get("data", {})
                aid = (
                    resp_data.get("activityId")
                    or resp_data.get("data", {}).get("activityId")
                    or resp_data.get("id")
                )
                if aid:
                    init = {"success": True, "model_used": model, "activity_id": aid}
                    method_used = "multipart_generate"
                    break

    if not init["success"]:
        raise HTTPException(502, {
            "error": "All edit strategies failed",
            "last_error": init.get("last_error"),
            "hint": "Try /api/discover-upload to find working upload endpoint",
        })

    JOBS[init["activity_id"]] = {
        "prompt": prompt, "model": init["model_used"], "tool": "edit",
        "method": method_used,
        "started": datetime.now(timezone.utc).isoformat(),
    }

    final = await wait_for_image(init["activity_id"], max_seconds=max_wait)
    if not final["success"]:
        raise HTTPException(500, final.get("message", "Edit failed"))

    JOBS[init["activity_id"]]["image_url"] = final["image_url"]

    return {
        "success": True,
        "tool": "edit",
        "prompt": prompt,
        "model_used": init["model_used"],
        "method": method_used,
        "image_url": final["image_url"],
        "all_urls": final["all_urls"],
        "activity_id": init["activity_id"],
        "time_taken_seconds": final["time_taken"],
    }


# ==================================================================
# ============ EDIT via Image URL ================================
# ==================================================================
@app.get("/api/edit-url", tags=["Edit"])
async def edit_url(
    image_url: str = Query(..., description="Public URL of image to edit"),
    prompt: str = Query(..., description="Edit instructions"),
    quality: str = Query("low"),
    max_wait: int = Query(90, ge=10, le=120),
):
    """Edit image from a URL (no file upload)."""
    payload_base = {
        "toolPrompt": "Edit the attached photo",
        "userPrompt": prompt,
        "quality": quality,
        "imageUrl": image_url,
        "tool": {
            "slug": "ai-photo-editor",
            "graphicType": "element",
            "from": "/ai-photo-editor__ai-photo-editor",
            "isPanelCall": False,
        },
    }

    init = await try_initiate(payload_base)
    if not init["success"]:
        raise HTTPException(502, {"error": "Initiate failed", "details": init.get("last_error")})

    final = await wait_for_image(init["activity_id"], max_seconds=max_wait)
    if not final["success"]:
        raise HTTPException(500, final.get("message", "Failed"))

    return {
        "success": True,
        "tool": "edit",
        "prompt": prompt,
        "image_url": final["image_url"],
        "all_urls": final["all_urls"],
        "activity_id": init["activity_id"],
        "time_taken_seconds": final["time_taken"],
    }


# ==================================================================
# ============ STATUS ============================================
# ==================================================================
@app.get("/api/status/{activity_id}", tags=["System"])
async def check_status(activity_id: str):
    res = await get_status(activity_id)
    if res.get("status") != 200:
        raise HTTPException(502, "Status check failed")

    data = res.get("data", {})
    inner = data.get("data", {})
    urls = inner.get("urls", []) or data.get("urls", [])
    status = data.get("status")

    if status == "completed" and urls:
        return {
            "success": True,
            "status": "completed",
            "image_url": urls[0],
            "all_urls": urls,
        }
    return {
        "success": True,
        "status": status,
        "prompt": inner.get("userPrompt"),
        "urls": urls,
    }


# ==================================================================
# ============ JOBS ==============================================
# ==================================================================
@app.get("/api/jobs", tags=["System"])
async def list_jobs():
    jobs = sorted(JOBS.values(), key=lambda x: x.get("started", ""), reverse=True)
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

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from typing import List, Dict, Optional
import httpx
import asyncio
import base64
import time
import os
import json
from datetime import datetime, timezone

app = FastAPI(
    title="AI Photo Editor API",
    description="Real photo editing + generation via template.net",
    version="10.0.0",
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

# ============ CONFIG ============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

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
POLL_INTERVAL = 1.5

WORKING_MODELS = [
    "gpt-image-2.5-flare",
    "gpt-image-1",
    "gpt-image-2.5",
    "gpt-image-2",
    "dall-e-3",
]

# Upload candidate endpoints — সব try হবে
UPLOAD_ENDPOINTS = [
    f"{BASE_URL}/api/v3/upload",
    f"{BASE_URL}/api/v3/images/upload",
    f"{BASE_URL}/api/v3/image/upload",
    f"{BASE_URL}/api/v3/upload-image",
    f"{BASE_URL}/api/v3/files",
    f"{BASE_URL}/api/v3/uploads",
    f"{BASE_URL}/api/v3/media/upload",
    f"{BASE_URL}/api/v3/workspace/upload",
    f"{BASE_URL}/api/v3/editor/upload",
    f"{BASE_URL}/api/v3/photo-editor/upload",
    f"{BASE_URL}/api/v3/image-editor/upload",
    f"{BASE_URL}/api/v3/reference/upload",
    f"{BASE_URL}/api/v3/attachments/upload",
    f"{BASE_URL}/api/v3/files/upload",
]

JOBS: Dict[str, dict] = {}
UPLOAD_CACHE: Dict = {"endpoint": None, "tested_at": 0}
UPLOAD_CACHE_TTL = 1800  # 30 min


# ==================================================================
# ============ HTTP HELPERS ======================================
# ==================================================================
def _safe_json(r) -> dict:
    try:
        return r.json()
    except Exception:
        return {"_raw": r.text[:500]}


async def post_json(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            r = await client.post(url, json=payload, headers=HEADERS)
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
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            r = await client.get(STATUS_URL, params={"activityId": activity_id}, headers=HEADERS)
            return {"status": r.status_code, "data": _safe_json(r)}
        except Exception as e:
            return {"status": 0, "error": str(e)[:200]}


# ==================================================================
# ============ UPLOAD DISCOVERY ==================================
# ==================================================================
def _extract_image_url(data: dict) -> Optional[str]:
    """Search nested data for any image URL."""
    if not isinstance(data, dict):
        return None

    # Common keys
    for key in ["url", "imageUrl", "image_url", "path", "file_url", "fileUrl", "src"]:
        v = data.get(key)
        if isinstance(v, str) and (v.startswith("http") or v.startswith("/")):
            return v

    # Nested "data"
    if "data" in data and isinstance(data["data"], (dict, list)):
        return _extract_image_url(data["data"]) if isinstance(data["data"], dict) else (
            _extract_image_url(data["data"][0]) if data["data"] else None
        )

    # Files list
    if "files" in data and isinstance(data["files"], list) and data["files"]:
        return _extract_image_url(data["files"][0])

    return None


def _extract_workspace_id(data: dict) -> Optional[int]:
    """Search for workspaceTemplateId."""
    if not isinstance(data, dict):
        return None
    for key in ["workspaceTemplateId", "workspace_template_id", "workspaceId", "workspace_id"]:
        v = data.get(key)
        if isinstance(v, (int, str)) and str(v).isdigit():
            return int(v)
    if "data" in data and isinstance(data["data"], dict):
        return _extract_workspace_id(data["data"])
    return None


async def discover_upload_endpoint(image_bytes: bytes = None, filename: str = "test.png") -> Optional[dict]:
    """
    Find a working upload endpoint by probing each candidate.
    Caches result for UPLOAD_CACHE_TTL.
    """
    now = time.time()
    if UPLOAD_CACHE["endpoint"] and (now - UPLOAD_CACHE["tested_at"] < UPLOAD_CACHE_TTL):
        return UPLOAD_CACHE["endpoint"]

    if not image_bytes:
        image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        filename = "test.png"

    for url in UPLOAD_ENDPOINTS:
        # Try field "file"
        r = await post_multipart(url, {"file": (filename, image_bytes, "image/png")})
        if r.get("status") in (200, 201):
            img_url = _extract_image_url(r.get("data", {}))
            ws_id = _extract_workspace_id(r.get("data", {}))
            if img_url:
                result = {
                    "endpoint": url,
                    "field": "file",
                    "image_url": img_url,
                    "workspace_id": ws_id,
                    "raw": r.get("data"),
                }
                UPLOAD_CACHE["endpoint"] = result
                UPLOAD_CACHE["tested_at"] = now
                return result

        # Try field "image"
        r = await post_multipart(url, {"image": (filename, image_bytes, "image/png")})
        if r.get("status") in (200, 201):
            img_url = _extract_image_url(r.get("data", {}))
            ws_id = _extract_workspace_id(r.get("data", {}))
            if img_url:
                result = {
                    "endpoint": url,
                    "field": "image",
                    "image_url": img_url,
                    "workspace_id": ws_id,
                    "raw": r.get("data"),
                }
                UPLOAD_CACHE["endpoint"] = result
                UPLOAD_CACHE["tested_at"] = now
                return result

    return None


async def upload_image(image_bytes: bytes, filename: str, content_type: str) -> dict:
    """Upload image using discovered endpoint."""
    # First, try cached/discovered endpoint
    endpoint_info = await discover_upload_endpoint(image_bytes, filename)

    if endpoint_info:
        field = endpoint_info.get("field", "file")
        endpoint = endpoint_info["endpoint"]
        files = {field: (filename, image_bytes, content_type)}
        r = await post_multipart(endpoint, files)
        if r.get("status") in (200, 201):
            img_url = _extract_image_url(r.get("data", {}))
            ws_id = _extract_workspace_id(r.get("data", {}))
            if img_url:
                return {
                    "success": True,
                    "image_url": img_url,
                    "workspace_id": ws_id,
                    "endpoint": endpoint,
                    "field": field,
                    "raw": r.get("data"),
                }

    # Fallback: try all endpoints
    for url in UPLOAD_ENDPOINTS:
        for field in ["file", "image", "photo", "media"]:
            files = {field: (filename, image_bytes, content_type)}
            r = await post_multipart(url, files)
            if r.get("status") in (200, 201):
                img_url = _extract_image_url(r.get("data", {}))
                if img_url:
                    return {
                        "success": True,
                        "image_url": img_url,
                        "workspace_id": _extract_workspace_id(r.get("data", {})),
                        "endpoint": url,
                        "field": field,
                        "raw": r.get("data"),
                    }

    return {"success": False, "error": "No upload endpoint worked"}


# ==================================================================
# ============ GENERATE / EDIT CORE ==============================
# ==================================================================
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
            return {
                "success": True,
                "model_used": model,
                "activity_id": activity_id,
                "raw": data,
            }
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


async def edit_photo_real(image_bytes: bytes, filename: str, content_type: str, prompt: str, quality: str = "low", max_wait: int = 90) -> dict:
    """
    Real photo editing flow:
    1. Upload image → get image_url + workspaceTemplateId
    2. Initiate photo editor with image_url + workspaceTemplateId + prompt
    3. Wait for completion
    """
    # ── Step 1: Upload ──
    upload = await upload_image(image_bytes, filename, content_type)
    if not upload.get("success"):
        return {
            "success": False,
            "error": "Upload failed",
            "detail": upload.get("error"),
        }

    image_url = upload["image_url"]
    workspace_id = upload.get("workspace_id")

    # ── Step 2: Initiate photo edit ──
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

    if workspace_id:
        payload_base["workspaceTemplateId"] = workspace_id

    init = await try_initiate(payload_base)

    # ── Fallback: try with references ──
    if not init["success"]:
        payload_base["references"] = [image_url]
        payload_base["referencesCount"] = 1
        payload_base.pop("imageUrl", None)
        init = await try_initiate(payload_base)

    # ── Fallback: try base64 ──
    if not init["success"]:
        b64 = base64.b64encode(image_bytes).decode()
        payload_base["imageBase64"] = b64
        payload_base["imageName"] = filename
        payload_base.pop("references", None)
        payload_base.pop("referencesCount", None)
        init = await try_initiate(payload_base)

    if not init["success"]:
        return {
            "success": False,
            "error": "Initiate failed",
            "detail": init.get("last_error"),
            "upload_info": {
                "image_url": image_url,
                "workspace_id": workspace_id,
            },
        }

    # ── Step 3: Wait for edited image ──
    final = await wait_for_image(init["activity_id"], max_seconds=max_wait)
    if not final["success"]:
        return {"success": False, "error": final.get("message", "Edit failed")}

    return {
        "success": True,
        "image_url": final["image_url"],
        "all_urls": final["all_urls"],
        "activity_id": init["activity_id"],
        "model_used": init["model_used"],
        "uploaded_image_url": image_url,
        "workspace_id": workspace_id,
        "time_taken_seconds": final["time_taken"],
    }


# ==================================================================
# ============ ROOT / HEALTH =====================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "AI Photo Editor API",
        "version": "10.0.0",
        "endpoints": {
            "generate": "GET /api/generate?prompt=cat",
            "edit": "POST /api/edit (image + prompt)",
            "edit_url": "GET /api/edit-url?image_url=...&prompt=...",
            "discover_upload": "GET /api/discover-upload",
            "jobs": "GET /api/jobs",
            "health": "GET /api/health",
        },
    }


@app.get("/api/health", tags=["System"])
async def health():
    return {
        "success": True,
        "status": "healthy",
        "upload_endpoint_cached": UPLOAD_CACHE.get("endpoint", {}).get("endpoint") if UPLOAD_CACHE.get("endpoint") else None,
        "jobs": len(JOBS),
    }


# ==================================================================
# ============ DISCOVER UPLOAD ===================================
# ==================================================================
@app.get("/api/discover-upload", tags=["System"])
async def discover_upload():
    """Find working upload endpoint."""
    # Clear cache to force fresh test
    UPLOAD_CACHE["endpoint"] = None
    UPLOAD_CACHE["tested_at"] = 0

    result = await discover_upload_endpoint()
    return {
        "success": result is not None,
        "working_endpoint": result,
        "tested_endpoints": UPLOAD_ENDPOINTS,
    }


@app.get("/api/discover-upload-full", tags=["System"])
async def discover_upload_full():
    """Test every endpoint and show all responses."""
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    results = []
    for url in UPLOAD_ENDPOINTS:
        for field in ["file", "image"]:
            r = await post_multipart(url, {field: ("test.png", tiny_png, "image/png")})
            results.append({
                "url": url,
                "field": field,
                "status": r.get("status"),
                "response": str(r.get("data"))[:250],
            })
    working = [x for x in results if x.get("status") in (200, 201)]
    return {
        "total": len(results),
        "working_count": len(working),
        "working": working,
        "all": results,
    }


# ==================================================================
# ============ GENERATE ==========================================
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
        raise HTTPException(502, {"error": "Initiate failed", "detail": init.get("last_error")})

    final = await wait_for_image(init["activity_id"], max_seconds=max_wait)
    if not final["success"]:
        raise HTTPException(500, final.get("message", "Failed"))

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
    image: UploadFile = File(...),
    prompt: str = Form(...),
    quality: str = Form("low"),
    max_wait: int = Form(90),
):
    """Upload image + prompt → real photo edit (not regenerate)."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (10MB max)")

    filename = image.filename or "photo.png"
    content_type = image.content_type or "image/png"

    result = await edit_photo_real(image_bytes, filename, content_type, prompt, quality, max_wait)

    if not result["success"]:
        raise HTTPException(500, result)

    return {
        "success": True,
        "tool": "edit",
        "prompt": prompt,
        "image_url": result["image_url"],
        "all_urls": result["all_urls"],
        "uploaded_image_url": result.get("uploaded_image_url"),
        "workspace_id": result.get("workspace_id"),
        "activity_id": result["activity_id"],
        "model_used": result.get("model_used"),
        "time_taken_seconds": result.get("time_taken_seconds"),
    }


# ==================================================================
# ============ EDIT FROM URL =====================================
# ==================================================================
@app.get("/api/edit-url", tags=["Edit"])
async def edit_url(
    image_url: str = Query(...),
    prompt: str = Query(...),
    quality: str = Query("low"),
    max_wait: int = Query(90, ge=10, le=120),
):
    """Edit image from public URL."""
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
        # Try with references
        payload_base["references"] = [image_url]
        payload_base["referencesCount"] = 1
        payload_base.pop("imageUrl", None)
        init = await try_initiate(payload_base)

    if not init["success"]:
        raise HTTPException(502, {"error": "Initiate failed", "detail": init.get("last_error")})

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
    return {
        "success": True,
        "status": data.get("status"),
        "image_url": urls[0] if urls else None,
        "all_urls": urls,
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

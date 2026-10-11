from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import re
import asyncio
import socket
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Free temporary phone numbers + SMS reader",
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

# ============ SOURCES (multi-fallback) ============
SOURCES = [
    "https://otp-api.shelex.dev/api",
    "https://otp-api.onrender.com/api",       # backup 1
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = 20.0


# ==================================================================
# ============ IPv4-FORCED TRANSPORT =============================
# ==================================================================
def ipv4_transport() -> httpx.AsyncHTTPTransport:
    """
    Force IPv4 for all DNS resolutions.
    Fixes 'Device or resource busy' on some Vercel environments.
    """
    return httpx.AsyncHTTPTransport(
        retries=2,
        local_address="0.0.0.0",  # Force IPv4
    )


async def robust_get(url: str, params: dict = None, timeout: float = TIMEOUT) -> Optional[httpx.Response]:
    """
    Robust GET with IPv4 forcing and multiple retry strategies.
    """
    # Attempt 1: IPv4-forced
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=HEADERS,
            transport=ipv4_transport(),
        ) as client:
            return await client.get(url, params=params)
    except Exception as e1:
        pass

    # Attempt 2: Default (no transport override)
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=HEADERS,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:
            return await client.get(url, params=params)
    except Exception as e2:
        pass

    # Attempt 3: http2 disabled + explicit close
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=HEADERS,
            http2=False,
        ) as client:
            resp = await client.get(url, params=params)
            return resp
    except Exception as e3:
        pass

    return None


async def try_all_sources(path: str, params: dict = None) -> tuple:
    """
    Try path on all source APIs.
    Returns (data, source_used) or raises HTTPException.
    """
    errors = []

    for source in SOURCES:
        url = f"{source}{path}"
        resp = await robust_get(url, params=params)

        if resp is None:
            errors.append({"source": source, "error": "network failure"})
            continue

        if resp.status_code == 200:
            try:
                return resp.json(), source
            except Exception:
                errors.append({"source": source, "error": "invalid JSON"})
                continue

        errors.append({
            "source": source,
            "status": resp.status_code,
            "error": resp.text[:100],
        })

    raise HTTPException(
        status_code=502,
        detail={
            "message": "All sources failed",
            "errors": errors,
        },
    )


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "6.0.0",
        "sources": SOURCES,
        "endpoints": {
            "test": "/api/test",
            "health": "/api/health",
            "countries": "/api/countries",
            "numbers": "/api/numbers?country=USA",
            "otp": "/api/otp/{country}/{phone}",
        },
    }


# ==================================================================
# ============ DNS RESOLVE TEST ==================================
# ==================================================================
def resolve_hostname(hostname: str) -> dict:
    """Resolve hostname to IPv4/IPv6 addresses."""
    result = {"hostname": hostname, "ipv4": [], "ipv6": [], "error": None}
    try:
        infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        for info in infos:
            family, _, _, _, addr = info
            if family == socket.AF_INET:
                result["ipv4"].append(addr[0])
            elif family == socket.AF_INET6:
                result["ipv6"].append(addr[0])
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


@app.get("/api/test", tags=["System"])
async def test():
    """Test DNS + network with different approaches."""
    results = {}

    # Test DNS resolution
    results["dns_google"] = resolve_hostname("www.google.com")
    results["dns_cloudflare"] = resolve_hostname("1.1.1.1")
    results["dns_otp_api"] = resolve_hostname("otp-api.shelex.dev")

    # Test HTTP connections
    async def test_url(url, name):
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                transport=ipv4_transport(),
                headers=HEADERS,
            ) as client:
                r = await client.get(url)
                return {"status": r.status_code, "ok": r.status_code < 500, "size": len(r.text)}
        except Exception as e:
            return {"error": str(e)[:150]}

    results["http_google"] = await test_url("https://www.google.com/generate_204", "google")
    results["http_otp_api"] = await test_url("https://otp-api.shelex.dev/api/countries", "otp_api")

    all_ok = (
        results["http_google"].get("ok", False)
        and results["http_otp_api"].get("ok", False)
    )

    return {
        "success": True,
        "all_working": all_ok,
        "results": results,
    }


# ==================================================================
# ============ HEALTH =============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    """Check aggregator reachability."""
    try:
        resp = await robust_get(f"{SOURCES[0]}/countries", timeout=10.0)
        if resp and resp.status_code == 200:
            return {
                "success": True,
                "status": "healthy",
                "source": SOURCES[0],
            }
        return {
            "success": False,
            "status": "unreachable",
            "source": SOURCES[0],
            "error": resp.text[:200] if resp else "no response",
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unreachable",
            "error": str(e)[:200],
        }


# ==================================================================
# ============ COUNTRIES =========================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    data, source_used = await try_all_sources("/countries")
    return {
        "success": True,
        "source_used": source_used,
        "countries": data,
        "count": len(data) if isinstance(data, list) else 0,
    }


# ==================================================================
# ============ NUMBERS ===========================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(..., description="Country (USA, UK, Canada)"),
    source: str = Query(None),
):
    params = {}
    if source:
        params["source"] = source

    data, source_used = await try_all_sources(f"/list/{country}", params=params)

    if isinstance(data, list):
        return {
            "success": True,
            "country": country,
            "source_used": source_used,
            "count": len(data),
            "numbers": data,
        }
    return {
        "success": True,
        "country": country,
        "source_used": source_used,
        "data": data,
    }


# ==================================================================
# ============ OTP ===============================================
# ==================================================================
@app.get("/api/otp/{country}/{phone}", tags=["SMS"])
async def get_otp(
    country: str,
    phone: str,
    match: str = Query(None),
    ago: str = Query("30s"),
    source: str = Query(None),
):
    clean_phone = re.sub(r"\D", "", phone)
    if not clean_phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    params = {"ago": ago}
    if match:
        params["match"] = match
    if source:
        params["source"] = source

    data, source_used = await try_all_sources(f"/{country}/{clean_phone}", params=params)

    return {
        "success": True,
        "country": country,
        "phone_number": clean_phone,
        "match": match,
        "ago": ago,
        "source_used": source_used,
        "data": data,
    }


# ==================================================================
# ============ ALL SMS ===========================================
# ==================================================================
@app.get("/api/sms/{country}/{phone}", tags=["SMS"])
async def get_all_sms(
    country: str,
    phone: str,
    ago: str = Query("1h"),
):
    clean_phone = re.sub(r"\D", "", phone)
    if not clean_phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    data, source_used = await try_all_sources(f"/{country}/{clean_phone}", params={"ago": ago})

    otp_codes = re.findall(r"\b\d{4,8}\b", str(data))

    return {
        "success": True,
        "country": country,
        "phone_number": clean_phone,
        "ago": ago,
        "source_used": source_used,
        "detected_otp_codes": list(set(otp_codes))[:10],
        "raw": data,
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

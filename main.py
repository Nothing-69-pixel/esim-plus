from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import re
import asyncio
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Free temporary phone numbers + SMS reader",
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

# ============ SOURCES ============
OTP_API = "https://otp-api.shelex.dev/api"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "close",
}

TIMEOUT = 20.0
MAX_RETRIES = 3


# ==================================================================
# ============ ROBUST HTTP WITH RETRY ============================
# ==================================================================
async def robust_get(url: str, params: dict = None, retries: int = MAX_RETRIES) -> Optional[httpx.Response]:
    """
    GET with retry and exponential backoff.
    Handles DNS/network errors gracefully.
    """
    last_error = None

    for attempt in range(retries):
        try:
            # Fresh client each time (avoids connection pool issues)
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers=HEADERS,
                limits=httpx.Limits(max_keepalive_connections=0, max_connections=5),
            ) as client:
                resp = await client.get(url, params=params)
                return resp
        except (httpx.ConnectError, httpx.NetworkError, OSError) as e:
            last_error = str(e)[:150]
            if attempt < retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # 1s, 2s backoff
            continue
        except httpx.TimeoutException as e:
            last_error = f"Timeout: {str(e)[:100]}"
            if attempt < retries - 1:
                await asyncio.sleep(1)
            continue
        except Exception as e:
            last_error = str(e)[:150]
            break

    return None


async def otp_fetch(path: str, params: dict = None) -> dict:
    """Fetch from OTP API with retry."""
    url = f"{OTP_API}{path}"
    resp = await robust_get(url, params=params)

    if resp is None:
        raise HTTPException(
            status_code=502,
            detail="Could not reach OTP aggregator (network/DNS issue)",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Aggregator error {resp.status_code}: {resp.text[:200]}",
        )

    try:
        return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from aggregator")


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "5.0.0",
        "backend": "otp-api.shelex.dev",
        "endpoints": {
            "countries": "/api/countries",
            "numbers": "/api/numbers?country=USA",
            "otp": "/api/otp/{country}/{phone}",
            "sms": "/api/sms/{country}/{phone}",
            "health": "/api/health",
            "test": "/api/test",
        },
    }


# ==================================================================
# ============ TEST ENDPOINT (no external call) ==================
# ==================================================================
@app.get("/api/test", tags=["System"])
async def test():
    """
    Test the basic HTTP connectivity without hitting OTP API.
    Uses a reliable endpoint to verify DNS/network works.
    """
    results = {}

    # Test 1: Google DNS
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("https://www.google.com/generate_204")
            results["google"] = {"status": r.status_code, "ok": r.status_code in (200, 204)}
    except Exception as e:
        results["google"] = {"error": str(e)[:100]}

    # Test 2: Cloudflare
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("https://1.1.1.1/cdn-cgi/trace")
            results["cloudflare"] = {"status": r.status_code, "ok": r.status_code == 200}
    except Exception as e:
        results["cloudflare"] = {"error": str(e)[:100]}

    # Test 3: OTP API countries
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
            r = await client.get(f"{OTP_API}/countries")
            results["otp_api"] = {
                "status": r.status_code,
                "size": len(r.text),
                "ok": r.status_code == 200,
            }
    except Exception as e:
        results["otp_api"] = {"error": str(e)[:150]}

    all_ok = all(r.get("ok", False) for r in results.values())

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
    """Quick health check."""
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
            r = await client.get(f"{OTP_API}/countries")
            return {
                "success": True,
                "status": "healthy" if r.status_code == 200 else "degraded",
                "aggregator_status": r.status_code,
                "aggregator_size": len(r.text),
            }
    except Exception as e:
        return {
            "success": False,
            "status": "unreachable",
            "error": str(e)[:200],
            "hint": "Try /api/test to diagnose DNS/network",
        }


# ==================================================================
# ============ COUNTRIES =========================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    data = await otp_fetch("/countries")
    return {
        "success": True,
        "countries": data,
        "count": len(data) if isinstance(data, list) else 0,
    }


# ==================================================================
# ============ NUMBERS ===========================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(..., description="Country name (USA, UK, Canada)"),
    source: str = Query(None, description="Optional provider filter"),
):
    params = {}
    if source:
        params["source"] = source

    data = await otp_fetch(f"/list/{country}", params=params)

    if isinstance(data, list):
        return {
            "success": True,
            "country": country,
            "count": len(data),
            "numbers": data,
        }
    return {
        "success": True,
        "country": country,
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

    data = await otp_fetch(f"/{country}/{clean_phone}", params=params)

    return {
        "success": True,
        "country": country,
        "phone_number": clean_phone,
        "match": match,
        "ago": ago,
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

    data = await otp_fetch(f"/{country}/{clean_phone}", params={"ago": ago})

    otp_codes = re.findall(r"\b\d{4,8}\b", str(data))

    return {
        "success": True,
        "country": country,
        "phone_number": clean_phone,
        "ago": ago,
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

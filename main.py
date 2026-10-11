from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import re
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Free temporary phone numbers + SMS reader — via OTP API aggregator",
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

# ============ OTP API AGGREGATOR ============
# Docs: https://otp-api.shelex.dev/docs/static/index.html
OTP_API_BASE = "https://otp-api.shelex.dev/api"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = 25.0


# ==================================================================
# ============ HTTP CLIENT =======================================
# ==================================================================
async def otp_request(path: str, params: dict = None):
    """Call otp-api.shelex.dev."""
    async with httpx.AsyncClient(
        timeout=TIMEOUT, follow_redirects=True, headers=HEADERS
    ) as client:
        url = f"{OTP_API_BASE}{path}"
        try:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream error: {resp.text[:200]}",
                )
            return resp.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "4.0.0",
        "backend": "otp-api.shelex.dev (multi-provider aggregator)",
        "providers": [
            "receive-sms-free.cc",
            "anonymsms.com",
            "quackr.io",
            "smstome.com",
            "receivesms.co",
            "receiveasmsonline.com",
            "receive-smss-online.com",
        ],
        "endpoints": {
            "countries": "/api/countries",
            "numbers": "/api/numbers?country=USA",
            "otp": "/api/otp/{country}/{phone}?match=code&ago=10m",
            "sms_all": "/api/sms/{country}/{phone}",
            "health": "/api/health",
        },
    }


# ==================================================================
# ============ LIST COUNTRIES ====================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    """List all available countries from aggregator."""
    data = await otp_request("/countries")
    return {
        "success": True,
        "countries": data,
        "count": len(data) if isinstance(data, list) else 0,
    }


# ==================================================================
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(..., description="Country name (e.g. USA, UK, Canada)"),
    source: str = Query(None, description="Optional provider filter"),
):
    """
    List all phone numbers for a country.
    Example: /api/numbers?country=USA
    """
    params = {}
    if source:
        params["source"] = source

    data = await otp_request(f"/list/{country}", params=params)

    if isinstance(data, list):
        return {
            "success": True,
            "country": country,
            "count": len(data),
            "numbers": data,
        }
    elif isinstance(data, dict):
        return {
            "success": True,
            "country": country,
            "data": data,
        }
    else:
        return {
            "success": False,
            "country": country,
            "message": "Unexpected response format",
            "raw": str(data)[:500],
        }


# ==================================================================
# ============ GET OTP / SMS =====================================
# ==================================================================
@app.get("/api/otp/{country}/{phone}", tags=["SMS"])
async def get_otp(
    country: str,
    phone: str,
    match: str = Query(None, description="Substring to look for in SMS"),
    ago: str = Query("30s", description="Time window: 10s, 5m, 1h, 2d"),
    source: str = Query(None, description="Provider: receive-sms-free.cc, smstome.com, ..."),
):
    """
    Fetch latest OTP/SMS for a specific number.

    Examples:
    - /api/otp/USA/1234567890
    - /api/otp/UK/447455958594?match=amazon&ago=5m
    - /api/otp/USA/1234567890?source=smstome.com
    """
    # Clean phone
    clean_phone = re.sub(r"\D", "", phone)
    if not clean_phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    params = {"ago": ago}
    if match:
        params["match"] = match
    if source:
        params["source"] = source

    data = await otp_request(f"/{country}/{clean_phone}", params=params)

    return {
        "success": True,
        "country": country,
        "phone_number": clean_phone,
        "match": match,
        "ago": ago,
        "source": source,
        "data": data,
    }


# ==================================================================
# ============ GET ALL SMS (no filter) ===========================
# ==================================================================
@app.get("/api/sms/{country}/{phone}", tags=["SMS"])
async def get_all_sms(
    country: str,
    phone: str,
    ago: str = Query("1h", description="Time window: 10s, 5m, 1h, 2d"),
):
    """
    Get all recent SMS for a number (no match filter).
    """
    clean_phone = re.sub(r"\D", "", phone)
    if not clean_phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    data = await otp_request(f"/{country}/{clean_phone}", params={"ago": ago})

    # Try to extract OTP codes from response
    otp_codes = []
    body_text = str(data)
    otp_codes = re.findall(r"\b\d{4,8}\b", body_text)

    return {
        "success": True,
        "country": country,
        "phone_number": clean_phone,
        "ago": ago,
        "detected_otp_codes": list(set(otp_codes))[:10],
        "raw": data,
    }


# ==================================================================
# ============ HEALTH =============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    """Check if aggregator API is reachable."""
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
            r = await client.get(f"{OTP_API_BASE}/countries")
            return {
                "success": True,
                "status": "healthy",
                "aggregator_reachable": r.status_code == 200,
                "aggregator_status": r.status_code,
            }
    except Exception as e:
        return {
            "success": False,
            "status": "unreachable",
            "error": str(e)[:200],
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

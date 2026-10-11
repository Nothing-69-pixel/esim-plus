from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import re
import json
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Temporary phone numbers + SMS reader — powered by esimplus.me",
    version="1.0.0",
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
BASE_URL = "https://esimplus.me"
NUMBERS_PAGE = f"{BASE_URL}/temporary-numbers"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; M2103K19I Build/TP1A.220624.014) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.8010.36 Mobile Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
}

TIMEOUT = 25.0


# ==================================================================
# ============ HTTP HELPERS ======================================
# ==================================================================
async def fetch_html(url: str) -> str:
    """Fetch HTML page from esimplus.me."""
    async with httpx.AsyncClient(
        timeout=TIMEOUT, follow_redirects=True, headers=HEADERS
    ) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream error {resp.status_code}",
                )
            return resp.text
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")


# ==================================================================
# ============ PARSING HELPERS ===================================
# ==================================================================
def extract_numbers_from_html(html: str) -> List[dict]:
    """
    Extract phone number objects from __next_f streamed JSON.
    Pattern: {"provider":"...","phoneNumber":"...","countryCode":"...","slug":"...","createdAt":"...","friendlyPhoneNumber":"..."}
    """
    # Find all number-like JSON objects
    pattern = re.compile(
        r'\{"provider":"([^"]+)","phoneNumber":"(\d+)","countryCode":"([^"]+)",'
        r'"slug":"([^"]+)","description":(?:null|"[^"]*"),'
        r'"createdAt":"([^"]+)","friendlyPhoneNumber":"([^"]+)"\}'
    )

    numbers = []
    seen = set()

    for m in pattern.finditer(html):
        provider, phone, cc, slug, created, friendly = m.groups()
        key = phone
        if key in seen:
            continue
        seen.add(key)

        numbers.append({
            "phone_number": phone,
            "friendly_number": friendly,
            "country_code": cc,
            "slug": slug,
            "provider": provider,
            "created_at": created,
            "detail_url": f"{BASE_URL}/temporary-numbers/{slug}/{phone}",
            "api_url": f"/api/number/{phone}",
        })

    return numbers


def extract_sms_from_html(html: str) -> List[dict]:
    """
    Extract SMS messages from __next_f streamed JSON.
    Pattern: {"provider":"...","from":"...","to":"...","body":"...","receivedAt":"...","friendlyFrom":"...","friendlyTo":"..."}
    """
    pattern = re.compile(
        r'\{"provider":"([^"]+)","from":"([^"]*)","to":"([^"]*)",'
        r'"body":"((?:[^"\\]|\\.)*)","receivedAt":"([^"]+)",'
        r'"friendlyFrom":"([^"]*)","friendlyTo":"([^"]*)"\}',
        re.DOTALL,
    )

    messages = []
    seen = set()

    for m in pattern.finditer(html):
        provider, frm, to, body, received, friendly_from, friendly_to = m.groups()
        key = (frm, body[:50], received)
        if key in seen:
            continue
        seen.add(key)

        # Unescape body
        try:
            body_clean = body.encode().decode("unicode_escape")
        except Exception:
            body_clean = body
        body_clean = body_clean.replace("\\n", "\n").replace('\\"', '"')

        # Extract OTP-like codes
        otp_codes = re.findall(r"\b\d{4,8}\b", body_clean)

        messages.append({
            "from": frm,
            "to": to,
            "friendly_from": friendly_from,
            "friendly_to": friendly_to,
            "body": body_clean,
            "received_at": received,
            "provider": provider,
            "otp_codes": otp_codes,
        })

    # Sort newest first
    messages.sort(key=lambda x: x.get("received_at", ""), reverse=True)
    return messages


def extract_total_count(html: str) -> Optional[int]:
    """Extract total SMS count from pagination."""
    m = re.search(r'"pagination":\{"total":(\d+)', html)
    if m:
        return int(m.group(1))
    return None


def extract_number_from_detail_page(html: str) -> Optional[dict]:
    """Extract the focused number object from a detail page."""
    numbers = extract_numbers_from_html(html)
    if numbers:
        return numbers[0]
    return None


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "1.0.0",
        "backend": "esimplus.me",
        "endpoints": {
            "numbers": "/api/numbers",
            "numbers_by_country": "/api/numbers?country=US",
            "countries": "/api/countries",
            "number_detail": "/api/number/{phone_number}",
            "sms": "/api/sms/{phone_number}",
            "sms_otp_only": "/api/sms/{phone_number}?otp_only=true",
            "health": "/api/health",
        },
    }


# ==================================================================
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(None, description="Filter by country code (US, CA, PR)"),
    limit: int = Query(200, ge=1, le=500),
):
    """
    List all temporary phone numbers from esimplus.me.
    Optionally filter by country code.
    """
    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n["country_code"] == cc]

    numbers = numbers[:limit]

    # Group by country for summary
    by_country: Dict[str, int] = {}
    for n in numbers:
        cc = n["country_code"]
        by_country[cc] = by_country.get(cc, 0) + 1

    return {
        "success": True,
        "count": len(numbers),
        "by_country": by_country,
        "numbers": numbers,
    }


# ==================================================================
# ============ LIST COUNTRIES ====================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    """List available country codes with number counts."""
    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)

    by_country: Dict[str, dict] = {}
    for n in numbers:
        cc = n["country_code"]
        if cc not in by_country:
            by_country[cc] = {
                "country_code": cc,
                "slug": n["slug"],
                "count": 0,
                "sample_number": n["friendly_number"],
            }
        by_country[cc]["count"] += 1

    countries = sorted(by_country.values(), key=lambda x: x["count"], reverse=True)

    return {
        "success": True,
        "count": len(countries),
        "countries": countries,
    }


# ==================================================================
# ============ NUMBER DETAIL =====================================
# ==================================================================
@app.get("/api/number/{phone_number}", tags=["SMS"])
async def number_detail(phone_number: str):
    """
    Get details for a specific phone number.
    Example: /api/number/19393150836
    """
    # Clean input
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    # First find the slug from the list page
    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)
    target = next((n for n in numbers if n["phone_number"] == phone), None)

    if not target:
        return {
            "success": False,
            "message": f"Number {phone} not found in current listing",
            "phone_number": phone,
        }

    # Fetch the detail page
    detail_url = target["detail_url"]
    detail_html = await fetch_html(detail_url)

    sms = extract_sms_from_html(detail_html)
    total = extract_total_count(detail_html) or len(sms)

    return {
        "success": True,
        "number": target,
        "total_sms": total,
        "loaded_sms": len(sms),
        "sms": sms,
    }


# ==================================================================
# ============ SMS MESSAGES ======================================
# ==================================================================
@app.get("/api/sms/{phone_number}", tags=["SMS"])
async def get_sms(
    phone_number: str,
    otp_only: bool = Query(False, description="Only return messages with OTP codes"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get SMS messages for a phone number.
    Optionally filter to only OTP messages.
    """
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)
    target = next((n for n in numbers if n["phone_number"] == phone), None)

    if not target:
        return {
            "success": False,
            "message": f"Number {phone} not found",
            "phone_number": phone,
        }

    detail_html = await fetch_html(target["detail_url"])
    sms = extract_sms_from_html(detail_html)
    total = extract_total_count(detail_html) or len(sms)

    if otp_only:
        sms = [m for m in sms if m["otp_codes"]]

    sms = sms[:limit]

    return {
        "success": True,
        "phone_number": phone,
        "friendly_number": target["friendly_number"],
        "country_code": target["country_code"],
        "total_sms": total,
        "otp_only": otp_only,
        "count": len(sms),
        "messages": sms,
    }


# ==================================================================
# ============ LATEST OTP (all numbers) ==========================
# ==================================================================
@app.get("/api/latest-otp", tags=["SMS"])
async def latest_otp(
    country: str = Query(None, description="Filter by country code"),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Get latest OTP messages across all numbers (or a specific country).
    """
    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n["country_code"] == cc]

    # Only fetch from the first 5 numbers to avoid too many requests
    numbers = numbers[:5]

    all_otps = []
    for n in numbers:
        try:
            detail_html = await fetch_html(n["detail_url"])
            sms = extract_sms_from_html(detail_html)
            for m in sms:
                if m["otp_codes"]:
                    all_otps.append({
                        "phone_number": n["phone_number"],
                        "friendly_number": n["friendly_number"],
                        "country_code": n["country_code"],
                        "from": m["from"],
                        "body": m["body"],
                        "otp_codes": m["otp_codes"],
                        "received_at": m["received_at"],
                    })
        except Exception:
            continue

    all_otps.sort(key=lambda x: x.get("received_at", ""), reverse=True)
    all_otps = all_otps[:limit]

    return {
        "success": True,
        "count": len(all_otps),
        "country_filter": country,
        "otps": all_otps,
    }


# ==================================================================
# ============ HEALTH =============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    return {
        "success": True,
        "status": "healthy",
        "backend": "esimplus.me",
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


# ============ LOCAL DEV ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

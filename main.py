from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import re
import time
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Temporary phone numbers + SMS reader — powered by esimplus.me",
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
BASE_URL = "https://esimplus.me"
NUMBERS_PAGE = f"{BASE_URL}/temporary-numbers"

# Full browser-like headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24", "Google Chrome";v="131"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "DNT": "1",
    "Cookie": "cookie-consent=analytics_cookies%3D1%26other_cookies%3D1; NEXT_LOCALE=en",
}

# Minimal headers for internal API
API_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": BASE_URL,
    "Referer": NUMBERS_PAGE,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": HEADERS["Sec-Ch-Ua"],
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Cookie": HEADERS["Cookie"],
}

TIMEOUT = 30.0


# ==================================================================
# ============ HTTP HELPERS ======================================
# ==================================================================
async def fetch_html(url: str) -> str:
    """Fetch HTML page with full browser headers."""
    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
        http2=False,
    ) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Access forbidden by upstream (esimplus.me blocked the request)",
                )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream error {resp.status_code}: {resp.text[:200]}",
                )
            return resp.text
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")


async def fetch_json(url: str) -> Optional[dict]:
    """Try to fetch JSON from internal API."""
    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=API_HEADERS,
        http2=False,
    ) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None


async def try_internal_api() -> Optional[List[dict]]:
    """
    Try common Next.js internal API endpoints.
    Return list of numbers if any works.
    """
    candidates = [
        f"{BASE_URL}/api/temporary-numbers",
        f"{BASE_URL}/api/v1/temporary-numbers",
        f"{BASE_URL}/api/phone-numbers",
        f"{BASE_URL}/api/v1/phone-numbers",
        f"{BASE_URL}/api/numbers",
    ]

    for url in candidates:
        data = await fetch_json(url)
        if data and isinstance(data, (list, dict)):
            # Try to extract numbers from response
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Look for common keys
                for key in ["data", "numbers", "items", "results"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
        time.sleep(0.2)

    return None


# ==================================================================
# ============ PARSING HELPERS ===================================
# ==================================================================
def extract_numbers_from_html(html: str) -> List[dict]:
    """Extract phone numbers from Next.js streamed data."""
    pattern = re.compile(
        r'\{"provider":"([^"]+)","phoneNumber":"(\d+)","countryCode":"([^"]+)",'
        r'"slug":"([^"]+)","description":(?:null|"[^"]*"),'
        r'"createdAt":"([^"]+)","friendlyPhoneNumber":"([^"]+)"\}'
    )

    numbers = []
    seen = set()

    for m in pattern.finditer(html):
        provider, phone, cc, slug, created, friendly = m.groups()
        if phone in seen:
            continue
        seen.add(phone)

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
    """Extract SMS messages from Next.js streamed data."""
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

        try:
            body_clean = body.encode().decode("unicode_escape")
        except Exception:
            body_clean = body
        body_clean = body_clean.replace("\\n", "\n").replace('\\"', '"')

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

    messages.sort(key=lambda x: x.get("received_at", ""), reverse=True)
    return messages


def extract_total_count(html: str) -> Optional[int]:
    m = re.search(r'"pagination":\{"total":(\d+)', html)
    return int(m.group(1)) if m else None


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "2.0.0",
        "backend": "esimplus.me",
        "endpoints": {
            "numbers": "/api/numbers",
            "numbers_by_country": "/api/numbers?country=US",
            "countries": "/api/countries",
            "number_detail": "/api/number/{phone_number}",
            "sms": "/api/sms/{phone_number}",
            "sms_otp_only": "/api/sms/{phone_number}?otp_only=true",
            "latest_otp": "/api/latest-otp",
            "debug": "/api/debug",
            "health": "/api/health",
        },
    }


# ==================================================================
# ============ DEBUG ==============================================
# ==================================================================
@app.get("/api/debug", tags=["System"])
async def debug():
    """
    Test connectivity to esimplus.me with different approaches.
    """
    results = {}

    # Test 1: Direct HTML fetch
    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=HEADERS
        ) as client:
            r = await client.get(NUMBERS_PAGE)
            results["html_fetch"] = {
                "status": r.status_code,
                "size": len(r.text),
                "has_next_data": "__next_f" in r.text,
                "is_cloudflare": "cloudflare" in r.text.lower()[:2000],
            }
    except Exception as e:
        results["html_fetch"] = {"error": str(e)}

    # Test 2: Internal API endpoints
    api_tests = {}
    candidates = [
        f"{BASE_URL}/api/temporary-numbers",
        f"{BASE_URL}/api/v1/temporary-numbers",
        f"{BASE_URL}/api/phone-numbers",
        f"{BASE_URL}/api/numbers",
    ]
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=True, headers=API_HEADERS
    ) as client:
        for url in candidates:
            try:
                r = await client.get(url)
                api_tests[url] = {
                    "status": r.status_code,
                    "preview": r.text[:200],
                }
            except Exception as e:
                api_tests[url] = {"error": str(e)}

    results["api_tests"] = api_tests

    return {
        "success": True,
        "results": results,
    }


# ==================================================================
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(None, description="Filter by country code (US, CA, PR)"),
    limit: int = Query(200, ge=1, le=500),
):
    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n["country_code"] == cc]

    numbers = numbers[:limit]

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
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)
    target = next((n for n in numbers if n["phone_number"] == phone), None)

    if not target:
        return {
            "success": False,
            "message": f"Number {phone} not found in current listing",
            "phone_number": phone,
        }

    detail_html = await fetch_html(target["detail_url"])
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
    otp_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
):
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
# ============ LATEST OTP ========================================
# ==================================================================
@app.get("/api/latest-otp", tags=["SMS"])
async def latest_otp(
    country: str = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    html = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n["country_code"] == cc]

    numbers = numbers[:5]  # Only first 5 to avoid timeouts

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
        "version": "2.0.0",
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

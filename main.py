
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import httpx
import re
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Temporary phone numbers + SMS reader",
    version="3.0.0",
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

# ⭐ FIX: Removed 'br' from Accept-Encoding to avoid brotli decompression issues
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",  # ⭐ No br
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
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
}

# Jina AI Reader — free proxy to bypass blocks
JINA_PREFIX = "https://r.jina.ai/"

TIMEOUT = 30.0


# ==================================================================
# ============ HTTP FETCH (multi-strategy) =======================
# ==================================================================
async def fetch_html(url: str) -> tuple:
    """
    Try multiple strategies to fetch a URL.
    Returns (html_text, strategy_used).
    """
    # ---- Strategy 1: Direct fetch with proper headers ----
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                text = resp.text
                # Sanity check — real page should contain Next.js data
                if "__next_f" in text or "temporary-numbers" in text:
                    return text, "direct"
    except Exception:
        pass

    # ---- Strategy 2: Jina AI Reader proxy ----
    try:
        jina_url = f"{JINA_PREFIX}{url}"
        async with httpx.AsyncClient(
            timeout=45.0,
            follow_redirects=True,
        ) as client:
            resp = await client.get(
                jina_url,
                headers={
                    "User-Agent": HEADERS["User-Agent"],
                    "Accept": "text/plain, text/html, */*",
                },
            )
            if resp.status_code == 200 and resp.text:
                return resp.text, "jina_proxy"
    except Exception:
        pass

    # ---- Strategy 3: AllOrigins proxy ----
    try:
        allorigins = f"https://api.allorigins.win/raw?url={url}"
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            resp = await client.get(allorigins)
            if resp.status_code == 200 and resp.text:
                return resp.text, "allorigins_proxy"
    except Exception:
        pass

    # ---- All failed ----
    raise HTTPException(
        status_code=502,
        detail="All fetch strategies failed. Upstream may be blocking Vercel IPs.",
    )


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
        "version": "3.0.0",
        "endpoints": {
            "numbers": "/api/numbers",
            "countries": "/api/countries",
            "number_detail": "/api/number/{phone}",
            "sms": "/api/sms/{phone}",
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
    """Test connectivity to esimplus.me."""
    results = {}

    # Strategy 1: Direct
    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=HEADERS
        ) as client:
            r = await client.get(NUMBERS_PAGE)
            results["direct"] = {
                "status": r.status_code,
                "size": len(r.text),
                "has_next_data": "__next_f" in r.text,
                "content_type": r.headers.get("content-type", ""),
                "content_encoding": r.headers.get("content-encoding", ""),
            }
    except Exception as e:
        results["direct"] = {"error": str(e)[:200]}

    # Strategy 2: Jina AI reader
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(f"{JINA_PREFIX}{NUMBERS_PAGE}")
            results["jina"] = {
                "status": r.status_code,
                "size": len(r.text),
                "preview": r.text[:300],
            }
    except Exception as e:
        results["jina"] = {"error": str(e)[:200]}

    # Strategy 3: AllOrigins
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            r = await client.get(
                f"https://api.allorigins.win/raw?url={NUMBERS_PAGE}"
            )
            results["allorigins"] = {
                "status": r.status_code,
                "size": len(r.text),
                "has_next_data": "__next_f" in r.text,
            }
    except Exception as e:
        results["allorigins"] = {"error": str(e)[:200]}

    # Pick best strategy
    best = None
    if results.get("direct", {}).get("status") == 200 and results["direct"].get("has_next_data"):
        best = "direct"
    elif results.get("jina", {}).get("status") == 200:
        best = "jina"
    elif results.get("allorigins", {}).get("status") == 200:
        best = "allorigins"

    return {
        "success": True,
        "best_strategy": best,
        "results": results,
    }


# ==================================================================
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(None),
    limit: int = Query(200, ge=1, le=500),
):
    html, strategy = await fetch_html(NUMBERS_PAGE)
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
        "fetch_strategy": strategy,
        "numbers": numbers,
    }


# ==================================================================
# ============ COUNTRIES =========================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    html, strategy = await fetch_html(NUMBERS_PAGE)
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
        "fetch_strategy": strategy,
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

    html, _ = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)
    target = next((n for n in numbers if n["phone_number"] == phone), None)

    if not target:
        return {
            "success": False,
            "message": f"Number {phone} not found",
            "phone_number": phone,
        }

    detail_html, strategy = await fetch_html(target["detail_url"])
    sms = extract_sms_from_html(detail_html)
    total = extract_total_count(detail_html) or len(sms)

    return {
        "success": True,
        "number": target,
        "total_sms": total,
        "loaded_sms": len(sms),
        "fetch_strategy": strategy,
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

    html, _ = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)
    target = next((n for n in numbers if n["phone_number"] == phone), None)

    if not target:
        return {"success": False, "message": f"Number {phone} not found"}

    detail_html, strategy = await fetch_html(target["detail_url"])
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
        "fetch_strategy": strategy,
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
    html, _ = await fetch_html(NUMBERS_PAGE)
    numbers = extract_numbers_from_html(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n["country_code"] == cc]

    numbers = numbers[:3]  # Only 3 to save time

    all_otps = []
    for n in numbers:
        try:
            detail_html, _ = await fetch_html(n["detail_url"])
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
        "version": "3.0.0",
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

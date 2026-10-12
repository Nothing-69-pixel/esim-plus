from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import httpx
import re
import asyncio
import json

app = FastAPI(
    title="SMS24 API",
    description="Temporary phone numbers + SMS reader — powered by sms24.me",
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
BASE_URL = "https://sms24.me"
NUMBERS_PAGE = f"{BASE_URL}/en/numbers"
COUNTRIES_PAGE = f"{BASE_URL}/en/countries"
API_MESSAGES = f"{BASE_URL}/api/messages"

BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; M2103K19I Build/TP1A.220624.014) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.8010.36 Mobile Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
}

API_HEADERS_TEMPLATE = {
    "User-Agent": BASE_HEADERS["User-Agent"],
    "Accept": "application/json",
    "Accept-Language": "en",
    "Accept-Encoding": "gzip, deflate",
    "Content-Type": "application/json",
    "Origin": BASE_URL,
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Android WebView";v="153", "Not_A Brand";v="8", "Chromium";v="153"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
}

TIMEOUT = 25.0


# ==================================================================
# ============ HTTP HELPERS ======================================
# ==================================================================
async def fetch_html(url: str, cookie: Optional[str] = None, retries: int = 2) -> str:
    """Fetch HTML listing page."""
    headers = {**BASE_HEADERS}
    if cookie:
        headers["Cookie"] = cookie

    last_err = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers=headers,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.text
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)[:150]
        if attempt < retries:
            await asyncio.sleep(0.8)

    raise HTTPException(status_code=502, detail=f"Fetch failed: {last_err}")


async def call_messages_api(
    phone: str,
    page: int = 1,
    locale: str = "en",
    cookie: Optional[str] = None,
    turnstile_token: Optional[str] = None,
) -> dict:
    """Call the sms24.me messages API."""
    url = f"{API_MESSAGES}/{phone}"
    headers = {**API_HEADERS_TEMPLATE}
    headers["Referer"] = f"{BASE_URL}/{locale}/numbers/{phone}"
    if cookie:
        headers["Cookie"] = cookie
    if turnstile_token:
        headers["x-turnstile-token"] = turnstile_token

    payload = {"page": page, "locale": locale}

    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                return {
                    "_error": True,
                    "_status": r.status_code,
                    "_body": r.text[:1000],
                }
            try:
                return r.json()
            except Exception:
                return {"_raw": r.text[:2000], "_status": r.status_code}
    except Exception as e:
        return {"_error": True, "_exception": str(e)[:300]}


# ==================================================================
# ============ PARSERS ===========================================
# ==================================================================
def parse_numbers(html: str) -> List[dict]:
    """Parse phone numbers from listing page."""
    soup = BeautifulSoup(html, "html.parser")
    numbers = []
    seen = set()

    # Match /en/numbers/{digits} links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"/en/numbers/(\d{6,15})", href)
        if not m:
            continue
        phone = m.group(1)
        if phone in seen:
            continue
        seen.add(phone)

        # Friendly number
        text = a.get_text(separator=" ", strip=True)
        fm = re.search(r"\+\d[\d\s\-]+", text)
        friendly = fm.group(0).strip() if fm else f"+{phone}"

        # Country flag
        country = ""
        flag_span = a.find("span", class_=re.compile(r"flag-emoji", re.I))
        if flag_span:
            country = flag_span.get_text(strip=True)

        # Country name from h3
        h3 = a.find("h3")
        if h3:
            country = h3.get_text(strip=True)

        numbers.append({
            "phone_number": phone,
            "friendly_number": friendly,
            "country": country,
            "detail_url": f"{BASE_URL}/en/numbers/{phone}",
        })

    return numbers


def parse_countries(html: str) -> List[dict]:
    """Parse countries from countries page."""
    soup = BeautifulSoup(html, "html.parser")
    countries = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"/en/countries/([a-z]{2,3})(?:$|[/?#])", href)
        if not m:
            continue
        cc = m.group(1)
        if cc in seen:
            continue
        seen.add(cc)

        # Name from inner text
        text = a.get_text(separator=" ", strip=True)
        name = re.sub(r"[^\w\s\-()]+", "", text).strip()

        countries.append({
            "country_code": cc,
            "name": name or cc.upper(),
            "url": href,
        })

    return countries


def parse_number_page(html: str) -> dict:
    """Extract key info from number detail page."""
    soup = BeautifulSoup(html, "html.parser")
    info = {}

    title_el = soup.find("title")
    info["page_title"] = title_el.get_text(strip=True) if title_el else ""

    h1 = soup.find("h1")
    if h1:
        info["friendly_number"] = h1.get_text(strip=True)

    # data-endpoint attribute
    gate = soup.find(attrs={"data-endpoint": True})
    if gate:
        info["messages_api_url"] = gate.get("data-endpoint")

    # Turnstile site key
    ts = soup.find("meta", attrs={"name": "turnstile-site-key"})
    if ts:
        info["turnstile_site_key"] = ts.get("content")

    return info


def extract_otp(body: str) -> List[str]:
    """Extract OTP-like codes from message body."""
    if not body:
        return []
    codes = re.findall(r"\b\d{4,8}\b", body)
    # Filter out years and obvious non-OTPs (optional)
    return sorted(set(codes))


def format_message(msg: dict) -> dict:
    """Format a message from sms24 API."""
    content = msg.get("content", "")
    return {
        "id": msg.get("id"),
        "from": msg.get("from", "unknown"),
        "sender_url": msg.get("senderUrl"),
        "body": content,
        "kind": msg.get("kind", "sms"),
        "received_at": msg.get("created"),
        "otp_codes": extract_otp(content),
    }


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "SMS24 API",
        "version": "1.0.0",
        "source": "sms24.me",
        "endpoints": {
            "numbers": "/api/numbers?country=gb",
            "countries": "/api/countries",
            "number_info": "/api/number/{phone}",
            "messages": "/api/messages/{phone}",
            "messages_full": "/api/messages/{phone}?page=1",
            "otp_only": "/api/messages/{phone}?otp_only=true",
            "latest_otp": "/api/latest-otp?country=gb&scan=5",
            "health": "/api/health",
        },
        "note": "Provide ?turnstile=TOKEN and/or ?cookie=YOUR_COOKIE if blocked by Cloudflare.",
    }


# ==================================================================
# ============ HEALTH =============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=BASE_HEADERS) as client:
            r = await client.get(NUMBERS_PAGE)
            return {
                "success": True,
                "status": "healthy" if r.status_code == 200 else "degraded",
                "source_status": r.status_code,
            }
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


# ==================================================================
# ============ NUMBERS ===========================================
# ==================================================================
@app.get("/api/numbers", tags=["Numbers"])
async def list_numbers(
    country: str = Query(None, description="Country code (gb, us, ca)"),
    page: int = Query(1, ge=1),
    cookie: str = Query(None),
):
    if country:
        url = f"{COUNTRIES_PAGE}/{country.lower()}"
    else:
        url = NUMBERS_PAGE
    if page > 1:
        url += f"?page={page}"

    html = await fetch_html(url, cookie=cookie)
    numbers = parse_numbers(html)

    return {
        "success": True,
        "country": country,
        "page": page,
        "count": len(numbers),
        "numbers": numbers,
    }


# ==================================================================
# ============ COUNTRIES =========================================
# ==================================================================
@app.get("/api/countries", tags=["Numbers"])
async def list_countries(cookie: str = Query(None)):
    html = await fetch_html(COUNTRIES_PAGE, cookie=cookie)
    countries = parse_countries(html)
    return {"success": True, "count": len(countries), "countries": countries}


# ==================================================================
# ============ NUMBER INFO =======================================
# ==================================================================
@app.get("/api/number/{phone_number}", tags=["Numbers"])
async def number_info(phone_number: str, cookie: str = Query(None)):
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone")

    url = f"{BASE_URL}/en/numbers/{phone}"
    html = await fetch_html(url, cookie=cookie)
    info = parse_number_page(html)

    return {
        "success": True,
        "phone_number": phone,
        "url": url,
        **info,
    }


# ==================================================================
# ============ MESSAGES ⭐ ========================================
# ==================================================================
@app.get("/api/messages/{phone_number}", tags=["Messages"])
async def get_messages(
    phone_number: str,
    page: int = Query(1, ge=1),
    locale: str = Query("en"),
    otp_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    cookie: str = Query(None, description="sms24.me cookie (needs cf_clearance)"),
    turnstile: str = Query(None, description="x-turnstile-token value from browser"),
):
    """
    Fetch SMS messages. Uses sms24.me/api/messages/{phone} (POST).

    **How to get turnstile token:**
    1. Open https://sms24.me/en/numbers/{phone} in browser
    2. Click "Show SMS messages" button
    3. Open DevTools (F12) → Network tab
    4. Find the POST request to /api/messages/{phone}
    5. Copy the `x-turnstile-token` header value
    6. Pass as ?turnstile=...
    """
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone")

    data = await call_messages_api(
        phone=phone,
        page=page,
        locale=locale,
        cookie=cookie,
        turnstile_token=turnstile,
    )

    # Error handling
    if isinstance(data, dict) and data.get("_error"):
        return {
            "success": False,
            "phone_number": phone,
            "status": data.get("_status"),
            "error": data.get("_body") or data.get("_exception"),
            "hint": (
                "Cloudflare Turnstile required. Get turnstile token from browser DevTools "
                "and pass as ?turnstile=..."
            ),
        }

    raw_messages = data.get("messages", []) if isinstance(data, dict) else []
    formatted = [format_message(m) for m in raw_messages]

    if otp_only:
        formatted = [m for m in formatted if m["otp_codes"]]

    formatted = formatted[:limit]

    return {
        "success": True,
        "phone_number": phone,
        "page": data.get("page", page) if isinstance(data, dict) else page,
        "total_pages": data.get("pages", 1) if isinstance(data, dict) else 1,
        "total": data.get("total", len(formatted)) if isinstance(data, dict) else len(formatted),
        "otp_only": otp_only,
        "count": len(formatted),
        "messages": formatted,
    }


# ==================================================================
# ============ MESSAGES RAW (debug) ==============================
# ==================================================================
@app.get("/api/messages-raw/{phone_number}", tags=["Messages"])
async def get_messages_raw(
    phone_number: str,
    page: int = Query(1, ge=1),
    cookie: str = Query(None),
    turnstile: str = Query(None),
):
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone")

    data = await call_messages_api(phone=phone, page=page, cookie=cookie, turnstile_token=turnstile)
    return {
        "success": True,
        "phone_number": phone,
        "raw_response": data,
    }


# ==================================================================
# ============ LATEST OTP — multi-number scan ====================
# ==================================================================
@app.get("/api/latest-otp", tags=["Messages"])
async def latest_otp(
    country: str = Query("gb", description="Country code (gb, us, ca)"),
    scan: int = Query(5, ge=1, le=15, description="How many numbers to check"),
    limit: int = Query(30, ge=1, le=100),
    otp_only: bool = Query(True),
    cookie: str = Query(None),
    turnstile: str = Query(None),
):
    """
    Scan multiple numbers and collect latest OTPs.
    """
    # Get numbers
    url = f"{COUNTRIES_PAGE}/{country.lower()}"
    html = await fetch_html(url, cookie=cookie)
    numbers = parse_numbers(html)[:scan]

    if not numbers:
        return {"success": False, "message": "No numbers found"}

    all_otps = []
    per_number = []

    for n in numbers:
        try:
            data = await call_messages_api(
                phone=n["phone_number"],
                cookie=cookie,
                turnstile_token=turnstile,
            )
            if isinstance(data, dict) and data.get("_error"):
                per_number.append({
                    "phone_number": n["phone_number"],
                    "success": False,
                    "error": str(data.get("_status")),
                })
                continue

            raw_msgs = data.get("messages", []) if isinstance(data, dict) else []
            formatted = [format_message(m) for m in raw_msgs]

            if otp_only:
                formatted = [m for m in formatted if m["otp_codes"]]

            per_number.append({
                "phone_number": n["phone_number"],
                "friendly_number": n["friendly_number"],
                "success": True,
                "message_count": len(formatted),
            })

            for m in formatted:
                all_otps.append({
                    "phone_number": n["phone_number"],
                    "friendly_number": n["friendly_number"],
                    "from": m["from"],
                    "body": m["body"][:200],
                    "otp_codes": m["otp_codes"],
                    "received_at": m["received_at"],
                })
        except Exception as e:
            per_number.append({
                "phone_number": n["phone_number"],
                "success": False,
                "error": str(e)[:150],
            })

    all_otps = all_otps[:limit]

    return {
        "success": True,
        "country": country,
        "scanned": len(numbers),
        "count": len(all_otps),
        "otps": all_otps,
        "per_number": per_number,
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

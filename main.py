from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import httpx
import re
import asyncio

app = FastAPI(
    title="Tap SMS API",
    description="Temporary phone numbers + SMS reader — powered by temp-number.com",
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
BASE_URL = "https://temp-number.com"
NUMBERS_PAGE = f"{BASE_URL}/temporary-numbers"

# Base browser headers (no auth)
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
    "Cache-Control": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
}

TIMEOUT = 25.0


# ==================================================================
# ============ HTTP FETCH ========================================
# ==================================================================
async def fetch(url: str, cookie: Optional[str] = None, retries: int = 2) -> str:
    """Fetch HTML with optional cookie and retry."""
    headers = {**BASE_HEADERS}
    if cookie:
        headers["Cookie"] = cookie
        # When authenticated, referer typically set
        headers["Referer"] = NUMBERS_PAGE

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
            await asyncio.sleep(1)

    raise HTTPException(
        status_code=502,
        detail=f"Fetch failed after {retries + 1} attempts: {last_err}",
    )


# ==================================================================
# ============ PARSERS ===========================================
# ==================================================================
def parse_numbers(html: str) -> List[dict]:
    """Parse phone numbers from listing page."""
    soup = BeautifulSoup(html, "html.parser")
    numbers = []
    seen = set()

    for card in soup.find_all("article", class_="number-card"):
        data_href = card.get("data-href", "")
        m = re.match(
            r"https?://temp-number\.com/temporary-numbers/([^/]+)/(\d+)",
            data_href,
        )
        if not m:
            continue

        country_slug = m.group(1)
        phone = m.group(2)
        if phone in seen:
            continue
        seen.add(phone)

        country_el = card.find("span", class_="number-card__country-name")
        country = country_el.get_text(strip=True) if country_el else country_slug.replace("-", " ").title()

        cc = ""
        flag_img = card.find("img")
        if flag_img and flag_img.get("src"):
            fm = re.search(r"flags/4x3/([a-z]{2})\.svg", flag_img["src"])
            if fm:
                cc = fm.group(1).upper()

        date_el = card.find("time", class_="number-card__date")
        added_at = date_el.get_text(strip=True) if date_el else ""

        msgs_el = card.find("span", class_="number-card__msgs")
        msg_text = msgs_el.get_text(strip=True) if msgs_el else ""
        count_match = re.search(r"(\d+)", msg_text)
        message_count = int(count_match.group(1)) if count_match else 0

        is_new = bool(card.find("span", class_="pill--new"))

        numbers.append({
            "phone_number": phone,
            "friendly_number": f"+{phone}",
            "country": country,
            "country_slug": country_slug,
            "country_code": cc,
            "message_count": message_count,
            "is_new": is_new,
            "added_at": added_at,
            "detail_url": data_href,
        })

    return numbers


def parse_countries(html: str) -> List[dict]:
    """Parse countries from listing page."""
    soup = BeautifulSoup(html, "html.parser")
    countries = []
    seen = set()

    for card in soup.find_all("a", class_="country-card"):
        if "country-card--more" in card.get("class", []):
            continue
        href = card.get("href", "")
        m = re.search(r"/countries/([^/?#]+)", href)
        if not m:
            continue
        slug = m.group(1)
        if slug in seen:
            continue
        seen.add(slug)

        name_el = card.find("span", class_="country-card__name")
        name = name_el.get_text(strip=True) if name_el else slug.replace("-", " ").title()

        cc = ""
        flag_img = card.find("img")
        if flag_img and flag_img.get("src"):
            fm = re.search(r"/([a-z]{2})\.svg", flag_img["src"])
            if fm:
                cc = fm.group(1).upper()

        countries.append({
            "name": name,
            "slug": slug,
            "country_code": cc,
            "url": href,
        })

    return countries


def parse_number_metadata(html: str) -> dict:
    """Extract number metadata from detail page (works without auth)."""
    soup = BeautifulSoup(html, "html.parser")
    meta = {}

    # Title
    title_el = soup.find("title")
    meta["page_title"] = title_el.get_text(strip=True) if title_el else ""

    # Phone display
    phone_el = soup.find("span", id="phone-number-display")
    if phone_el:
        # Prefer data-full-number if available
        meta["phone_display"] = phone_el.get("data-full-number") or phone_el.get_text(strip=True)

    # Stats (status, today, total, listed)
    stats = {}
    stats_dl = soup.find("dl", class_="ni-stats")
    if stats_dl:
        for div in stats_dl.find_all("div"):
            dt = div.find("dt")
            dd = div.find("dd")
            if dt and dd:
                key = dt.get_text(strip=True).lower().replace(" ", "_")
                stats[key] = dd.get_text(strip=True)
    meta["stats"] = stats

    # Number hero info
    country_el = soup.find("span", class_="ni-num__country")
    if country_el:
        meta["country"] = country_el.get_text(strip=True)

    return meta


def parse_messages(html: str) -> List[dict]:
    """
    Parse SMS messages from detail page (only works when authenticated).
    Tries multiple class patterns since we don't have authenticated HTML sample.
    """
    soup = BeautifulSoup(html, "html.parser")
    messages = []

    # Known message container patterns for temp-number.com
    patterns = [
        # Possible classes when authenticated
        ("div", "ni-msg"),
        ("div", "sms-msg"),
        ("div", "message-card"),
        ("div", "sms-card"),
        ("div", "message-item"),
        ("div", "inbox-message"),
        ("div", "message"),
        ("article", "ni-msg"),
        ("article", "sms-msg"),
    ]

    containers = []
    for tag, cls in patterns:
        found = soup.find_all(tag, class_=cls)
        if found:
            containers = found
            break

    if not containers:
        # Fallback: any div with data-message-id attribute
        containers = soup.find_all(attrs={"data-message-id": True})

    for card in containers:
        text = card.get_text(separator="\n", strip=True)
        if not text or len(text) < 5:
            continue

        # Extract sender
        sender = "unknown"
        from_el = card.find(class_=re.compile(r"(from|sender|origin)", re.I))
        if from_el:
            sender = from_el.get_text(strip=True)
        else:
            m = re.search(r"From[:\s]+([+\d\s\-()]+)", text)
            if m:
                sender = m.group(1).strip()

        # Extract received time
        received = ""
        time_el = card.find("time")
        if time_el:
            received = time_el.get_text(strip=True)
            if time_el.get("datetime"):
                received = time_el["datetime"]

        # Extract OTP codes — from .otp-code elements or regex
        otps = []
        for otp_el in card.find_all(class_="otp-code"):
            code = otp_el.get("data-clipboard-text") or otp_el.get_text(strip=True)
            code = re.sub(r"\D", "", code)
            if code:
                otps.append(code)

        if not otps:
            otps = list(set(re.findall(r"\b\d{4,8}\b", text)))

        # Body
        body_el = card.find(class_=re.compile(r"(body|content|text|message-body)", re.I))
        body = body_el.get_text(separator="\n", strip=True) if body_el else text

        messages.append({
            "from": sender,
            "body": body,
            "received_at": received,
            "otp_codes": otps,
        })

    return messages


def detect_auth_required(html: str) -> bool:
    """Check if the page requires login."""
    soup = BeautifulSoup(html, "html.parser")
    gate = soup.find("div", class_="ni-gate")
    if gate:
        return True
    if "Sign in to view" in html or "require a free account" in html:
        return True
    return False


def extract_gate_message(html: str) -> dict:
    """Extract info from the auth gate."""
    soup = BeautifulSoup(html, "html.parser")
    gate = soup.find("div", class_="ni-gate")
    if not gate:
        return {}

    info = {}

    h3 = gate.find("h3")
    if h3:
        info["title"] = h3.get_text(strip=True)

    p = gate.find("p")
    if p:
        info["message"] = p.get_text(strip=True)

    number_el = gate.find(class_="ni-gate__number")
    if number_el:
        info["number"] = number_el.get_text(strip=True)

    return info


def extract_page_meta(html: str) -> dict:
    """Extract total numbers/page info."""
    meta = {}
    m = re.search(r"([\d,]+)\s+numbers\s+available", html)
    if m:
        meta["total_numbers"] = int(m.group(1).replace(",", ""))

    m = re.search(r'aria-current=page>(\d+)</a>', html)
    if m:
        meta["current_page"] = int(m.group(1))

    last = 0
    for m in re.finditer(r'/temporary-numbers/(\d+)"', html):
        n = int(m.group(1))
        if n > last:
            last = n
    if last:
        meta["total_pages"] = last

    return meta


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "2.0.0",
        "source": "temp-number.com",
        "endpoints": {
            "numbers": "/api/numbers?page=1&country=Belgium",
            "countries": "/api/countries",
            "sms": "/api/sms/{country_slug}/{phone}?cookie=YOUR_SESSION_COOKIE",
            "raw": "/api/raw?url=...&cookie=...",
            "health": "/api/health",
            "debug": "/api/debug",
        },
        "note": "SMS messages require authentication. Pass your temp-number.com cookies via ?cookie= parameter.",
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
                "source_size": len(r.text),
            }
    except Exception as e:
        return {"success": False, "status": "unreachable", "error": str(e)[:200]}


# ==================================================================
# ============ DEBUG ==============================================
# ==================================================================
@app.get("/api/debug", tags=["System"])
async def debug(cookie: str = Query(None)):
    """Check connectivity and parser status."""
    try:
        html = await fetch(NUMBERS_PAGE)
        numbers = parse_numbers(html)
        countries = parse_countries(html)
        meta = extract_page_meta(html)

        return {
            "success": True,
            "html_size": len(html),
            "numbers_found": len(numbers),
            "countries_found": len(countries),
            "meta": meta,
            "sample_number": numbers[0] if numbers else None,
            "sample_country": countries[0] if countries else None,
        }
    except HTTPException as e:
        return {"success": False, "error": e.detail}


# ==================================================================
# ============ RAW HTML (Debug) ==================================
# ==================================================================
@app.get("/api/raw", tags=["System"])
async def raw_html(
    url: str = Query(..., description="URL to fetch"),
    cookie: str = Query(None, description="Optional cookie header"),
    limit: int = Query(5000, ge=100, le=100000),
):
    """Fetch raw HTML for debugging. Pass cookie to test authenticated requests."""
    html = await fetch(url, cookie=cookie)
    return {
        "success": True,
        "url": url,
        "has_auth": bool(cookie),
        "size": len(html),
        "auth_required": detect_auth_required(html),
        "preview": html[:limit],
    }


# ==================================================================
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    page: int = Query(1, ge=1),
    country: str = Query(None, description="Country name (Belgium, United States)"),
    per_page: int = Query(20, ge=10, le=50),
    new_only: bool = Query(False),
):
    """List temporary phone numbers with filters."""
    url = NUMBERS_PAGE
    params = []
    if page > 1:
        url = f"{NUMBERS_PAGE}/{page}"
    if per_page != 20:
        params.append(f"per_page={per_page}")
    if country:
        params.append(f"country={country}")
    if new_only:
        params.append("new24=1")
    if params:
        url += "?" + "&".join(params)

    html = await fetch(url)
    numbers = parse_numbers(html)
    meta = extract_page_meta(html)

    return {
        "success": True,
        "page": page,
        "per_page": per_page,
        "count": len(numbers),
        "total": meta.get("total_numbers"),
        "total_pages": meta.get("total_pages"),
        "filter": {"country": country, "new_only": new_only},
        "numbers": numbers,
    }


# ==================================================================
# ============ LIST COUNTRIES ====================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    """List all available countries."""
    html = await fetch(NUMBERS_PAGE)
    countries = parse_countries(html)
    return {
        "success": True,
        "count": len(countries),
        "countries": countries,
    }


# ==================================================================
# ============ SMS MESSAGES ======================================
# ==================================================================
@app.get("/api/sms/{country_slug}/{phone_number}", tags=["SMS"])
async def get_sms(
    country_slug: str,
    phone_number: str,
    cookie: str = Query(
        None,
        description="Your temp-number.com session cookie (from browser DevTools → Network → Copy as cURL)"
    ),
    otp_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Fetch SMS messages. Requires authentication cookie for most countries.

    **How to get your cookie:**
    1. Login to temp-number.com in browser
    2. Open DevTools (F12) → Network tab
    3. Refresh page and click any request to temp-number.com
    4. Find `Cookie` header → copy entire value
    5. Pass as ?cookie=<value> (URL-encoded)

    Example cookie: `XSRF-TOKEN=xxx; tempnumber-session=yyy`
    """
    phone = re.sub(r"\D", "", phone_number)
    slug = country_slug.strip().lower().replace(" ", "-")

    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    detail_url = f"{BASE_URL}/temporary-numbers/{slug}/{phone}"
    html = await fetch(detail_url, cookie=cookie)

    # Metadata (always available)
    meta = parse_number_metadata(html)

    # Check if auth gate is present
    auth_required = detect_auth_required(html)
    gate_info = extract_gate_message(html)

    # Try to parse messages
    messages = parse_messages(html)

    if otp_only:
        messages = [m for m in messages if m["otp_codes"]]

    messages = messages[:limit]

    # Build response
    response = {
        "success": True,
        "country_slug": slug,
        "phone_number": phone,
        "friendly_number": f"+{phone}",
        "url": detail_url,
        "authenticated": not auth_required,
        "auth_required": auth_required,
        "metadata": meta,
        "otp_only": otp_only,
        "count": len(messages),
        "messages": messages,
    }

    # Add helpful info if auth required
    if auth_required and not messages:
        response["info"] = {
            "gate": gate_info,
            "hint": "Pass ?cookie=YOUR_SESSION_COOKIE to view messages.",
            "how_to_get_cookie": [
                "1. Login at https://temp-number.com/login",
                "2. Open DevTools (F12) → Network",
                "3. Refresh page, click any request",
                "4. Find 'Cookie' header, copy entire value",
                "5. URL-encode and pass as ?cookie="
            ],
        }

    return response


# ==================================================================
# ============ LATEST OTP ========================================
# ==================================================================
@app.get("/api/latest-otp", tags=["SMS"])
async def latest_otp(
    cookie: str = Query(None),
    limit: int = Query(20, ge=1, le=100),
    country: str = Query(None),
    scan: int = Query(3, ge=1, le=10),
):
    """Latest OTPs across multiple numbers. Requires cookie for message access."""
    url = NUMBERS_PAGE
    if country:
        url += f"?country={country}"

    list_html = await fetch(url)
    numbers = parse_numbers(list_html)
    numbers = numbers[:scan]

    all_otps = []
    auth_blocked = 0

    for n in numbers:
        try:
            detail_html = await fetch(n["detail_url"], cookie=cookie)
            if detect_auth_required(detail_html):
                auth_blocked += 1
                continue
            messages = parse_messages(detail_html)
            for m in messages:
                if m["otp_codes"]:
                    all_otps.append({
                        "phone_number": n["phone_number"],
                        "friendly_number": n["friendly_number"],
                        "country": n["country"],
                        "from": m["from"],
                        "body": m["body"][:300],
                        "otp_codes": m["otp_codes"],
                        "received_at": m["received_at"],
                    })
        except Exception:
            continue

    all_otps = all_otps[:limit]

    response = {
        "success": True,
        "scanned": len(numbers),
        "auth_blocked": auth_blocked,
        "count": len(all_otps),
        "country_filter": country,
        "otps": all_otps,
    }

    if auth_blocked and not all_otps:
        response["hint"] = "Most numbers require auth. Pass ?cookie=YOUR_SESSION_COOKIE"

    return response


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

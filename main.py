from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import httpx
import re
import asyncio
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Temporary phone numbers + SMS reader — powered by temp-number.com",
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
BASE_URL = "https://temp-number.com"
NUMBERS_PAGE = f"{BASE_URL}/temporary-numbers"

HEADERS = {
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
MAX_RETRIES = 2


# ==================================================================
# ============ HTTP FETCH WITH RETRY =============================
# ==================================================================
async def fetch(url: str, retries: int = MAX_RETRIES) -> str:
    """Fetch HTML with retry logic."""
    last_err = None

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers=HEADERS,
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
    """
    Parse phone numbers from temp-number.com listing page.

    HTML structure:
    <article class="number-card" data-href=".../{country}/{phone}">
      <span class="number-card__country-name">Belgium</span>
      <img src=".../flags/4x3/be.svg">
      <span class="pill--new">NEW</span>
      <time class="number-card__date">40 minutes ago</time>
      <span class="number-card__number">32468798009</span>
      <span class="number-card__msgs">3 messages</span>
    </article>
    """
    soup = BeautifulSoup(html, "html.parser")
    numbers = []
    seen = set()

    for card in soup.find_all("article", class_="number-card"):
        data_href = card.get("data-href", "")

        # Extract phone + country from URL
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

        # Country name
        country_el = card.find("span", class_="number-card__country-name")
        country = country_el.get_text(strip=True) if country_el else country_slug.replace("-", " ").title()

        # Country code from flag image
        cc = ""
        flag_img = card.find("img")
        if flag_img and flag_img.get("src"):
            flag_match = re.search(r"flags/4x3/([a-z]{2})\.svg", flag_img["src"])
            if flag_match:
                cc = flag_match.group(1).upper()

        # Date
        date_el = card.find("time", class_="number-card__date")
        added_at = date_el.get_text(strip=True) if date_el else ""

        # Message count
        msgs_el = card.find("span", class_="number-card__msgs")
        msg_text = msgs_el.get_text(strip=True) if msgs_el else ""
        count_match = re.search(r"(\d+)", msg_text)
        message_count = int(count_match.group(1)) if count_match else 0

        # Is new?
        is_new = bool(card.find("span", class_="pill--new"))

        # Friendly number from country code
        friendly = f"+{phone}"

        numbers.append({
            "phone_number": phone,
            "friendly_number": friendly,
            "country": country,
            "country_slug": country_slug,
            "country_code": cc,
            "message_count": message_count,
            "is_new": is_new,
            "added_at": added_at,
            "detail_url": data_href,
            "sms_api_url": f"/api/sms/{country_slug}/{phone}",
        })

    return numbers


def parse_countries(html: str) -> List[dict]:
    """
    Parse available countries from listing page.

    HTML structure:
    <a href="https://temp-number.com/countries/united-states" class="country-card">
      <span class="country-card__flag"><img src=".../us.svg"></span>
      <span class="country-card__name">United States</span>
    </a>
    """
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

        # Country code
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


def parse_messages(html: str) -> List[dict]:
    """
    Parse SMS messages from temp-number.com detail page.
    Tries multiple known class patterns.
    """
    soup = BeautifulSoup(html, "html.parser")
    messages = []

    # Try known message card classes
    possible_classes = [
        "message-card", "sms-card", "message", "sms-item",
        "sms-message", "message-item", "inbox-message",
    ]

    containers = []
    for cls in possible_classes:
        found = soup.find_all("div", class_=cls)
        if found:
            containers = found
            break

    # Fallback: look for any div containing "from" and body
    if not containers:
        # Look for cards with common heading patterns
        containers = soup.find_all("div", class_=re.compile(r"(message|sms)", re.I))

    for card in containers:
        text_content = card.get_text(separator="\n", strip=True)
        if not text_content or len(text_content) < 5:
            continue

        # Try to find sender
        sender = "unknown"
        for pattern in [
            r"From[:\s]+([+\d\s\-()]+)",
            r"from[:\s]+([+\d\s\-()]+)",
        ]:
            m = re.search(pattern, text_content)
            if m:
                sender = m.group(1).strip()
                break

        # Try to find time
        received = ""
        time_el = card.find("time")
        if time_el:
            received = time_el.get_text(strip=True)
        else:
            m_time = re.search(
                r"(\d+\s+(?:second|minute|hour|day)s?\s+ago|"
                r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2})",
                text_content,
            )
            if m_time:
                received = m_time.group(1)

        # Extract OTP codes
        otps = list(set(re.findall(r"\b\d{4,8}\b", text_content)))

        # Clean body: remove header lines
        body_lines = []
        for line in text_content.split("\n"):
            if re.match(r"^(From|Received|Time|Date)[:\s]", line):
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

        if not body:
            continue

        messages.append({
            "from": sender,
            "body": body,
            "received_at": received,
            "otp_codes": otps,
        })

    return messages


def extract_page_meta(html: str) -> dict:
    """Extract total count and page info."""
    meta = {
        "total_numbers": None,
        "current_page": None,
        "total_pages": None,
    }

    # Total numbers: "10,410 numbers available"
    m = re.search(r"([\d,]+)\s+numbers\s+available", html)
    if m:
        meta["total_numbers"] = int(m.group(1).replace(",", ""))

    # Current page
    m = re.search(r'aria-current=page>(\d+)</a>', html)
    if m:
        meta["current_page"] = int(m.group(1))

    # Last page (from pagination)
    last_page = 0
    for m in re.finditer(r'/temporary-numbers/(\d+)"', html):
        page_num = int(m.group(1))
        if page_num > last_page:
            last_page = page_num
    if last_page:
        meta["total_pages"] = last_page

    return meta


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "1.0.0",
        "source": "temp-number.com",
        "endpoints": {
            "numbers": "/api/numbers?page=1&country=Belgium",
            "countries": "/api/countries",
            "sms": "/api/sms/{country_slug}/{phone}",
            "latest_otp": "/api/latest-otp?limit=20",
            "health": "/api/health",
            "debug": "/api/debug",
        },
    }


# ==================================================================
# ============ HEALTH =============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
            r = await client.get(NUMBERS_PAGE)
            return {
                "success": True,
                "status": "healthy" if r.status_code == 200 else "degraded",
                "source_status": r.status_code,
                "source_size": len(r.text),
            }
    except Exception as e:
        return {
            "success": False,
            "status": "unreachable",
            "error": str(e)[:200],
        }


# ==================================================================
# ============ DEBUG ==============================================
# ==================================================================
@app.get("/api/debug", tags=["System"])
async def debug():
    """Check connectivity and parse sanity."""
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
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    page: int = Query(1, ge=1, description="Page number"),
    country: str = Query(None, description="Filter by country name (e.g. Belgium)"),
    per_page: int = Query(20, ge=10, le=50, description="Numbers per page"),
    new_only: bool = Query(False, description="Only show numbers added in last 24h"),
):
    """
    List temporary phone numbers.
    Supports pagination and country filtering.
    """
    # Build URL with page/per_page/country
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
        "filter": {
            "country": country,
            "new_only": new_only,
        },
        "numbers": numbers,
    }


# ==================================================================
# ============ LIST COUNTRIES ====================================
# ==================================================================
@app.get("/api/countries", tags=["SMS"])
async def list_countries():
    """List all available countries with temporary numbers."""
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
    otp_only: bool = Query(False, description="Only return messages with OTP codes"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Fetch SMS messages for a specific phone number.

    Example: /api/sms/belgium/32468798009
    """
    # Clean inputs
    phone = re.sub(r"\D", "", phone_number)
    slug = country_slug.strip().lower().replace(" ", "-")

    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    detail_url = f"{BASE_URL}/temporary-numbers/{slug}/{phone}"
    html = await fetch(detail_url)

    # Extract number info from the page
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Parse messages
    messages = parse_messages(html)

    if otp_only:
        messages = [m for m in messages if m["otp_codes"]]

    messages = messages[:limit]

    return {
        "success": True,
        "country_slug": slug,
        "phone_number": phone,
        "page_title": title,
        "url": detail_url,
        "otp_only": otp_only,
        "count": len(messages),
        "messages": messages,
    }


# ==================================================================
# ============ LATEST OTP ========================================
# ==================================================================
@app.get("/api/latest-otp", tags=["SMS"])
async def latest_otp(
    limit: int = Query(20, ge=1, le=100),
    country: str = Query(None, description="Filter by country name"),
    scan_numbers: int = Query(5, ge=1, le=20, description="How many numbers to scan"),
):
    """
    Get latest OTP codes across multiple numbers.
    Fetches the first N numbers' detail pages and extracts OTP codes.
    """
    # Build list URL
    url = NUMBERS_PAGE
    if country:
        url += f"?country={country}"

    list_html = await fetch(url)
    numbers = parse_numbers(list_html)
    numbers = numbers[:scan_numbers]

    all_otps = []
    errors = 0

    for n in numbers:
        try:
            detail_html = await fetch(n["detail_url"])
            messages = parse_messages(detail_html)
            for m in messages:
                if m["otp_codes"]:
                    all_otps.append({
                        "phone_number": n["phone_number"],
                        "friendly_number": n["friendly_number"],
                        "country": n["country"],
                        "country_code": n["country_code"],
                        "from": m["from"],
                        "body": m["body"][:300],
                        "otp_codes": m["otp_codes"],
                        "received_at": m["received_at"],
                    })
        except Exception:
            errors += 1

    all_otps = all_otps[:limit]

    return {
        "success": True,
        "scanned_numbers": len(numbers),
        "errors": errors,
        "count": len(all_otps),
        "country_filter": country,
        "otps": all_otps,
    }


# ==================================================================
# ============ RAW HTML (Debug helper) ===========================
# ==================================================================
@app.get("/api/raw", tags=["System"])
async def raw_html(
    url: str = Query(..., description="URL to fetch"),
    limit: int = Query(5000, ge=100, le=50000),
):
    """Fetch raw HTML for debugging parsers."""
    html = await fetch(url)
    return {
        "success": True,
        "url": url,
        "size": len(html),
        "preview": html[:limit],
    }


# ============ ERROR HANDLERS ============
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
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

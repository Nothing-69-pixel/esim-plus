from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import httpx
import re
import asyncio

app = FastAPI(
    title="Tap SMS API",
    description="Temporary phone numbers + SMS — public/private auto-detect",
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

BASE_URL = "https://temp-number.com"
NUMBERS_PAGE = f"{BASE_URL}/temporary-numbers"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; M2103K19I Build/TP1A.220624.014) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.8010.36 Mobile Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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
MAX_CONCURRENT = 5  # Concurrent detail-page fetches

# ── Cache for scan results (per-lambda-instance, cheap) ───────────
PUBLIC_NUMBERS_CACHE: Dict[str, dict] = {}


# ==================================================================
# ============ HTTP FETCH ========================================
# ==================================================================
async def fetch(url: str, cookie: Optional[str] = None, retries: int = 2) -> str:
    headers = {**HEADERS}
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
            await asyncio.sleep(0.6)

    raise HTTPException(status_code=502, detail=f"Fetch failed: {last_err}")


# ==================================================================
# ============ PARSERS ===========================================
# ==================================================================
def parse_numbers(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    numbers = []
    seen = set()

    for card in soup.find_all("article", class_="number-card"):
        href = card.get("data-href", "")
        m = re.match(r"https?://temp-number\.com/temporary-numbers/([^/]+)/(\d+)", href)
        if not m:
            continue
        slug, phone = m.group(1), m.group(2)
        if phone in seen:
            continue
        seen.add(phone)

        country_el = card.find("span", class_="number-card__country-name")
        country = country_el.get_text(strip=True) if country_el else slug.replace("-", " ").title()

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
            "country_slug": slug,
            "country_code": cc,
            "message_count": message_count,
            "is_new": is_new,
            "added_at": added_at,
            "detail_url": href,
        })

    return numbers


def parse_countries(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    countries, seen = [], set()

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

        countries.append({"name": name, "slug": slug, "country_code": cc, "url": href})

    return countries


def parse_messages(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    messages = []

    # Known message container selectors
    selectors = [
        "div.ni-msg",
        "div.ni-message",
        "div.ni-inbox__item",
        "div.ni-inbox__msg",
        "div.ni-msg-item",
        "article.ni-msg",
        "div.sms-item",
        "div.sms-message",
        "div.message-item",
        "div[data-message-id]",
        "div[data-sms-id]",
    ]

    containers = []
    for sel in selectors:
        found = soup.select(sel)
        if found:
            containers = found
            break

    if not containers:
        containers = soup.find_all(
            "div",
            class_=re.compile(r"(?:^|\s)(?:msg|sms|message)(?:\s|$|__|--)", re.I),
        )

    for card in containers:
        text = card.get_text(separator="\n", strip=True)
        if not text or len(text) < 3:
            continue

        sender = "unknown"
        from_el = card.find(class_=re.compile(r"(from|sender|origin)", re.I))
        if from_el:
            sender = from_el.get_text(strip=True)
        else:
            m = re.search(r"(?:From|Sender)[:\s]+([+\d\s\-()A-Za-z]+)", text, re.I)
            if m:
                sender = m.group(1).strip()

        received = ""
        time_el = card.find("time")
        if time_el:
            received = time_el.get("datetime") or time_el.get_text(strip=True)
        else:
            m = re.search(
                r"(\d+\s+(?:second|minute|hour|day)s?\s+ago|"
                r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2})",
                text,
            )
            if m:
                received = m.group(1)

        otps = []
        for otp_el in card.find_all(class_="otp-code"):
            code = otp_el.get("data-clipboard-text") or otp_el.get_text(strip=True)
            code = re.sub(r"\D", "", code)
            if code:
                otps.append(code)
        if not otps:
            otps = sorted(set(re.findall(r"\b\d{4,8}\b", text)))

        body_el = card.find(
            class_=re.compile(r"(body|content|text|message-body|sms-body)", re.I)
        )
        body = body_el.get_text(separator="\n", strip=True) if body_el else text

        messages.append({
            "from": sender,
            "body": body,
            "received_at": received,
            "otp_codes": otps,
        })

    return messages


def extract_gate(html: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")
    gate = soup.find("div", class_="ni-gate")
    if not gate:
        # Also check for text-based gate indicators
        if "Sign in to view" in html or "require a free account" in html:
            return {"title": "Login required", "message": "Sign in to view messages"}
        return None

    info = {}
    h3 = gate.find("h3")
    if h3:
        info["title"] = h3.get_text(strip=True)
    p = gate.find("p")
    if p:
        info["message"] = p.get_text(strip=True)
    num = gate.find(class_="ni-gate__number")
    if num:
        info["number"] = num.get_text(strip=True)
    return info


def extract_metadata(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    meta = {}
    title_el = soup.find("title")
    meta["page_title"] = title_el.get_text(strip=True) if title_el else ""

    stats = {}
    dl = soup.find("dl", class_="ni-stats")
    if dl:
        for div in dl.find_all("div"):
            dt, dd = div.find("dt"), div.find("dd")
            if dt and dd:
                key = dt.get_text(strip=True).lower().replace(" ", "_")
                stats[key] = dd.get_text(strip=True)
    meta["stats"] = stats
    return meta


def extract_page_meta(html: str) -> dict:
    meta = {}
    m = re.search(r"([\d,]+)\s+numbers\s+available", html)
    if m:
        meta["total_numbers"] = int(m.group(1).replace(",", ""))
    last = 0
    for m in re.finditer(r"/temporary-numbers/(\d+)\"", html):
        n = int(m.group(1))
        if n > last:
            last = n
    if last:
        meta["total_pages"] = last
    return meta


# ==================================================================
# ============ CORE: Check if a number is public =================
# ==================================================================
async def check_number_status(num: dict, cookie: Optional[str] = None) -> dict:
    """
    Check if a single number is public or gated.
    Returns enriched dict with 'is_public', 'messages', 'message_count_actual'.
    """
    try:
        html = await fetch(num["detail_url"], cookie=cookie)
        gate = extract_gate(html)
        messages = parse_messages(html)
        metadata = extract_metadata(html)

        return {
            **num,
            "is_public": gate is None,
            "requires_login": gate is not None,
            "messages": messages,
            "actual_message_count": len(messages),
            "metadata": metadata,
            "gate": gate,
        }
    except Exception as e:
        return {
            **num,
            "is_public": False,
            "requires_login": None,
            "messages": [],
            "actual_message_count": 0,
            "error": str(e)[:200],
        }


async def check_batch(numbers: List[dict], cookie: Optional[str] = None) -> List[dict]:
    """Check multiple numbers concurrently."""
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def limited(n):
        async with sem:
            return await check_number_status(n, cookie)

    tasks = [limited(n) for n in numbers]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "4.0.0",
        "source": "temp-number.com",
        "endpoints": {
            "numbers": "/api/numbers",
            "public_numbers": "/api/public-numbers",           # ⭐ Only public
            "check_status": "/api/check-status/{slug}/{phone}", # ⭐ Single number check
            "scan_all": "/api/scan-all?pages=3",                # ⭐ Full scan
            "scan_public": "/api/scan-public?pages=3",          # ⭐ Scan + return only public
            "sms": "/api/sms/{slug}/{phone}",
            "latest_otp": "/api/latest-otp",
            "countries": "/api/countries",
            "health": "/api/health",
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
            return {"success": True, "source_status": r.status_code}
    except Exception as e:
        return {"success": False, "error": str(e)[:200]}


# ==================================================================
# ============ LIST NUMBERS ======================================
# ==================================================================
@app.get("/api/numbers", tags=["Numbers"])
async def list_numbers(
    page: int = Query(1, ge=1),
    country: str = Query(None),
    per_page: int = Query(20, ge=10, le=50),
    new_only: bool = Query(False),
):
    url = NUMBERS_PAGE if page == 1 else f"{NUMBERS_PAGE}/{page}"
    params = []
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
        "count": len(numbers),
        "total": meta.get("total_numbers"),
        "total_pages": meta.get("total_pages"),
        "numbers": numbers,
    }


# ==================================================================
# ============ COUNTRIES =========================================
# ==================================================================
@app.get("/api/countries", tags=["Numbers"])
async def list_countries():
    html = await fetch(NUMBERS_PAGE)
    countries = parse_countries(html)
    return {"success": True, "count": len(countries), "countries": countries}


# ==================================================================
# ============ ⭐ CHECK STATUS (single number) ===================
# ==================================================================
@app.get("/api/check-status/{country_slug}/{phone_number}", tags=["Status"])
async def check_status(
    country_slug: str,
    phone_number: str,
    cookie: str = Query(None, description="Optional session cookie for gated check"),
):
    """
    Check if a single number is public or requires login.

    Returns:
    - is_public: true if messages visible without login
    - requires_login: true if gated
    - message_count: actual visible messages
    - messages: list (if public)
    """
    phone = re.sub(r"\D", "", phone_number)
    slug = country_slug.strip().lower().replace(" ", "-")

    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    num = {
        "phone_number": phone,
        "friendly_number": f"+{phone}",
        "country_slug": slug,
        "detail_url": f"{BASE_URL}/temporary-numbers/{slug}/{phone}",
    }

    result = await check_number_status(num, cookie)

    return {
        "success": True,
        "phone_number": phone,
        "is_public": result["is_public"],
        "requires_login": result["requires_login"],
        "message_count": result["actual_message_count"],
        "metadata": result.get("metadata", {}),
        "gate": result.get("gate"),
        "messages": result.get("messages", []),
    }


# ==================================================================
# ============ ⭐ SCAN ALL — categorize every number =============
# ==================================================================
@app.get("/api/scan-all", tags=["Scan"])
async def scan_all(
    pages: int = Query(1, ge=1, le=20, description="Pages to scan (20 per page)"),
    cookie: str = Query(None),
    only_public: bool = Query(False, description="Return only public numbers"),
    only_with_messages: bool = Query(False, description="Only numbers with actual messages"),
    country: str = Query(None, description="Filter by country name"),
):
    """
    Scan listing pages, check EVERY number, categorize as public/gated.

    - `pages=3` → scans 60 numbers (3 × 20)
    - `only_public=true` → returns only public ones
    - `only_with_messages=true` → returns only numbers that have actual SMS
    """
    all_numbers = []

    # Collect from listing pages
    for page in range(1, pages + 1):
        url = NUMBERS_PAGE if page == 1 else f"{NUMBERS_PAGE}/{page}"
        if country:
            url += f"?country={country}"
        try:
            html = await fetch(url)
            page_numbers = parse_numbers(html)
            all_numbers.extend(page_numbers)
        except Exception:
            continue

    if not all_numbers:
        return {
            "success": False,
            "message": "No numbers found",
        }

    # Check all concurrently
    checked = await check_batch(all_numbers, cookie)

    # Filter
    result_numbers = checked
    if only_public:
        result_numbers = [n for n in result_numbers if n["is_public"]]
    if only_with_messages:
        result_numbers = [n for n in result_numbers if n["actual_message_count"] > 0]

    # Categorize
    public = [n for n in checked if n["is_public"]]
    gated = [n for n in checked if n["requires_login"]]
    failed = [n for n in checked if n.get("error")]

    # Clean output (strip full messages unless it's the final result)
    def slim(n, include_messages=False):
        out = {
            "phone_number": n["phone_number"],
            "friendly_number": n.get("friendly_number"),
            "country": n.get("country"),
            "country_slug": n.get("country_slug"),
            "country_code": n.get("country_code"),
            "is_public": n["is_public"],
            "requires_login": n.get("requires_login"),
            "actual_message_count": n["actual_message_count"],
            "listed_count": n.get("message_count", 0),
            "detail_url": n.get("detail_url"),
        }
        if n.get("error"):
            out["error"] = n["error"]
        if include_messages:
            out["messages"] = n.get("messages", [])
        return out

    return {
        "success": True,
        "scanned": len(checked),
        "public_count": len(public),
        "gated_count": len(gated),
        "failed_count": len(failed),
        "filter": {
            "only_public": only_public,
            "only_with_messages": only_with_messages,
            "country": country,
        },
        "returned_count": len(result_numbers),
        "numbers": [slim(n, include_messages=only_with_messages or len(result_numbers) <= 30) for n in result_numbers],
    }


# ==================================================================
# ============ ⭐ SCAN PUBLIC — only public numbers ==============
# ==================================================================
@app.get("/api/scan-public", tags=["Scan"])
async def scan_public(
    pages: int = Query(2, ge=1, le=10),
    cookie: str = Query(None),
    min_messages: int = Query(0, ge=0, description="Min actual messages to include"),
    country: str = Query(None),
):
    """
    Quick way to get only PUBLIC numbers with messages.
    Skips gated numbers entirely from response.
    """
    all_numbers = []
    for page in range(1, pages + 1):
        url = NUMBERS_PAGE if page == 1 else f"{NUMBERS_PAGE}/{page}"
        if country:
            url += f"?country={country}"
        try:
            html = await fetch(url)
            all_numbers.extend(parse_numbers(html))
        except Exception:
            continue

    if not all_numbers:
        return {"success": False, "message": "No numbers found"}

    checked = await check_batch(all_numbers, cookie)
    public = [n for n in checked if n["is_public"] and n["actual_message_count"] >= min_messages]

    return {
        "success": True,
        "scanned": len(checked),
        "public_count": len(public),
        "min_messages": min_messages,
        "numbers": [
            {
                "phone_number": n["phone_number"],
                "friendly_number": n["friendly_number"],
                "country": n["country"],
                "country_slug": n["country_slug"],
                "country_code": n["country_code"],
                "message_count": n["actual_message_count"],
                "detail_url": n["detail_url"],
                "sms_api_url": f"/api/sms/{n['country_slug']}/{n['phone_number']}",
            }
            for n in public
        ],
    }


# ==================================================================
# ============ ⭐ PUBLIC NUMBERS (from cache or quick scan) ======
# ==================================================================
@app.get("/api/public-numbers", tags=["Scan"])
async def public_numbers(
    country: str = Query(None, description="Filter by country"),
    max_pages: int = Query(2, ge=1, le=10, description="Pages to scan before returning"),
    cookie: str = Query(None),
):
    """
    Get a list of public numbers only (skipping gated ones).
    Fast — stops when it has enough public numbers.
    """
    all_numbers = []
    for page in range(1, max_pages + 1):
        url = NUMBERS_PAGE if page == 1 else f"{NUMBERS_PAGE}/{page}"
        if country:
            url += f"?country={country}"
        try:
            html = await fetch(url)
            all_numbers.extend(parse_numbers(html))
        except Exception:
            continue

    if not all_numbers:
        return {"success": False, "message": "No numbers found"}

    checked = await check_batch(all_numbers, cookie)
    public = [n for n in checked if n["is_public"]]

    return {
        "success": True,
        "total_scanned": len(checked),
        "public_count": len(public),
        "country_filter": country,
        "public_numbers": [
            {
                "phone_number": n["phone_number"],
                "friendly_number": n["friendly_number"],
                "country": n["country"],
                "country_code": n["country_code"],
                "message_count": n["actual_message_count"],
                "detail_url": n["detail_url"],
            }
            for n in public
        ],
    }


# ==================================================================
# ============ SMS ===============================================
# ==================================================================
@app.get("/api/sms/{country_slug}/{phone_number}", tags=["SMS"])
async def get_sms(
    country_slug: str,
    phone_number: str,
    cookie: str = Query(None),
    otp_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
):
    phone = re.sub(r"\D", "", phone_number)
    slug = country_slug.strip().lower().replace(" ", "-")
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    url = f"{BASE_URL}/temporary-numbers/{slug}/{phone}"
    html = await fetch(url, cookie=cookie)

    metadata = extract_metadata(html)
    gate = extract_gate(html)
    messages = parse_messages(html)

    if otp_only:
        messages = [m for m in messages if m["otp_codes"]]

    messages = messages[:limit]

    return {
        "success": True,
        "phone_number": phone,
        "friendly_number": f"+{phone}",
        "country_slug": slug,
        "url": url,
        "is_public": gate is None,
        "auth_required": gate is not None,
        "metadata": metadata,
        "otp_only": otp_only,
        "count": len(messages),
        "messages": messages,
        "gate": gate,
    }


# ==================================================================
# ============ LATEST OTP ========================================
# ==================================================================
@app.get("/api/latest-otp", tags=["SMS"])
async def latest_otp(
    cookie: str = Query(None),
    limit: int = Query(20, ge=1, le=100),
    country: str = Query(None),
    scan: int = Query(8, ge=1, le=20),
):
    url = NUMBERS_PAGE
    if country:
        url += f"?country={country}"

    list_html = await fetch(url)
    numbers = parse_numbers(list_html)[:scan]

    if not numbers:
        return {"success": True, "count": 0, "otps": []}

    checked = await check_batch(numbers, cookie)

    all_otps = []
    for n in checked:
        if not n["is_public"]:
            continue
        for m in n.get("messages", []):
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

    return {
        "success": True,
        "scanned": len(checked),
        "public_count": len([n for n in checked if n["is_public"]]),
        "count": len(all_otps[:limit]),
        "country_filter": country,
        "otps": all_otps[:limit],
    }


# ==================================================================
# ============ RAW ==============================================
# ==================================================================
@app.get("/api/raw", tags=["System"])
async def raw_html(
    url: str = Query(...),
    cookie: str = Query(None),
    limit: int = Query(5000, ge=100, le=100000),
):
    html = await fetch(url, cookie=cookie)
    return {
        "success": True,
        "url": url,
        "size": len(html),
        "auth_gate": extract_gate(html),
        "preview": html[:limit],
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

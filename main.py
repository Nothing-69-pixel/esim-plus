
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import httpx
import re
import asyncio
import socket
from datetime import datetime, timezone

app = FastAPI(
    title="Tap SMS API",
    description="Free temporary phone numbers + SMS reader",
    version="7.0.0",
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
RECEIVE_SMSS = "https://receive-smss.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
}

TIMEOUT = 25.0


# ==================================================================
# ============ HTTP CLIENT =======================================
# ==================================================================
async def fetch(url: str, params: dict = None, retries: int = 2) -> str:
    """Fetch with retry logic."""
    last_err = None

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers=HEADERS,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                r = await client.get(url, params=params)
                if r.status_code == 200:
                    return r.text
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)[:150]
            if attempt < retries - 1:
                await asyncio.sleep(1)

    raise HTTPException(status_code=502, detail=f"Fetch failed: {last_err}")


# ==================================================================
# ============ PARSERS ===========================================
# ==================================================================
def parse_numbers(html: str) -> List[dict]:
    """Parse phone numbers from receive-smss.com homepage."""
    soup = BeautifulSoup(html, "html.parser")
    numbers = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.match(r"^/sms/(\d+)/?$", href)
        if not m:
            continue
        phone = m.group(1)
        if phone in seen:
            continue
        seen.add(phone)

        friendly = a.get_text(strip=True)
        if not friendly:
            friendly = f"+{phone}"

        # Extract country flag if present
        country = ""
        flag = a.find("span", class_=re.compile("flag-icon"))
        if flag:
            classes = flag.get("class", [])
            for c in classes:
                if c.startswith("flag-icon-"):
                    country = c.replace("flag-icon-", "").upper()
                    break

        numbers.append({
            "phone_number": phone,
            "friendly_number": friendly,
            "country": country,
            "url": f"{RECEIVE_SMSS}/sms/{phone}/",
            "api_url": f"/api/sms/{phone}",
        })

    return numbers


def parse_messages(html: str) -> List[dict]:
    """Parse SMS messages from receive-smss.com message page."""
    soup = BeautifulSoup(html, "html.parser")
    messages = []

    # Find all card divs
    for card in soup.find_all("div", class_="card"):
        header = card.find("div", class_="card-header")
        body = card.find("div", class_="card-body")
        if not header or not body:
            continue

        header_text = header.get_text(separator=" ", strip=True)
        body_text = body.get_text(separator="\n", strip=True)

        if not body_text:
            continue

        # Extract sender
        sender = "unknown"
        m_from = re.search(r"From:\s*(.+?)(?:\s+Received|$)", header_text)
        if m_from:
            sender = m_from.group(1).strip()

        # Extract received time
        received = ""
        m_recv = re.search(r"Received:\s*(.+?)$", header_text)
        if m_recv:
            received = m_recv.group(1).strip()

        # Extract OTP codes
        otps = list(set(re.findall(r"\b\d{4,8}\b", body_text)))

        messages.append({
            "from": sender,
            "body": body_text,
            "received_at": received,
            "otp_codes": otps,
        })

    return messages


# ==================================================================
# ============ ROOT ===============================================
# ==================================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "success": True,
        "service": "Tap SMS API",
        "version": "7.0.0",
        "source": "receive-smss.com",
        "endpoints": {
            "test": "/api/test",
            "health": "/api/health",
            "numbers": "/api/numbers",
            "sms": "/api/sms/{phone}",
            "latest_otp": "/api/latest-otp",
            "raw": "/api/raw?url=...",
        },
    }


# ==================================================================
# ============ TEST ==============================================
# ==================================================================
def resolve_hostname(hostname: str) -> dict:
    result = {"hostname": hostname, "ipv4": [], "error": None}
    try:
        infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        for info in infos:
            family, _, _, _, addr = info
            if family == socket.AF_INET:
                result["ipv4"].append(addr[0])
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


@app.get("/api/test", tags=["System"])
async def test():
    results = {}

    # DNS tests
    for host in ["www.google.com", "receive-smss.com", "otp-api.shelex.dev"]:
        results[f"dns_{host}"] = resolve_hostname(host)

    # HTTP tests
    async def test_url(url, name):
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
                r = await client.get(url)
                return {
                    "status": r.status_code,
                    "ok": r.status_code == 200,
                    "size": len(r.text),
                    "has_content": len(r.text) > 1000,
                }
        except Exception as e:
            return {"error": str(e)[:150]}

    results["http_receive_smss"] = await test_url(RECEIVE_SMSS, "receive-smss")

    return {"success": True, "results": results}


# ==================================================================
# ============ HEALTH =============================================
# ==================================================================
@app.get("/api/health", tags=["System"])
async def health():
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
            r = await client.get(RECEIVE_SMSS)
            return {
                "success": True,
                "status": "healthy" if r.status_code == 200 else "degraded",
                "source": RECEIVE_SMSS,
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
# ============ RAW HTML (Debug) ==================================
# ==================================================================
@app.get("/api/raw", tags=["System"])
async def raw_html(url: str = Query(...), limit: int = Query(5000)):
    """
    Fetch raw HTML from a URL for debugging.
    Example: /api/raw?url=https://receive-smss.com
    """
    html = await fetch(url)
    return {
        "success": True,
        "url": url,
        "size": len(html),
        "preview": html[:limit],
    }


# ==================================================================
# ============ NUMBERS ===========================================
# ==================================================================
@app.get("/api/numbers", tags=["SMS"])
async def list_numbers(
    country: str = Query(None, description="Filter by country code (US, CA, GB)"),
    limit: int = Query(200, ge=1, le=500),
):
    """List all temporary phone numbers."""
    html = await fetch(RECEIVE_SMSS)
    numbers = parse_numbers(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n.get("country") == cc]

    numbers = numbers[:limit]

    # Group by country
    by_country: Dict[str, int] = {}
    for n in numbers:
        cc = n.get("country") or "unknown"
        by_country[cc] = by_country.get(cc, 0) + 1

    return {
        "success": True,
        "count": len(numbers),
        "by_country": by_country,
        "numbers": numbers,
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
    """Get SMS messages for a phone number."""
    phone = re.sub(r"\D", "", phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    url = f"{RECEIVE_SMSS}/sms/{phone}/"
    html = await fetch(url)
    messages = parse_messages(html)

    if otp_only:
        messages = [m for m in messages if m["otp_codes"]]

    messages = messages[:limit]

    return {
        "success": True,
        "phone_number": phone,
        "total_loaded": len(messages),
        "otp_only": otp_only,
        "messages": messages,
    }


# ==================================================================
# ============ LATEST OTP ========================================
# ==================================================================
@app.get("/api/latest-otp", tags=["SMS"])
async def latest_otp(
    country: str = Query(None, description="Filter by country (US, CA, GB)"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get latest OTP codes across numbers."""
    html = await fetch(RECEIVE_SMSS)
    numbers = parse_numbers(html)

    if country:
        cc = country.upper()
        numbers = [n for n in numbers if n.get("country") == cc]

    # Take first 5 numbers
    numbers = numbers[:5]

    all_otps = []
    for n in numbers:
        try:
            msg_html = await fetch(n["url"])
            messages = parse_messages(msg_html)
            for m in messages:
                if m["otp_codes"]:
                    all_otps.append({
                        "phone_number": n["phone_number"],
                        "friendly_number": n["friendly_number"],
                        "country": n.get("country"),
                        "from": m["from"],
                        "body": m["body"][:200],
                        "otp_codes": m["otp_codes"],
                        "received_at": m["received_at"],
                    })
        except Exception:
            continue

    all_otps = all_otps[:limit]

    return {
        "success": True,
        "count": len(all_otps),
        "country_filter": country,
        "otps": all_otps,
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

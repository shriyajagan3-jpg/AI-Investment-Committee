"""
sec_data.py - pulls real company data from the SEC's free EDGAR APIs.

EDGAR is the SEC's database of every public company's filings. No API key
is needed, but the SEC requires a User-Agent header that identifies you
(a name and contact email). We read it from the SEC_USER_AGENT environment
variable so personal info never gets committed to GitHub.
"""

import os
import re

import requests
from bs4 import BeautifulSoup


def _headers() -> dict:
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise RuntimeError(
            "Set the SEC_USER_AGENT environment variable to 'Your Name your_email' "
            "(the SEC blocks requests without it). See README."
        )
    return {"User-Agent": agent}


def _get(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp


def get_cik(ticker: str) -> tuple[str, str]:
    """Ticker ('AAPL') -> (10-digit SEC company ID, company name).

    The SEC's other APIs expect the CIK zero-padded: 320193 -> '0000320193'.
    """
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    for row in data.values():
        if row["ticker"].upper() == ticker.strip().upper():
            return str(row["cik_str"]).zfill(10), row["title"]
    raise ValueError(f"Ticker '{ticker}' not found on SEC EDGAR.")


def get_company_facts(cik: str) -> dict:
    """Every number the company has reported in XBRL, as JSON.

    XBRL tags each line of the financial statements, so instead of scraping
    tables we get clean data like:
      facts["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = [{val, start, end, form, fp, filed}, ...]
    """
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()


def get_latest_10k_url(cik: str) -> str:
    """Link to the company's most recent 10-K (annual report)."""
    recent = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()["filings"]["recent"]
    # Parallel lists: the same index is the same filing, newest first.
    for form, accession, doc in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"]):
        if form == "10-K":
            folder = accession.replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{doc}"
    raise ValueError("No 10-K found (foreign companies file 20-F instead).")


def extract_risk_factors(html: str, max_chars: int = 15000) -> str:
    """Cut 'Item 1A. Risk Factors' out of a 10-K's HTML.

    'Item 1A' appears twice: in the table of contents (short) and as the real
    section (long). We collect every candidate span up to the next item
    heading and keep the longest one.
    """
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    # Filings often use non-breaking and zero-width spaces between "Item" and "1A".
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)

    # \W* allows any punctuation/spacing between the pieces: "ITEM 1A. RISK FACTORS",
    # "Item 1A - Risk Factors", "Item 1 A.Risk Factors", ...
    start_re = re.compile(r"item\W*1\W*a\W*risk\W*factors", re.I)
    end_re = re.compile(r"item\W*1\W*b\W|item\W*1\W*c\W|item\W*2\W*properties", re.I)

    best = ""
    for m in start_re.finditer(text):
        end_m = end_re.search(text, m.end())
        section = text[m.start(): end_m.start() if end_m else m.start() + max_chars]
        if len(section) > len(best):
            best = section

    if len(best) < 2000:  # fallback: some filings drop "Item 1A" from the heading
        for m in re.finditer(r"risk\W*factors", text, re.I):
            section = text[m.start(): m.start() + max_chars]
            if len(section) > len(best) and "could" in section[:3000].lower():
                best = section
                break
    return best[:max_chars]  # cap length: LLM context limits + cost


def get_risk_factors(cik: str) -> tuple[str, str]:
    """(risk factors text, 10-K url) for the latest annual report."""
    url = get_latest_10k_url(cik)
    return extract_risk_factors(_get(url, timeout=60).text), url

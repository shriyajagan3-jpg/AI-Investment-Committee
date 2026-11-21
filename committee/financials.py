"""
financials.py - turns raw XBRL facts into a clean annual financial summary.

Two real-world problems this handles:
1. Companies use different tags for the same thing (Apple reported revenue as
   'SalesRevenueNet' before 2018 and 'RevenueFromContractWithCustomer...'
   after), so each metric has a list of fallback tags.
2. 10-Ks also contain quarterly and prior-year numbers. We keep only
   full-year values and let later filings overwrite earlier ones, so restated
   numbers win.
"""

from datetime import date

TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                "DepreciationAmortizationAndAccretionNet"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "equity": ["StockholdersEquity"],
}

# Balance-sheet items are a snapshot on one date; the rest cover a period.
INSTANT = {"cash", "total_debt", "equity"}


def annual_series(facts: dict, tags: list[str], instant: bool = False) -> dict[int, float]:
    """{fiscal_year_end_year: value} for one metric, merging fallback tags."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out: dict[int, float] = {}
    for tag in tags:
        entries = gaap.get(tag, {}).get("units", {}).get("USD", [])
        found: dict[int, float] = {}
        for e in sorted(entries, key=lambda e: e.get("filed", "")):  # oldest filing first
            if e.get("form") not in ("10-K", "10-K/A"):
                continue
            if not instant:
                if "start" not in e:
                    continue
                days = (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days
                if not 350 <= days <= 380:  # full fiscal year only
                    continue
            found[int(e["end"][:4])] = float(e["val"])  # later filings overwrite = restated wins
        for year, val in found.items():
            out.setdefault(year, val)  # first tag in the list has priority
    return out


def build_financials(facts: dict, years: int = 5) -> list[dict]:
    """Last `years` fiscal years with core lines plus derived ratios."""
    series = {k: annual_series(facts, tags, k in INSTANT) for k, tags in TAGS.items()}
    if not series["revenue"]:
        raise ValueError("No annual revenue found in XBRL data.")

    fy = sorted(series["revenue"])[-years:]
    rows = []
    for i, y in enumerate(fy):
        r = {"year": y, **{k: series[k].get(y) for k in TAGS}}
        rev = r["revenue"]
        if r["operating_income"] is not None:
            r["ebitda"] = r["operating_income"] + (r["d_and_a"] or 0.0)
        else:
            r["ebitda"] = None

        def pct(x):
            return None if x is None or not rev else x / rev

        r["revenue_growth"] = (rev / rows[i - 1]["revenue"] - 1) if i > 0 and rows[i - 1]["revenue"] else None
        r["gross_margin"] = pct(r["gross_profit"])
        r["ebitda_margin"] = pct(r["ebitda"])
        r["net_margin"] = pct(r["net_income"])
        r["fcf"] = (r["operating_cash_flow"] - (r["capex"] or 0.0)) if r["operating_cash_flow"] is not None else None
        r["fcf_margin"] = pct(r["fcf"])
        r["fcf_conversion"] = (r["fcf"] / r["ebitda"]) if r["fcf"] is not None and r["ebitda"] else None
        r["capex_pct"] = pct(r["capex"])
        r["da_pct"] = pct(r["d_and_a"])
        rows.append(r)
    return rows


def _avg(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def base_case_assumptions(rows: list[dict]) -> dict:
    """Starting LBO assumptions derived from the company's own history."""
    first, last = rows[0], rows[-1]
    n = len(rows) - 1
    cagr = (last["revenue"] / first["revenue"]) ** (1 / n) - 1 if n > 0 and first["revenue"] > 0 else 0.03
    recent = rows[-3:]
    return {
        "revenue_growth": round(max(-0.05, min(0.15, cagr)), 4),  # cap at sane bounds
        "ebitda_margin": round(_avg(r["ebitda_margin"] for r in recent) or 0.15, 4),
        "capex_pct": round(_avg(r["capex_pct"] for r in recent) or 0.04, 4),
        "da_pct": round(_avg(r["da_pct"] for r in recent) or 0.03, 4),
    }


def summarize(rows: list[dict]) -> str:
    """Plain-text table for the agents ($ in millions, ratios in %)."""
    def m(x):
        return "n/a" if x is None else f"{x / 1e6:,.0f}"

    def p(x):
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lines = ["Year | Revenue $M | Growth | Gross Margin | EBITDA $M | EBITDA Margin | Net Income $M | FCF $M | FCF Margin | FCF/EBITDA | Debt $M | Cash $M"]
    for r in rows:
        lines.append(" | ".join([str(r["year"]), m(r["revenue"]), p(r["revenue_growth"]), p(r["gross_margin"]),
                                 m(r["ebitda"]), p(r["ebitda_margin"]), m(r["net_income"]), m(r["fcf"]),
                                 p(r["fcf_margin"]), p(r["fcf_conversion"]), m(r["total_debt"]), m(r["cash"])]))
    return "\n".join(lines)

"""
factcheck.py - catches numbers the AI made up.

LLMs sometimes "hallucinate" figures that sound right. After the memo is
written, we pull every number out of it and check it against a pool of every
number the system actually computed (financials, assumptions, scenario
results, the sensitivity grid), in all the units the memo might use:
raw dollars, millions, billions, and ratios as percentages.
"""

import re

NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?\s*(%|x|[BM]\b|bn|billion|million)?", re.I)


def _variants(v: float) -> list[float]:
    """One true value -> every scale it might appear in."""
    return [v, v / 1e3, v / 1e6, v / 1e9, v * 100]  # /1e3: a $M figure quoted in billions


def build_pool(ctx, grid: dict) -> list[float]:
    pool = []
    for row in ctx.rows:
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != "year":
                pool += _variants(float(v))
    for s in ctx.scenarios:
        for k, v in s.items():
            if isinstance(v, (int, float)):
                pool += _variants(float(v))
        for v in (s.get("assumptions") or {}).values():
            pool += _variants(float(v))
    for entry, row in grid.items():
        pool += [float(entry)]
        for x, irr in row.items():
            pool += [float(x)] + ([irr * 100] if irr is not None else [])
    return pool


def _matches(n: float, pool: list[float]) -> bool:
    # Allow rounding: 2% relative error or 0.15 absolute (e.g. 18.43% written as 18.4%).
    return any(abs(n - p) <= max(0.15, abs(p) * 0.02) for p in pool)


def fact_check(memo: str, ctx, grid: dict) -> dict:
    pool = build_pool(ctx, grid)
    verified, unverified = [], []
    for m in NUMBER.finditer(memo or ""):
        token, unit = m.group(0).strip(), m.group(1)
        n = float(re.sub(r"[^\d.\-]", "", token.replace(",", "")) or 0)
        if 1900 <= n <= 2100 and not unit:      # a year
            continue
        if n.is_integer() and abs(n) <= 10 and not unit:  # counts like "4 risks", "Item 1"
            continue
        (verified if _matches(n, pool) else unverified).append(token)
    total = len(verified) + len(unverified)
    return {"verified": verified, "unverified": unverified,
            "score": (len(verified) / total) if total else 1.0}

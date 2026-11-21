"""
storage.py - saves every committee run to a SQLite database (committee.db).

SQLite is a full SQL database stored in a single file; Python ships with it,
so there is nothing to install. Two tables:

  runs       one row per committee meeting (company, deal terms, decision, fact-check score)
  scenarios  one row per LBO scenario an agent ran, linked to its meeting by run_id

Explore it with `sqlite3 committee.db` or run the starter queries in queries.sql.
"""

import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at              TEXT    NOT NULL,          -- UTC timestamp
    ticker              TEXT    NOT NULL,
    company             TEXT,
    model               TEXT,                      -- which LLM ran the committee
    entry_multiple      REAL,
    exit_multiple       REAL,
    leverage            REAL,
    interest_rate_pct   REAL,
    hold_years          INTEGER,
    hurdle_pct          REAL,
    revenue_musd        REAL,                      -- latest fiscal year
    revenue_cagr_pct    REAL,                      -- over the years pulled
    ebitda_margin_pct   REAL,                      -- latest fiscal year
    base_irr_pct        REAL,                      -- starting base case
    final_irr_pct       REAL,                      -- Deal Lead's final proposed terms
    decision            TEXT,                      -- INVEST / PASS
    conviction          INTEGER,
    guardrail_triggered INTEGER,                   -- 1 if code overrode the Chair
    numbers_checked     INTEGER,                   -- fact-checker totals for the memo
    numbers_verified    INTEGER,
    risk_chars          INTEGER                    -- size of 10-K risk text the Risk Officer got
);

CREATE TABLE IF NOT EXISTS scenarios (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL REFERENCES runs(id),
    name                TEXT,
    agent               TEXT,                      -- who ran it (Modeler, Skeptic, ...)
    irr_pct             REAL,
    moic                REAL,
    entry_multiple      REAL,
    leverage            REAL,
    revenue_growth_pct  REAL,
    ebitda_margin_pct   REAL,
    market_floor_hit    INTEGER                    -- 1 if the agent tried a cheaper price/rate
);
"""


def connect(path: str = "committee.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _pct(x):
    return None if x is None else round(x * 100, 2)


def save_run(conn: sqlite3.Connection, ctx, verdict: dict, factcheck: dict, model: str = "") -> int:
    """Insert one meeting and all its scenarios. Returns the new run id.

    Uses ? placeholders (parameterized queries) rather than building SQL strings,
    which is the standard protection against SQL injection.
    """
    rows, a = ctx.rows, ctx.base
    first, last = rows[0], rows[-1]
    n = len(rows) - 1
    cagr = (last["revenue"] / first["revenue"]) ** (1 / n) - 1 if n > 0 and first["revenue"] else None
    checked = len(factcheck.get("verified", [])) + len(factcheck.get("unverified", []))

    cur = conn.execute(
        """INSERT INTO runs (run_at, ticker, company, model, entry_multiple, exit_multiple, leverage,
               interest_rate_pct, hold_years, hurdle_pct, revenue_musd, revenue_cagr_pct, ebitda_margin_pct,
               base_irr_pct, final_irr_pct, decision, conviction, guardrail_triggered,
               numbers_checked, numbers_verified, risk_chars)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), ctx.ticker, ctx.company, model,
         a.entry_multiple, a.exit_multiple, a.leverage, _pct(a.interest_rate), a.hold_years, _pct(ctx.hurdle),
         round(last["revenue"] / 1e6, 1), _pct(cagr), _pct(last.get("ebitda_margin")),
         ctx.scenarios[0].get("irr_pct") if ctx.scenarios else None, verdict.get("final_terms_irr_pct"),
         str(verdict.get("decision", "")).upper(), verdict.get("conviction"), int(bool(verdict.get("guardrail"))),
         checked, len(factcheck.get("verified", [])), len(ctx.risk_text or "")))
    run_id = cur.lastrowid

    for s in ctx.scenarios:
        asm = s.get("assumptions") or {}
        conn.execute(
            """INSERT INTO scenarios (run_id, name, agent, irr_pct, moic, entry_multiple, leverage,
                   revenue_growth_pct, ebitda_margin_pct, market_floor_hit)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, s.get("scenario"), s.get("by"), s.get("irr_pct"), s.get("moic"), asm.get("entry_multiple"),
             asm.get("leverage"), _pct(asm.get("revenue_growth")), _pct(asm.get("ebitda_margin")),
             int(bool(s.get("market_floor_notes")))))
    conn.commit()
    return run_id

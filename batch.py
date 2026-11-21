"""
batch.py - run the committee on many companies and save every result to committee.db.

    python3 batch.py                      # the default list of 25 companies
    python3 batch.py CHD CLX DPZ          # or pick your own tickers
    python3 batch.py --entry 12 --pause 30

Safe to stop and restart: tickers already saved with the same deal terms are skipped.
Free-tier APIs have per-minute limits, so it pauses between companies.
"""

import argparse
import sys
import time

from committee.agents import build_context, run_committee
from committee.llm import OpenAILLM
from committee.sec_data import get_cik, get_company_facts, get_risk_factors
from committee.storage import connect, save_run

# Mid-size US companies with steady cash flow: the kind PE firms actually look at.
DEFAULT_TICKERS = ["CHD", "CLX", "HRB", "DPZ", "POOL", "ROL", "SJM", "CPB", "MKC", "HAS",
                   "MAT", "LKQ", "GNTX", "SNA", "ALLE", "TTC", "CHE", "FDS", "BR", "JKHY",
                   "EXPD", "CHRW", "AOS", "LECO", "MSA"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--entry", type=float, default=11.0, help="entry EV/EBITDA multiple")
    p.add_argument("--exit", type=float, default=11.0, help="exit multiple")
    p.add_argument("--leverage", type=float, default=5.0, help="debt as x EBITDA")
    p.add_argument("--rate", type=float, default=0.09, help="interest rate, e.g. 0.09")
    p.add_argument("--hold", type=int, default=5, help="hold period in years")
    p.add_argument("--hurdle", type=float, default=0.20, help="fund hurdle IRR, e.g. 0.20")
    p.add_argument("--pause", type=int, default=20, help="seconds to wait between companies")
    p.add_argument("--db", default="committee.db")
    args = p.parse_args()

    conn = connect(args.db)
    llm = OpenAILLM()
    terms = {"entry_multiple": args.entry, "exit_multiple": args.exit, "leverage": args.leverage,
             "interest_rate": args.rate, "hold_years": args.hold}

    done = {r[0] for r in conn.execute(
        "SELECT ticker FROM runs WHERE entry_multiple = ? AND leverage = ? AND hurdle_pct = ?",
        (args.entry, args.leverage, round(args.hurdle * 100, 2)))}

    for i, ticker in enumerate(args.tickers, 1):
        ticker = ticker.upper()
        if ticker in done:
            print(f"[{i}/{len(args.tickers)}] {ticker}: already saved, skipping")
            continue
        print(f"[{i}/{len(args.tickers)}] {ticker}: running...", flush=True)
        try:
            cik, name = get_cik(ticker)
            facts = get_company_facts(cik)
            try:
                risks, url = get_risk_factors(cik)
            except Exception:
                risks, url = "", ""
            ctx = build_context(ticker, name, facts, risks, url, deal_terms=terms, hurdle=args.hurdle)
            verdict, check = {}, {}
            for ev in run_committee(llm, ctx):
                if ev["type"] == "verdict":
                    verdict = ev["verdict"]
                elif ev["type"] == "memo":
                    check = ev["factcheck"]
            run_id = save_run(conn, ctx, verdict, check, llm.model)
            total = len(check.get("verified", [])) + len(check.get("unverified", []))
            print(f"    -> {verdict.get('decision')} (conviction {verdict.get('conviction')}), "
                  f"fact check {len(check.get('verified', []))}/{total}, saved as run {run_id}")
        except KeyboardInterrupt:
            print("\nStopped. Finished companies are saved; run again to continue.")
            sys.exit(0)
        except Exception as e:  # one bad filing shouldn't stop the batch
            print(f"    -> skipped: {e}")
        if i < len(args.tickers):
            time.sleep(args.pause)

    print("\nDone. Summary:")
    row = conn.execute("""
        SELECT COUNT(*),
               SUM(decision = 'INVEST'),
               ROUND(AVG(base_irr_pct), 1),
               SUM(numbers_verified), SUM(numbers_checked),
               SUM(guardrail_triggered)
        FROM runs""").fetchone()
    n, invest, avg_irr, ver, chk, guard = row
    print(f"  companies evaluated:   {n}")
    print(f"  INVEST / PASS:         {invest or 0} / {n - (invest or 0)}")
    print(f"  average base-case IRR: {avg_irr}%")
    if chk:
        print(f"  fact check:            {ver}/{chk} numbers verified ({100 * ver / chk:.1f}%)")
    print(f"  guardrail overrides:   {guard or 0}")
    print("\nExplore more with:  sqlite3 committee.db < queries.sql")


if __name__ == "__main__":
    main()

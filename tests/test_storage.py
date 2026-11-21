"""SQL storage + batch runner, offline."""

import sqlite3
from unittest import mock

from committee.agents import build_context, run_committee
from committee.storage import connect, save_run
from tests.test_committee import FakeLLM, make_facts


def run_meeting():
    ctx = build_context("TEST", "Test Co", make_facts(), "risk text")
    verdict, check = {}, {}
    for ev in run_committee(FakeLLM(), ctx):
        if ev["type"] == "verdict":
            verdict = ev["verdict"]
        elif ev["type"] == "memo":
            check = ev["factcheck"]
    return ctx, verdict, check


def test_save_run_writes_run_and_scenarios(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    ctx, verdict, check = run_meeting()
    run_id = save_run(conn, ctx, verdict, check, "fake-model")
    ticker, decision, checked, verified = conn.execute(
        "SELECT ticker, decision, numbers_checked, numbers_verified FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert (ticker, decision) == ("TEST", "PASS")
    assert checked == 3 and verified == 2
    assert conn.execute("SELECT COUNT(*) FROM scenarios WHERE run_id = ?", (run_id,)).fetchone()[0] == len(ctx.scenarios)


def test_starter_queries_all_run(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(str(db))
    ctx, verdict, check = run_meeting()
    save_run(conn, ctx, verdict, check)
    sql = open("queries.sql").read()
    statements = [s for s in sql.split(";") if "SELECT" in s]
    for stmt in statements:
        body = "\n".join(l for l in stmt.splitlines() if not l.startswith("."))  # drop sqlite3 dot-commands
        conn.execute(body).fetchall()
    assert len(statements) == 8


def test_batch_skips_done_and_survives_errors(tmp_path, monkeypatch, capsys):
    import batch
    db = str(tmp_path / "b.db")

    def fake_cik(t):
        if t == "BAD":
            raise ValueError("not found")
        return "0000000001", f"{t} Inc"

    monkeypatch.setattr(batch, "get_cik", fake_cik)
    monkeypatch.setattr(batch, "get_company_facts", lambda cik: make_facts())
    monkeypatch.setattr(batch, "get_risk_factors", lambda cik: ("risks", "http://x"))
    monkeypatch.setattr(batch, "OpenAILLM", FakeLLM)
    FakeLLM.model = "fake"
    monkeypatch.setattr("sys.argv", ["batch.py", "AAA", "BAD", "BBB", "--pause", "0", "--db", db])
    batch.main()
    monkeypatch.setattr("sys.argv", ["batch.py", "AAA", "--pause", "0", "--db", db])
    batch.main()
    out = capsys.readouterr().out
    assert "skipped: not found" in out and "already saved" in out
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2

"""End-to-end test of the committee with a scripted fake model (no API key needed)."""

import json

from committee.agents import build_context, run_committee
from committee.llm import LLM
from tests.test_core import fact, facts_with


class FakeLLM(LLM):
    """Agents with tools request one LBO scenario first, then answer in text."""

    def chat(self, messages, tools=None):
        system = messages[0]["content"]
        already_called = any(m.get("role") == "tool" for m in messages)
        if tools and not already_called:
            call = {"id": "c1", "name": "run_lbo_scenario",
                    "arguments": {"scenario_name": "margin squeeze", "rationale": "test",
                                  "overrides": {"ebitda_margin": 0.15, "bogus_field": 1}}}
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "run_lbo_scenario", "arguments": json.dumps(call["arguments"])}}]}
            return {"content": None, "tool_calls": [call], "message": msg}
        if "Chair" in system:
            text = '```json\n{"decision": "PASS", "conviction": 7, "thesis": "Too fragile."}\n```'
        elif "memo" in system:
            text = "## Recommendation\nPASS. Revenue was 1,100 and EBITDA margin 18.2%. IRR could hit 97.3%."
        else:
            text = "Looks fine."
        return {"content": text, "tool_calls": [], "message": {"role": "assistant", "content": text}}


def make_facts():
    years = [(2021, 1000), (2022, 1050), (2023, 1100)]
    return facts_with(
        Revenues=[fact(r * 1e6, f"{y}-12-31", f"{y}-01-01") for y, r in years],
        OperatingIncomeLoss=[fact(r * 0.15e6, f"{y}-12-31", f"{y}-01-01") for y, r in years],
        DepreciationDepletionAndAmortization=[fact(r * 0.03e6, f"{y}-12-31", f"{y}-01-01") for y, r in years],
    )


def test_full_meeting():
    ctx = build_context("TEST", "Test Co", make_facts(), "Item 1A Risk Factors: competition.")
    events = list(run_committee(FakeLLM(), ctx))
    types = [e["type"] for e in events]

    assert types[0] == "model"
    assert [e["agent"] for e in events if e["type"] == "agent"] == \
        ["Analyst", "Risk Officer", "Modeler", "Skeptic", "Deal Lead"]
    # three tool-using agents each ran one scenario, plus the starting base case
    assert len(ctx.scenarios) == 4
    assert all("bogus_field" not in s["assumptions"] for s in ctx.scenarios)  # bad keys dropped
    assert ctx.scenarios[1]["assumptions"]["ebitda_margin"] == 0.15

    verdict = next(e for e in events if e["type"] == "verdict")["verdict"]
    assert verdict["decision"] == "PASS" and verdict["conviction"] == 7

    check = next(e for e in events if e["type"] == "memo")["factcheck"]
    assert "1,100" in check["verified"] and "18.2%" in check["verified"]
    assert check["unverified"] == ["97.3%"]  # the made-up IRR gets caught


def test_guardrail_overrides_unsupported_invest():
    from committee.agents import apply_guardrail
    ctx = build_context("TEST", "Test Co", make_facts(), "")
    ctx.scenarios = [{"scenario": "Starting base case", "by": "model", "irr_pct": 12.8},
                     {"scenario": "final terms", "by": "Deal Lead", "irr_pct": 11.2}]
    v = apply_guardrail({"decision": "INVEST", "conviction": 7}, ctx)
    assert v["decision"] == "PASS" and "11.2" in v["guardrail"]
    ctx.scenarios[-1]["irr_pct"] = 22.0
    assert apply_guardrail({"decision": "INVEST"}, ctx)["decision"] == "INVEST"


def test_agents_cannot_lower_price_or_rate():
    from committee.agents import run_scenario_tool
    ctx = build_context("TEST", "Test Co", make_facts(), "", deal_terms={"entry_multiple": 11, "interest_rate": 0.09})
    out = run_scenario_tool(ctx, {"scenario_name": "wishful", "overrides": {"entry_multiple": 8.5, "interest_rate": 0.07,
                                                                          "leverage": 4.0}})
    assert out["assumptions"]["entry_multiple"] == 11 and out["assumptions"]["interest_rate"] == 0.09
    assert out["assumptions"]["leverage"] == 4.0 and len(out["market_floor_notes"]) == 2
    up = run_scenario_tool(ctx, {"scenario_name": "pricier", "overrides": {"entry_multiple": 13}})
    assert up["assumptions"]["entry_multiple"] == 13 and "market_floor_notes" not in up

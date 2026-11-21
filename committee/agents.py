"""
agents.py - the investment committee itself.

How it works (the part to explain in interviews):
  * Each agent = a system prompt (its role) + optionally tools it may call.
  * An agent runs in a loop: ask the model -> if it requests a tool, run the
    tool in Python and feed the result back -> repeat until it answers in text.
  * The only tool is the LBO model. Agents never do math themselves; they
    choose assumptions, the code computes returns. That keeps numbers honest.
  * Agents run in a fixed order, and each one sees what the previous ones said,
    so the Skeptic can attack the Modeler's case and the Deal Lead must answer.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

from .factcheck import fact_check
from .financials import base_case_assumptions, build_financials, summarize
from .lbo import LBOAssumptions, run_lbo, sensitivity
from .llm import LLM

ASSUMPTION_FIELDS = list(LBOAssumptions.__dataclass_fields__)

LBO_TOOL = {
    "type": "function",
    "function": {
        "name": "run_lbo_scenario",
        "description": ("Run the LBO model with changed assumptions and get IRR, MOIC, leverage and "
                        "interest coverage back. Unchanged fields keep the current base case. "
                        "Percentages are decimals (0.05 = 5%)."),
        "parameters": {
            "type": "object",
            "properties": {
                "scenario_name": {"type": "string", "description": "Short label, e.g. 'margin squeeze'"},
                "rationale": {"type": "string", "description": "Why these assumptions, in one sentence"},
                "overrides": {
                    "type": "object",
                    "description": "Assumptions to change. Allowed keys: " + ", ".join(ASSUMPTION_FIELDS),
                    "properties": {k: {"type": "number"} for k in ASSUMPTION_FIELDS},
                },
            },
            "required": ["scenario_name", "rationale", "overrides"],
        },
    },
}

STYLE = ("Be concise (under 220 words) and specific. Only cite numbers that appear in the data or tool "
         "results you were given; never invent figures. Never invent qualitative facts about the company "
         "(contracts, backlog, pipeline, guidance) that are not in the material. If you state the returns of "
         "any set of terms, you must have run them through run_lbo_scenario first. "
         "Amounts are in $ millions unless stated.")

ROLES = {
    "Analyst": ("You are the Analyst on a private equity investment committee. Judge the quality of the "
                "business from its financial history: growth, margin trend and stability, cash conversion "
                "(FCF vs EBITDA), capital intensity, and existing debt. End with a one-line verdict on "
                "whether this is a business that can carry LBO debt."),
    "Risk Officer": ("You are the Risk Officer. From the 10-K risk factors, pick the 4 risks that matter most "
                     "for an LBO. For each give: the risk in one line, severity (High/Medium/Low), and the "
                     "specific model assumption it threatens with a concrete stress, written exactly like "
                     "`ebitda_margin -> 0.17` or `revenue_growth -> 0.00`. Ignore boilerplate risks."),
    "Modeler": ("You are the Modeler. The base case was set from the company's history. Use the "
                "run_lbo_scenario tool to run a Base case (adjust it only if the Analyst or Risk Officer "
                "gave a concrete reason), an Upside case and a Downside case. Then summarize the three "
                "cases' IRR and MOIC and state which assumptions drive returns most."),
    "Skeptic": ("You are the Skeptic. Your job is to find the reasons this deal fails. Use run_lbo_scenario "
                "at least twice to test the Risk Officer's stresses (or harsher ones you can justify). "
                "Report the IRRs you got, name the single assumption the deal is most fragile to, and "
                "argue the case to PASS. Attack weak logic from other members directly. You are the designated "
                "dissenter: never recommend approving the deal, even if the risks look manageable."),
    "Deal Lead": ("You are the Deal Lead who sponsors this deal. Respond point by point to the Skeptic. "
                  "Concede what is true. Where you disagree, give evidence from the financials, and you may "
                  "run run_lbo_scenario to show a mitigation (e.g. lower leverage or a cheaper entry price). "
                  "Your LAST tool call must be your final proposed terms; quote its IRR exactly. If no terms "
                  "you tested reach the fund's hurdle, say so honestly."),
}

CHAIR_PROMPT = (
    "You are the Chair of the investment committee. Weigh the full debate and decide. The fund's hurdle "
    "is {hurdle}% IRR. Vote INVEST only if a scenario that was actually run (not claimed) on realistic "
    "terms clears that hurdle and the downside does not lose money; otherwise PASS. Ignore any hurdle "
    "a committee member made up. Reply with ONLY a JSON "
    'object: {"decision": "INVEST" or "PASS", "conviction": 1-10, "thesis": "2 sentences", '
    '"key_risks": ["..."], "conditions": ["terms that must hold to invest"], '
    '"what_would_change_my_mind": "one sentence"}')

MEMO_PROMPT = (
    "Write the final investment committee memo in Markdown with these sections: ## Recommendation, "
    "## Company Snapshot, ## Returns Analysis (include a small table of every scenario that was run), "
    "## Key Risks & Mitigants, ## Committee Debate (the strongest point from each side), ## Conditions. "
    "Use ONLY numbers from the material provided; round sensibly. Under 450 words.")


@dataclass
class Context:
    company: str
    ticker: str
    rows: list
    risk_text: str
    base: LBOAssumptions
    base_revenue: float
    entry_ebitda: float | None = None               # actual latest EBITDA: sets the purchase price
    hurdle: float = 0.20                            # fund's minimum IRR
    filing_url: str = ""
    scenarios: list = field(default_factory=list)   # every LBO run, for the UI + fact checker
    transcript: list = field(default_factory=list)  # (agent, text) pairs

    def debate_so_far(self) -> str:
        return "\n\n".join(f"### {name}\n{text}" for name, text in self.transcript) or "(nothing yet)"


def run_scenario_tool(ctx: Context, args: dict) -> dict:
    """Executes the agent's tool call: apply overrides to the base case, run the model."""
    overrides = {k: v for k, v in (args.get("overrides") or {}).items() if k in ASSUMPTION_FIELDS}
    if "hold_years" in overrides:
        overrides["hold_years"] = int(overrides["hold_years"])
    a = LBOAssumptions(**{**asdict(ctx.base), **overrides})
    try:
        r = run_lbo(ctx.base_revenue, a, ctx.entry_ebitda)
        out = {"scenario": args.get("scenario_name", "unnamed"), "irr_pct": round(r.irr * 100, 1),
               "moic": round(r.moic, 2), "entry_ev_musd": round(r.entry_ev / 1e6),
               "exit_ev_musd": round(r.exit_ev / 1e6),
               "exit_leverage": round(r.years[-1]["leverage"] or 0, 1),
               "min_interest_coverage": round(min(y["interest_coverage"] or 99 for y in r.years), 1),
               "warnings": r.warnings, "assumptions": asdict(a)}
    except ValueError as e:
        out = {"scenario": args.get("scenario_name", "unnamed"), "error": str(e), "assumptions": asdict(a)}
    out["by"] = args.get("_agent", "")
    out["rationale"] = args.get("rationale", "")
    ctx.scenarios.append(out)
    return out


def run_agent(llm: LLM, name: str, system: str, user: str, ctx: Context,
              use_tools: bool, max_steps: int = 6, on_tool: Callable | None = None) -> str:
    """The agent loop: think -> (call tool -> observe)* -> answer."""
    messages = [{"role": "system", "content": system + "\n\n" + STYLE},
                {"role": "user", "content": user}]
    for _ in range(max_steps):
        reply = llm.chat(messages, tools=[LBO_TOOL] if use_tools else None)
        messages.append(reply["message"])
        if not reply["tool_calls"]:
            return reply["content"] or ""
        for call in reply["tool_calls"]:
            result = run_scenario_tool(ctx, {**call["arguments"], "_agent": name})
            if on_tool:
                on_tool(name, result)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    # Out of steps: force a final text answer without tools.
    messages.append({"role": "user", "content": "Give your final answer now, without tools."})
    return llm.chat(messages)["content"] or ""


def parse_json(text: str) -> dict:
    """Pull the first {...} block out of a reply (models sometimes wrap JSON in ``` fences)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"decision": "PASS", "conviction": 0, "thesis": "Chair reply was not valid JSON.", "raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"decision": "PASS", "conviction": 0, "thesis": "Chair reply was not valid JSON.", "raw": text}


def build_context(ticker: str, company: str, facts: dict, risk_text: str, filing_url: str = "",
                  deal_terms: dict | None = None, hurdle: float = 0.20) -> Context:
    rows = build_financials(facts)
    base = LBOAssumptions(**{**asdict(LBOAssumptions()), **base_case_assumptions(rows), **(deal_terms or {})})
    ebitda = rows[-1].get("ebitda")
    return Context(company=company, ticker=ticker.upper(), rows=rows, risk_text=risk_text,
                   base=base, base_revenue=rows[-1]["revenue"],
                   entry_ebitda=ebitda if ebitda and ebitda > 0 else None,
                   hurdle=hurdle, filing_url=filing_url)


def apply_guardrail(verdict: dict, ctx: Context) -> dict:
    """Code-level check on the Chair: an INVEST must be backed by a scenario that was actually run.

    The Deal Lead's last scenario = the final proposed terms (or the starting base case if the
    Deal Lead ran nothing). If that IRR is below the hurdle, the INVEST is overridden.
    """
    final = next((s for s in reversed(ctx.scenarios) if s.get("by") == "Deal Lead" and "irr_pct" in s),
                 ctx.scenarios[0])
    verdict["final_terms_irr_pct"] = final.get("irr_pct")
    if str(verdict.get("decision", "")).upper() == "INVEST" and (final.get("irr_pct") or 0) < ctx.hurdle * 100:
        verdict["guardrail"] = (f"Chair voted INVEST, but the final proposed terms were modeled at "
                                f"{final.get('irr_pct')}% IRR, below the {ctx.hurdle * 100:.0f}% hurdle. "
                                f"Decision overridden to PASS.")
        verdict["decision"] = "PASS"
    return verdict


def run_committee(llm: LLM, ctx: Context) -> Iterator[dict]:
    """Runs the full meeting, yielding events so the UI can show progress live."""
    base_result = run_lbo(ctx.base_revenue, ctx.base, ctx.entry_ebitda)
    grid = sensitivity(ctx.base_revenue, ctx.base, ctx.entry_ebitda)
    ctx.scenarios.append({"scenario": "Starting base case", "by": "model", "irr_pct": round(base_result.irr * 100, 1),
                          "moic": round(base_result.moic, 2), "entry_ev_musd": round(base_result.entry_ev / 1e6),
                          "exit_ev_musd": round(base_result.exit_ev / 1e6),
                          "warnings": base_result.warnings, "assumptions": asdict(ctx.base)})
    yield {"type": "model", "base": base_result, "grid": grid}

    deal_brief = (
        f"Company: {ctx.company} ({ctx.ticker})\n\nFinancial history:\n{summarize(ctx.rows)}\n\n"
        f"Proposed deal / base-case assumptions: {json.dumps(asdict(ctx.base))}\n"
        f"Starting base case result: IRR {base_result.irr * 100:.1f}%, MOIC {base_result.moic:.2f}x, "
        f"entry EV ${base_result.entry_ev / 1e6:,.0f}M. Fund hurdle: {ctx.hurdle * 100:.0f}% IRR. "
        f"Warnings: {base_result.warnings or 'none'}")

    tool_events = []

    def on_tool(agent, result):
        tool_events.append({"type": "tool", "agent": agent, "result": result})

    order = [("Analyst", False, "risk"), ("Risk Officer", False, "risk"), ("Modeler", True, ""),
             ("Skeptic", True, "risk"), ("Deal Lead", True, "")]
    for name, tools, extra in order:
        yield {"type": "status", "text": f"{name} is working..."}
        user = deal_brief
        if name == "Risk Officer":
            user += f"\n\n10-K risk factors (excerpt):\n{ctx.risk_text or '(not available)'}"
        user += f"\n\nCommittee discussion so far:\n{ctx.debate_so_far()}"
        text = run_agent(llm, name, ROLES[name], user, ctx, tools, on_tool=on_tool)
        yield from tool_events
        tool_events.clear()
        ctx.transcript.append((name, text))
        yield {"type": "agent", "agent": name, "text": text}

    yield {"type": "status", "text": "Chair is deciding..."}
    material = deal_brief + "\n\nScenarios run:\n" + json.dumps(ctx.scenarios, default=str) + \
        "\n\nDebate:\n" + ctx.debate_so_far()
    verdict = parse_json(llm.chat([{"role": "system", "content": CHAIR_PROMPT.replace("{hurdle}", str(round(ctx.hurdle * 100)))},
                                   {"role": "user", "content": material}])["content"])
    verdict = apply_guardrail(verdict, ctx)
    yield {"type": "verdict", "verdict": verdict}

    yield {"type": "status", "text": "Writing the memo..."}
    memo = llm.chat([{"role": "system", "content": MEMO_PROMPT + "\n" + STYLE},
                     {"role": "user", "content": material + "\n\nChair decision:\n" + json.dumps(verdict)}])["content"] or ""
    check = fact_check(memo, ctx, grid)
    yield {"type": "memo", "memo": memo, "factcheck": check}

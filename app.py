"""
app.py - Streamlit front end. Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from committee.agents import build_context, run_committee
from committee.llm import OpenAILLM
from committee.sec_data import get_cik, get_company_facts, get_risk_factors
from committee.storage import connect, save_run

st.set_page_config(page_title="AI Investment Committee", page_icon="📈", layout="wide")
st.title("AI Investment Committee")
st.caption("Five AI agents debate a leveraged buyout using the company's real SEC filings. "
           "All math runs in Python; a fact-checker flags any number the AI invents.")

def md(text: str) -> str:
    """Escape $ so Streamlit doesn't render '$5,190 M to $6,203 M' as a math formula."""
    return (text or "").replace("$", "\\$")


AVATARS = {"Analyst": "📊", "Risk Officer": "⚠️", "Modeler": "🧮", "Skeptic": "🤨", "Deal Lead": "🤝"}

with st.sidebar:
    st.header("Deal setup")
    ticker = st.text_input("Ticker", "CHD", help="Mid-size, stable companies (e.g. CHD, CLX, HRB, DPZ) make the most realistic LBO targets.")
    entry = st.slider("Entry multiple (EV / EBITDA)", 6.0, 20.0, 11.0, 0.5)
    exit_ = st.slider("Exit multiple", 6.0, 20.0, 11.0, 0.5)
    leverage = st.slider("Debt (x EBITDA)", 2.0, 7.0, 5.0, 0.5)
    rate = st.slider("Interest rate", 0.04, 0.14, 0.09, 0.005, format="%.3f")
    hold = st.slider("Hold period (years)", 3, 7, 5)
    hurdle = st.slider("Fund hurdle IRR (%)", 10, 30, 20) / 100
    go = st.button("Convene the committee", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False, ttl=3600)
def load(t: str):
    cik, name = get_cik(t)
    facts = get_company_facts(cik)
    try:
        risks, url = get_risk_factors(cik)
    except Exception:  # risk section is a bonus; the deal still runs without it
        risks, url = "", ""
    return name, facts, risks, url


def money(x):
    return "n/a" if x is None else f"${x / 1e6:,.0f}M"


if go:
    try:
        with st.spinner("Pulling SEC filings..."):
            name, facts, risks, url = load(ticker)
        ctx = build_context(ticker, name, facts, risks, url, deal_terms={
            "entry_multiple": entry, "exit_multiple": exit_, "leverage": leverage,
            "interest_rate": rate, "hold_years": hold}, hurdle=hurdle)
    except Exception as e:
        st.error(f"Couldn't load {ticker}: {e}")
        st.stop()

    st.subheader(f"{name} ({ticker.upper()})")
    if url:
        st.caption(f"Source: [latest 10-K]({url}) and SEC XBRL financial data")
    with st.expander(f"10-K risk factors given to the Risk Officer ({len(risks):,} characters)"):
        st.write(risks[:3000] + ("..." if len(risks) > 3000 else "") if risks else
                 "Could not extract the risk section from this filing; the Risk Officer works from the financials only.")

    df = pd.DataFrame(ctx.rows)
    df["year"] = df["year"].astype(str)  # show 2024, not 2,024
    df = df.set_index("year")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Financial history ($M)**")
        st.bar_chart((df[["revenue", "ebitda", "fcf"]] / 1e6).rename(
            columns={"revenue": "Revenue", "ebitda": "EBITDA", "fcf": "Free cash flow"}), stack=False)
    with c2:
        st.markdown("**Margins (%)**")
        st.line_chart((df[["gross_margin", "ebitda_margin", "fcf_margin"]] * 100).rename(
            columns={"gross_margin": "Gross", "ebitda_margin": "EBITDA", "fcf_margin": "FCF"}))

    status = st.empty()
    llm = OpenAILLM()
    verdict = {}
    for ev in run_committee(llm, ctx):
        t = ev["type"]
        if t == "model":
            b = ev["base"]
            st.markdown("### Starting model")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Base IRR", f"{b.irr * 100:.1f}%")
            m2.metric("MOIC", f"{b.moic:.2f}x")
            m3.metric("Entry EV", money(b.entry_ev))
            m4.metric("Equity check", money(b.entry_equity))
            for w in b.warnings:
                st.warning(w)
            grid = pd.DataFrame(ev["grid"]).T * 100
            grid.index = [f"Entry {i}x" for i in grid.index]
            grid.columns = [f"Exit {c}x" for c in grid.columns]
            st.markdown("**IRR sensitivity (%)**")
            st.dataframe(grid.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=None))
            st.markdown("### Committee debate")
        elif t == "status":
            status.info(ev["text"])
        elif t == "tool":
            r = ev["result"]
            label = f"{AVATARS.get(ev['agent'], '')} {ev['agent']} ran scenario **{r['scenario']}**"
            with st.expander(label + (f": IRR {r['irr_pct']}%, MOIC {r['moic']}x" if "irr_pct" in r else ": failed")):
                st.write(r.get("rationale", ""))
                st.json(r, expanded=False)
        elif t == "agent":
            with st.chat_message(ev["agent"], avatar=AVATARS.get(ev["agent"])):
                st.markdown(f"**{ev['agent']}**\n\n{md(ev['text'])}")
        elif t == "verdict":
            v = verdict = ev["verdict"]
            st.markdown("### Decision")
            color = "green" if v.get("decision") == "INVEST" else "red"
            st.markdown(f"## :{color}[{v.get('decision', '?')}]  ·  conviction {v.get('conviction', '?')}/10")
            if v.get("guardrail"):
                st.error("Guardrail: " + v["guardrail"])
            st.markdown(md(v.get("thesis", "")))
            if v.get("conditions"):
                st.markdown("**Conditions:** " + "; ".join(v["conditions"]))
            if v.get("what_would_change_my_mind"):
                st.caption("What would change the Chair's mind: " + v["what_would_change_my_mind"])
        elif t == "memo":
            status.empty()
            fc = ev["factcheck"]
            st.markdown("### Investment memo")
            st.progress(fc["score"], text=f"Fact check: {len(fc['verified'])} of "
                        f"{len(fc['verified']) + len(fc['unverified'])} numbers traced to source data")
            if fc["unverified"]:
                st.warning("Could not verify: " + ", ".join(fc["unverified"]))
            st.markdown(md(ev["memo"]))
            st.download_button("Download memo (.md)", ev["memo"], file_name=f"{ticker.upper()}_IC_memo.md")
            try:
                run_id = save_run(connect(), ctx, verdict, fc, llm.model)
                st.caption(f"Saved to committee.db as run #{run_id}")
            except Exception as e:
                st.caption(f"Could not save this run to the database: {e}")
else:
    st.info("Pick a ticker and deal terms on the left, then convene the committee.")

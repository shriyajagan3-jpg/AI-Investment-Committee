"""Offline tests for the math and parsing (no API keys or internet needed).
Run with:  python -m pytest -q
"""

import pytest

from committee.financials import annual_series, build_financials, base_case_assumptions
from committee.lbo import LBOAssumptions, run_lbo, sensitivity
from committee.sec_data import extract_risk_factors


def fact(val, end, start=None, form="10-K", filed="2024-01-01"):
    e = {"val": val, "end": end, "form": form, "filed": filed}
    if start:
        e["start"] = start
    return e


def facts_with(**tags):
    return {"facts": {"us-gaap": {t: {"units": {"USD": v}} for t, v in tags.items()}}}


def test_annual_series_skips_quarters_and_keeps_restated():
    f = facts_with(Revenues=[
        fact(100, "2022-12-31", "2022-01-01", filed="2023-02-01"),
        fact(25, "2022-03-31", "2022-01-01", filed="2023-02-01"),      # quarter -> ignored
        fact(105, "2022-12-31", "2022-01-01", filed="2024-02-01"),     # restated later -> wins
        fact(50, "2021-12-31", "2021-01-01", form="10-Q"),             # 10-Q -> ignored
    ])
    assert annual_series(f, ["Revenues"]) == {2022: 105}


def test_tag_fallback_merges_old_and_new_revenue_tags():
    f = facts_with(
        SalesRevenueNet=[fact(90, "2017-12-31", "2017-01-01")],
        RevenueFromContractWithCustomerExcludingAssessedTax=[fact(100, "2018-12-31", "2018-01-01")],
    )
    tags = ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
    assert annual_series(f, tags) == {2017: 90, 2018: 100}


def test_build_financials_ratios():
    f = facts_with(
        Revenues=[fact(1000, "2022-12-31", "2022-01-01"), fact(1100, "2023-12-31", "2023-01-01")],
        OperatingIncomeLoss=[fact(150, "2023-12-31", "2023-01-01")],
        DepreciationDepletionAndAmortization=[fact(50, "2023-12-31", "2023-01-01")],
    )
    rows = build_financials(f)
    last = rows[-1]
    assert last["ebitda"] == 200
    assert last["revenue_growth"] == pytest.approx(0.10)
    assert last["ebitda_margin"] == pytest.approx(200 / 1100)
    assert base_case_assumptions(rows)["revenue_growth"] == pytest.approx(0.10)


def test_lbo_year_one_by_hand():
    a = LBOAssumptions(entry_multiple=10, exit_multiple=10, leverage=5, interest_rate=0.10,
                       revenue_growth=0.0, ebitda_margin=0.20, capex_pct=0.03, da_pct=0.03,
                       nwc_pct=0.0, tax_rate=0.25, cash_sweep=1.0, hold_years=1)
    r = run_lbo(1000, a)
    # EBITDA 200 -> EV 2000, debt 1000, equity 1000
    # Interest 100, pretax 200-30-100=70, tax 17.5, NI 52.5, FCF = 52.5+30-30 = 52.5
    assert r.entry_equity == pytest.approx(1000)
    assert r.years[0]["fcf"] == pytest.approx(52.5)
    assert r.years[0]["ending_debt"] == pytest.approx(947.5)
    # Exit EV 2000 - debt 947.5 = 1052.5 -> MOIC 1.0525, 1-year IRR 5.25%
    assert r.moic == pytest.approx(1.0525)
    assert r.irr == pytest.approx(0.0525)


def test_higher_exit_multiple_raises_irr():
    grid = sensitivity(1000, LBOAssumptions())
    assert grid[10][14] > grid[10][10] > grid[10][8]
    assert grid[8][10] > grid[12][10]  # paying less is better


def test_overlevered_deal_rejected():
    with pytest.raises(ValueError):
        run_lbo(1000, LBOAssumptions(entry_multiple=5, leverage=6))


def test_risk_factor_extraction_skips_table_of_contents():
    html = ("<p>Item 1A. Risk Factors ... 12</p><p>Item 1B. Unresolved</p>"
            "<p>Item 1A. Risk Factors</p><p>" + "Competition is intense. " * 50 +
            "</p><p>Item 1B. Unresolved Staff Comments</p>")
    out = extract_risk_factors(html)
    assert "Competition is intense" in out
    assert "Unresolved" not in out


def test_margin_stress_does_not_change_purchase_price():
    # Price is set on today's actual EBITDA; a lower future margin must hurt returns.
    base = LBOAssumptions(ebitda_margin=0.20)
    stressed = LBOAssumptions(ebitda_margin=0.15)
    a = run_lbo(1000, base, entry_ebitda=200)
    b = run_lbo(1000, stressed, entry_ebitda=200)
    assert a.entry_ev == b.entry_ev == 2000
    assert b.irr < a.irr - 0.05


def test_risk_extraction_handles_nbsp_and_caps():
    html = ("<div>ITEM 1A. RISK FACTORS</div><p>" + "Our results could be harmed. " * 100 +
            "</p><div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div>")
    out = extract_risk_factors(html)
    assert "could be harmed" in out and "UNRESOLVED" not in out

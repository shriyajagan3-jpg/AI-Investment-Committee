"""
lbo.py - a leveraged buyout (LBO) model, written in plain Python.

The idea of an LBO in one paragraph: a PE fund buys a company using mostly
borrowed money. The company's own cash flow pays the debt down over ~5 years.
The fund then sells the company. Because the fund only put in a small slice of
equity, paying down debt + growing earnings turns into a big return on that
equity. We measure it with:
  MOIC = money out / money in          (e.g. 2.5x)
  IRR  = the annual return that gets you that MOIC over the hold period

All math lives here in code, never in the LLM: language models are bad at
arithmetic, so agents call this function as a tool instead.
"""

from dataclasses import dataclass, asdict, field


@dataclass
class LBOAssumptions:
    entry_multiple: float = 10.0     # purchase price = entry_multiple x EBITDA
    exit_multiple: float = 10.0      # sale price = exit_multiple x final-year EBITDA
    leverage: float = 5.0            # debt raised = leverage x EBITDA
    interest_rate: float = 0.09      # annual interest on the debt
    revenue_growth: float = 0.05
    ebitda_margin: float = 0.20
    capex_pct: float = 0.04          # capital spending, % of revenue
    da_pct: float = 0.03             # depreciation & amortization, % of revenue
    nwc_pct: float = 0.10            # working capital tied up per $ of new revenue
    tax_rate: float = 0.25
    cash_sweep: float = 1.0          # share of free cash flow used to repay debt
    hold_years: int = 5


@dataclass
class LBOResult:
    irr: float
    moic: float
    entry_ev: float
    entry_equity: float
    exit_ev: float
    exit_equity: float
    years: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_lbo(base_revenue: float, a: LBOAssumptions) -> LBOResult:
    """Project the deal year by year and compute returns."""
    warnings = []
    ebitda0 = base_revenue * a.ebitda_margin
    if ebitda0 <= 0:
        raise ValueError("EBITDA must be positive to underwrite an LBO.")

    entry_ev = ebitda0 * a.entry_multiple
    debt = ebitda0 * a.leverage
    equity_in = entry_ev - debt
    if equity_in <= 0:
        raise ValueError("Leverage is higher than the purchase multiple: no equity left.")
    if entry_ev > 100e9:
        warnings.append(f"Entry EV ${entry_ev / 1e9:,.0f}B is larger than any LBO in history; treat as a thought experiment.")
    if a.leverage > 7:
        warnings.append("Leverage above 7x EBITDA is beyond what lenders typically provide.")

    revenue, cash, years = base_revenue, 0.0, []
    for y in range(1, a.hold_years + 1):
        new_revenue = revenue * (1 + a.revenue_growth)
        ebitda = new_revenue * a.ebitda_margin
        d_and_a = new_revenue * a.da_pct
        interest = debt * a.interest_rate                    # interest on opening debt balance
        pretax = ebitda - d_and_a - interest
        taxes = max(0.0, pretax) * a.tax_rate
        net_income = pretax - taxes
        capex = new_revenue * a.capex_pct
        delta_nwc = (new_revenue - revenue) * a.nwc_pct
        fcf = net_income + d_and_a - capex - delta_nwc        # cash left after running the business

        paydown = min(debt, max(0.0, fcf) * a.cash_sweep)
        debt -= paydown
        cash += fcf - paydown
        if cash < 0:
            warnings.append(f"Year {y}: cash goes negative; the company cannot service its debt.")

        years.append({"year": y, "revenue": new_revenue, "ebitda": ebitda, "interest": interest,
                      "net_income": net_income, "fcf": fcf, "debt_paydown": paydown,
                      "ending_debt": debt, "leverage": debt / ebitda if ebitda > 0 else None,
                      "interest_coverage": ebitda / interest if interest > 0 else None})
        revenue = new_revenue

    exit_ev = years[-1]["ebitda"] * a.exit_multiple
    exit_equity = exit_ev - debt + cash
    moic = exit_equity / equity_in
    # No dividends during the hold, so IRR is just the annualized MOIC.
    irr = moic ** (1 / a.hold_years) - 1 if moic > 0 else -1.0

    min_cov = min((yr["interest_coverage"] for yr in years if yr["interest_coverage"]), default=None)
    if min_cov is not None and min_cov < 2:
        warnings.append(f"Interest coverage falls to {min_cov:.1f}x EBITDA/interest; lenders usually want 2x+.")

    return LBOResult(irr, moic, entry_ev, equity_in, exit_ev, exit_equity, years, warnings)


def sensitivity(base_revenue: float, a: LBOAssumptions,
                entry=(8, 10, 12, 14), exit=(8, 10, 12, 14)) -> dict:
    """IRR grid across entry and exit multiples - the table every PE memo has."""
    grid = {}
    for e in entry:
        grid[e] = {}
        for x in exit:
            try:
                r = run_lbo(base_revenue, LBOAssumptions(**{**asdict(a), "entry_multiple": e, "exit_multiple": x}))
                grid[e][x] = r.irr
            except ValueError:
                grid[e][x] = None
    return grid

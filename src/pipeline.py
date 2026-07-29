"""End-to-end pipeline: download, analyse, and regenerate every figure.

Importable from the notebook and runnable on its own::

    python -m src.pipeline

Keeping the orchestration here rather than in the notebook means the numbers in
the README and the numbers in the notebook come from the same code path.
"""

from __future__ import annotations

import itertools

import pandas as pd

from . import analysis, fetch, plots

COINS = ("BTC", "ETH", "SOL")
VENUES = ("binance", "bybit", "okx", "hyperliquid")

# The pair the analysis focuses on: one centralised order book against one
# on-chain venue. It has the longest common history and the largest spread.
FOCUS_PAIR = ("binance", "hyperliquid")
# A control: two centralised venues of the same type, to show what an
# *unbiased* spread looks like.
CONTROL_PAIR = ("binance", "bybit")

# Fixed order, because it is also the colour order in every chart.
STRATEGIES = (
    "Slow side (30-day review)",
    "Active rotation (8-hour signal)",
    "Passive hold (side fixed at start)",
)

BASE_PROFILE = "maker/taker (mixed)"


def load_funding(use_cache: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
    """Funding history for every coin and venue, normalised to a rate per hour."""
    return {
        coin: {venue: fetch.VENUE_FETCHERS[venue](coin=coin, use_cache=use_cache)
               for venue in VENUES}
        for coin in COINS
    }


def build_spreads(data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, pd.DataFrame]]:
    """Every venue-pair spread, per coin, on each pair's overlapping window."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for coin, venues in data.items():
        out[coin] = {}
        for a, b in itertools.combinations(VENUES, 2):
            if venues.get(a) is None or venues.get(b) is None:
                continue
            spread = analysis.venue_spread(venues[a], venues[b], a, b)
            if not spread.empty:
                out[coin][f"{a}/{b}"] = spread
    return out


def run_strategies(spread: pd.DataFrame, costs: analysis.CostModel) -> dict[str, pd.DataFrame]:
    """The three decision rules, run on the same spread under the same costs."""
    return {
        STRATEGIES[0]: analysis.backtest_slow_side(spread, costs),
        STRATEGIES[1]: analysis.backtest_spread(spread, costs, entry_annual_pct=10.0,
                                                exit_annual_pct=3.0, lookback_hours=8,
                                                min_consistency=0.7),
        STRATEGIES[2]: analysis.passive_hold(spread, costs),
    }


def strategy_summary(spreads: dict[str, dict[str, pd.DataFrame]],
                     costs: analysis.CostModel) -> pd.DataFrame:
    """Every strategy on every coin under one cost profile."""
    pair = "/".join(FOCUS_PAIR)
    rows = {}
    for coin in spreads:
        for label, bt in run_strategies(spreads[coin][pair], costs).items():
            rows[(coin, label)] = analysis.backtest_summary(bt)
    return pd.DataFrame(rows).T


def cost_ladder(spreads: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Net annualised return by cost profile and strategy, averaged over coins.

    Averaging across the three coins is a deliberate simplification: the point
    of this table is the *shape* of the response to cost, which is the same for
    all three, not the level for any one of them.
    """
    pair = "/".join(FOCUS_PAIR)
    rows = {}
    for pname, model in analysis.COST_PROFILES.items():
        per_strategy = {s: [] for s in STRATEGIES}
        for coin in spreads:
            for label, bt in run_strategies(spreads[coin][pair], model).items():
                per_strategy[label].append(analysis.backtest_summary(bt)["annualised_return_pct"])
        rows[pname] = {s: sum(v) / len(v) for s, v in per_strategy.items()}
    return pd.DataFrame(rows).T[list(STRATEGIES)]


def load_basis(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Cross-venue price gap per coin, in basis points."""
    out = {}
    for coin in COINS:
        a = fetch.binance_perp_prices(coin, use_cache=use_cache)
        b = fetch.hyperliquid_prices(coin, use_cache=use_cache)
        if a.empty or b.empty:
            continue
        out[coin] = analysis.price_basis(a, b, "binance", "hyperliquid")
    return out


def make_figures(use_cache: bool = True) -> dict[str, str]:
    """Regenerate every figure the README embeds. Returns name -> path."""
    data = load_funding(use_cache=use_cache)
    spreads = build_spreads(data)
    pair, control = "/".join(FOCUS_PAIR), "/".join(CONTROL_PAIR)
    base = analysis.COST_PROFILES[BASE_PROFILE]

    summaries = {coin: analysis.compare_venues(data[coin], common_window=True)
                 for coin in COINS}
    edges = {coin: spreads[coin][pair]["abs_spread"].mean() for coin in COINS}
    quarterly = {coin: analysis.quarterly_profile(spreads[coin][pair]) for coin in COINS}
    runs = {coin: run_strategies(spreads[coin][pair], base) for coin in COINS}

    paths = {
        "venue_funding": plots.plot_venue_funding(summaries),
        "spread_bias": plots.plot_spread_bias(spreads["SOL"][control],
                                              spreads["SOL"][pair], "SOL"),
        "breakeven": plots.plot_breakeven(edges, analysis.COST_PROFILES),
        "strategies": plots.plot_strategies(runs, BASE_PROFILE),
        "cost_ladder": plots.plot_cost_ladder(cost_ladder(spreads)),
        "decay": plots.plot_decay(quarterly),
        "basis": plots.plot_basis(load_basis(use_cache=use_cache)),
    }
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    for name, path in make_figures().items():
        print(f"{name:<16} -> {path}")

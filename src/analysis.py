"""Carry computation, cost modelling and a rule-based backtest.

Two trades are analysed, and the distinction is the point of the repository:

**Single-venue carry.** Hold the perpetual short against a long spot position on
the same venue. Income is that venue's funding rate. This is the classic
cash-and-carry basis trade.

**Cross-venue carry.** Hold the perpetual short on the venue paying the higher
funding and long on the venue paying the lower one. Income is the *spread*
between the two rates. Directional exposure still nets to zero, but the
position is now long one venue's basis and short another's.

All rates entering this module are already normalised to a fraction per hour by
``fetch``, so annualisation is a single constant everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #

@dataclass
class CostModel:
    """Round-trip execution cost of a two-legged position, in basis points.

    A delta-neutral pair has two legs, and each leg is opened and closed, so a
    complete round trip crosses the market four times. ``entry_fee_bps`` and
    ``exit_fee_bps`` are per leg per side, which is how exchange fee schedules
    are quoted.

    Slippage is separated from fees because they behave differently: fees are
    contractual and known in advance, slippage is a realised outcome that
    depends on book depth at the moment of execution. Keeping them apart makes
    it visible which part of the cost is negotiable.
    """

    name: str = "taker/taker"
    entry_fee_bps: float = 4.5    # per leg, per side
    exit_fee_bps: float = 4.5     # per leg, per side
    slippage_bps: float = 1.0     # per leg, per side, assumed
    legs: int = 2

    @property
    def round_trip_bps(self) -> float:
        per_side = self.entry_fee_bps + self.exit_fee_bps + 2 * self.slippage_bps
        return per_side * self.legs

    @property
    def round_trip(self) -> float:
        return self.round_trip_bps / 10_000


# Fee schedules as published by the venues (retail tier, no volume discount).
# The maker/taker profile is the one an execution-aware implementation reaches:
# rest the first leg as a passive limit order, cross only the hedge.
COST_PROFILES = {
    "taker/taker (retail CEX)": CostModel("taker/taker (retail CEX)", 4.5, 4.5, 1.0),
    "maker/taker (mixed)": CostModel("maker/taker (mixed)", 1.0, 4.5, 0.5),
    "maker/maker (passive both legs)": CostModel("maker/maker (passive both legs)", 1.0, 1.0, 0.5),
    "on-chain maker (DEX rebate tier)": CostModel("on-chain maker (DEX rebate tier)", 0.5, 0.5, 0.5),
}


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #

def annualise(rate_per_hour: float | pd.Series) -> float | pd.Series:
    """Per-hour funding rate to an annualised fraction (simple, not compounded)."""
    return rate_per_hour * HOURS_PER_YEAR


def summarise_funding(df: pd.DataFrame) -> pd.Series:
    """Headline statistics for one venue's funding series."""
    rate = df["rate_per_hour"]
    return pd.Series({
        "observations": len(rate),
        "start": df["time"].min(),
        "end": df["time"].max(),
        "interval_hours": df["interval_hours"].iloc[0],
        "annualised_mean_pct": annualise(rate.mean()) * 100,
        "annualised_median_pct": annualise(rate.median()) * 100,
        "annualised_std_pct": annualise(rate.std()) * 100,
        "pct_periods_negative": (rate < 0).mean() * 100,
        "best_period_annualised_pct": annualise(rate.max()) * 100,
        "worst_period_annualised_pct": annualise(rate.min()) * 100,
    })


def compare_venues(frames: dict[str, pd.DataFrame],
                   common_window: bool = True) -> pd.DataFrame:
    """Side-by-side funding statistics.

    With ``common_window`` the comparison is restricted to the overlap of all
    series. Venues have very different history depths, and comparing a venue
    measured over a bull run against one measured over a drawdown says more
    about the sample than about the venue.
    """
    frames = {v: d for v, d in frames.items() if d is not None and not d.empty}
    if not frames:
        return pd.DataFrame()

    if common_window:
        start = max(d["time"].min() for d in frames.values())
        end = min(d["time"].max() for d in frames.values())
        frames = {v: d[(d["time"] >= start) & (d["time"] <= end)] for v, d in frames.items()}

    return pd.DataFrame({v: summarise_funding(d) for v, d in frames.items()}).T


def cumulative_carry(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative gross funding earned by a continuously held neutral position."""
    out = df.copy()
    out["cumulative_gross"] = out["funding_rate"].cumsum()
    out["cumulative_gross_pct"] = out["cumulative_gross"] * 100
    return out


# --------------------------------------------------------------------------- #
# Cross-venue spread
# --------------------------------------------------------------------------- #

def hourly_series(df: pd.DataFrame) -> pd.Series:
    """Per-hour funding rate on a continuous hourly index.

    Venues settling every 8 hours publish one rate covering the whole window;
    spreading it evenly across those hours is what makes an 8-hour venue
    comparable with an hourly one. This is an accrual assumption, not a claim
    about when cash actually moves, and it is exact for expected values over
    windows longer than one settlement.

    The direction of the assignment matters and is easy to get backwards. A
    settlement stamped at 08:00 pays for the window that *ended* at 08:00, so
    the hours 00:00-07:00 accrue at that rate, not at the rate stamped 00:00.
    Forward-filling instead of backward-filling shifts an 8-hour venue a full
    window against an hourly one, which silently corrupts every cross-venue
    spread. Each hour is therefore mapped to the first settlement strictly
    after it, and the same rule is applied to both venues so the spread stays
    internally consistent.
    """
    s = df.set_index("time")["rate_per_hour"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return s

    hours = pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h"),
                          freq="1h", tz="UTC")
    nxt = s.index.searchsorted(hours, side="right")
    covered = nxt < len(s)

    out = pd.Series(np.nan, index=hours, dtype=float)
    out.iloc[covered.nonzero()[0]] = s.to_numpy()[nxt[covered]]
    # Hours past the last settlement belong to a window that has not paid yet;
    # dropping them is preferable to guessing a rate that does not exist.
    return out.dropna()


def venue_spread(a: pd.DataFrame, b: pd.DataFrame,
                 name_a: str, name_b: str) -> pd.DataFrame:
    """Hourly funding spread between two venues, on their overlapping window.

    ``spread`` is ``rate_a - rate_b``. A position long the perpetual on ``b``
    and short the perpetual on ``a`` earns that spread; the sign simply says
    which venue to short.
    """
    sa, sb = hourly_series(a), hourly_series(b)
    idx = sa.index.intersection(sb.index)
    if len(idx) == 0:
        return pd.DataFrame()

    out = pd.DataFrame({
        "time": idx,
        f"rate_{name_a}": sa.loc[idx].to_numpy(),
        f"rate_{name_b}": sb.loc[idx].to_numpy(),
    })
    out["spread"] = out[f"rate_{name_a}"] - out[f"rate_{name_b}"]
    out["abs_spread"] = out["spread"].abs()
    out["annualised_pct"] = annualise(out["spread"]) * 100
    out["abs_annualised_pct"] = annualise(out["abs_spread"]) * 100
    return out


def spread_summary(spreads: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Statistics for every venue pair, in annualised percentage terms.

    ``abs`` columns describe the tradable edge: the strategy chooses which venue
    to short, so it captures the magnitude of the spread, not its sign.
    """
    rows = {}
    for pair, df in spreads.items():
        if df is None or df.empty:
            continue
        rows[pair] = pd.Series({
            "hours": len(df),
            "start": df["time"].min(),
            "end": df["time"].max(),
            "mean_abs_annualised_pct": df["abs_annualised_pct"].mean(),
            "median_abs_annualised_pct": df["abs_annualised_pct"].median(),
            "p90_abs_annualised_pct": df["abs_annualised_pct"].quantile(0.90),
            "pct_hours_above_10pct": (df["abs_annualised_pct"] > 10).mean() * 100,
            "pct_hours_above_25pct": (df["abs_annualised_pct"] > 25).mean() * 100,
            "sign_stability_pct": max((df["spread"] > 0).mean(), (df["spread"] < 0).mean()) * 100,
        })
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Break-even
# --------------------------------------------------------------------------- #

def breakeven_hours(edge_per_hour: float, costs: CostModel) -> float:
    """Hours the position must be held for the edge to repay the round trip."""
    if edge_per_hour <= 0:
        return float("inf")
    return costs.round_trip / edge_per_hour


def breakeven_table(edges: dict[str, float],
                    profiles: dict[str, CostModel] | None = None) -> pd.DataFrame:
    """Minimum holding period, in hours, for each edge and cost profile.

    ``edges`` maps a label to a per-hour edge. The result is the answer to the
    only question that matters operationally: how long must this position
    survive before it has paid for its own execution?
    """
    profiles = profiles or COST_PROFILES
    rows = []
    for label, edge in edges.items():
        row = {"edge": label, "annualised_edge_pct": annualise(edge) * 100}
        for pname, model in profiles.items():
            row[pname] = breakeven_hours(edge, model)
        rows.append(row)
    return pd.DataFrame(rows).set_index("edge")


def cost_sensitivity(edge_per_hour: float, holding_hours: int,
                     max_bps: float = 30.0, steps: int = 61) -> pd.DataFrame:
    """Net annualised return across a range of round-trip cost assumptions."""
    grid = np.linspace(0, max_bps, steps)
    rows = []
    for bps in grid:
        cost_per_hour = (bps / 10_000) / holding_hours
        rows.append({
            "round_trip_bps": bps,
            "net_annualised_pct": annualise(edge_per_hour - cost_per_hour) * 100,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def backtest_spread(df: pd.DataFrame, costs: CostModel,
                    entry_annual_pct: float = 10.0,
                    exit_annual_pct: float = 3.0,
                    lookback_hours: int = 8,
                    min_consistency: float = 0.7,
                    max_hold_hours: int | None = None) -> pd.DataFrame:
    """Trade the cross-venue spread under an explicit entry/exit rule.

    The rule mirrors a live implementation rather than an idealised one:

    * enter only when the *trailing* edge over ``lookback_hours`` clears
      ``entry_annual_pct`` — a single-hour spike is usually a print that
      reverses before a position can be established;
    * additionally require the sign of the spread to have held for at least
      ``min_consistency`` of that window, so the position is not opened into a
      rate that is in the middle of flipping;
    * exit when the trailing edge decays below ``exit_annual_pct``, or after
      ``max_hold_hours`` if set;
    * charge half a round trip on entry and half on exit;
    * and, critically, hold the side chosen *at entry* for the life of the
      position. Funding earned is then the signed spread, which goes negative
      whenever the relationship inverts mid-trade. Recomputing the favourable
      side every hour would be look-ahead: the direction is only knowable after
      the rate prints.

    Returns the full hourly frame so the equity curve and the position mask can
    both be plotted and audited.
    """
    out = df.copy().reset_index(drop=True)

    entry = entry_annual_pct / 100 / HOURS_PER_YEAR
    exit_ = exit_annual_pct / 100 / HOURS_PER_YEAR

    trailing = out["spread"].rolling(lookback_hours, min_periods=lookback_hours).mean()
    out["trailing"] = trailing
    # Share of the lookback whose sign agrees with the trailing mean: the mean
    # of +-1 signs runs from -1 to +1, so rescale it to a 0-1 fraction.
    mean_sign = np.sign(out["spread"]).rolling(lookback_hours, min_periods=lookback_hours).mean()
    out["consistency"] = (mean_sign * np.sign(trailing) + 1) / 2

    in_position = False
    side = 0            # +1 = short the first venue, -1 = short the second
    held = 0
    positions, sides, transitions = [], [], []

    for trail, cons in zip(out["trailing"], out["consistency"]):
        transition = 0
        if not in_position:
            if pd.notna(trail) and abs(trail) > entry and cons >= min_consistency:
                in_position, side, held, transition = True, int(np.sign(trail)), 0, 1
        else:
            held += 1
            decayed = pd.isna(trail) or abs(trail) < exit_
            expired = max_hold_hours is not None and held >= max_hold_hours
            if decayed or expired:
                in_position, side, held, transition = False, 0, 0, 1

        positions.append(1 if in_position else 0)
        sides.append(side)
        transitions.append(transition)

    out["in_position"] = positions
    out["side"] = sides
    out["transition"] = transitions
    out["cost"] = out["transition"] * (costs.round_trip / 2)
    out["gross"] = out["side"] * out["spread"]
    out["pnl"] = out["gross"] - out["cost"]
    out["cumulative_pnl"] = out["pnl"].cumsum()
    out["cumulative_pnl_pct"] = out["cumulative_pnl"] * 100
    out["cumulative_gross_pct"] = out["gross"].cumsum() * 100
    return out


def passive_hold(df: pd.DataFrame, costs: CostModel,
                 lookback_hours: int = 8) -> pd.DataFrame:
    """Benchmark: pick a side once, hold it for the whole sample.

    The side is fixed from the trailing spread over the first ``lookback_hours``
    and never revisited, so the position pays exactly one round trip. This is
    the null hypothesis every active rule has to beat, and it is deliberately
    the dumbest thing that can be done with the signal.
    """
    out = df.copy().reset_index(drop=True)
    warmup = out["spread"].iloc[:lookback_hours].mean()
    side = int(np.sign(warmup)) or 1

    out["trailing"] = out["spread"].rolling(lookback_hours, min_periods=lookback_hours).mean()
    out["consistency"] = np.nan
    out["in_position"] = 0
    out.loc[lookback_hours:, "in_position"] = 1
    out["side"] = out["in_position"] * side

    out["transition"] = 0
    out.loc[lookback_hours, "transition"] = 1          # entry
    out.loc[out.index[-1], "transition"] = 1           # exit at the end of the sample
    out["cost"] = out["transition"] * (costs.round_trip / 2)

    out["gross"] = out["side"] * out["spread"]
    out["pnl"] = out["gross"] - out["cost"]
    out["cumulative_pnl"] = out["pnl"].cumsum()
    out["cumulative_pnl_pct"] = out["cumulative_pnl"] * 100
    out["cumulative_gross_pct"] = out["gross"].cumsum() * 100
    return out


def backtest_slow_side(df: pd.DataFrame, costs: CostModel,
                       lookback_hours: int = 720,
                       rebalance_hours: int = 720) -> pd.DataFrame:
    """Stay in the market permanently; revisit only *which* venue to short.

    This separates the two decisions an operator actually faces. Whether to
    hold a funding position at all is one question; which side of it to be on
    is another, and only the second one is worth paying for. The side is
    re-evaluated on a slow clock from a long trailing mean, and a cost is
    charged only when the side genuinely flips.

    Defaults are 30 days for both the lookback and the rebalance interval,
    chosen because the spread's directional bias is a regime that persists for
    months while its hour-to-hour value mean-reverts within hours.
    """
    out = df.copy().reset_index(drop=True)
    out["trailing"] = out["spread"].rolling(lookback_hours, min_periods=lookback_hours).mean()

    side = 0
    sides, transitions = [], []
    for i, trail in enumerate(out["trailing"]):
        transition = 0
        if pd.isna(trail):
            pass                                   # still warming up, flat
        elif side == 0:
            side = int(np.sign(trail)) or 1
            transition = 1                         # opening: half a round trip
        elif i % rebalance_hours == 0:
            new_side = int(np.sign(trail)) or side
            if new_side != side:
                side = new_side
                transition = 2                     # close and reopen: a full round trip
        sides.append(side)
        transitions.append(transition)

    if sides and sides[-1] != 0:
        transitions[-1] += 1                       # closing the final position

    out["side"] = sides
    out["in_position"] = (out["side"] != 0).astype(int)
    out["transition"] = transitions
    out["consistency"] = np.nan
    out["cost"] = out["transition"] * (costs.round_trip / 2)
    out["gross"] = out["side"] * out["spread"]
    out["pnl"] = out["gross"] - out["cost"]
    out["cumulative_pnl"] = out["pnl"].cumsum()
    out["cumulative_pnl_pct"] = out["cumulative_pnl"] * 100
    out["cumulative_gross_pct"] = out["gross"].cumsum() * 100
    return out


def spread_halflife(df: pd.DataFrame) -> float:
    """Half-life of the spread in hours, from an AR(1) fit.

    Answers whether the edge is a slow-moving state or hour-to-hour noise. A
    long half-life means the signal observed now is still there tomorrow, which
    is what makes holding viable and rotating unnecessary.
    """
    x = df["spread"].to_numpy()
    x0, x1 = x[:-1], x[1:]
    mask = np.isfinite(x0) & np.isfinite(x1)
    if mask.sum() < 10:
        return float("nan")
    x0, x1 = x0[mask], x1[mask]
    phi = np.polyfit(x0 - x0.mean(), x1 - x1.mean(), 1)[0]
    if not 0 < phi < 1:
        return float("nan")
    return float(np.log(0.5) / np.log(phi))


def quarterly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute and signed spread by calendar quarter, annualised.

    A single average over two years hides whether the opportunity is stable or
    decaying. Reporting it per quarter is the cheapest way to find out.
    """
    out = df.copy()
    # Periods carry no timezone, so drop it explicitly rather than letting the
    # conversion do it and warn. The timestamps are UTC throughout.
    out["quarter"] = out["time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("Q")
    grouped = out.groupby("quarter", observed=True).agg(
        hours=("spread", "size"),
        mean_abs_annualised_pct=("abs_annualised_pct", "mean"),
        mean_signed_annualised_pct=("annualised_pct", "mean"),
        pct_hours_above_10pct=("abs_annualised_pct", lambda s: (s > 10).mean() * 100),
    )
    return grouped


def backtest_summary(bt: pd.DataFrame) -> pd.Series:
    """Headline numbers for a backtest run, on notional, not on margin."""
    hours = max(len(bt), 1)
    total = bt["cumulative_pnl"].iloc[-1]
    gross = bt["gross"].sum()

    running_max = bt["cumulative_pnl"].cummax()
    drawdown = bt["cumulative_pnl"] - running_max

    in_pos = bt["in_position"] == 1
    hourly_pnl = bt.loc[in_pos, "gross"]

    return pd.Series({
        "total_return_pct": total * 100,
        "annualised_return_pct": total * (HOURS_PER_YEAR / hours) * 100,
        "gross_return_pct": gross * 100,
        "total_costs_pct": bt["cost"].sum() * 100,
        "cost_share_of_gross_pct": (bt["cost"].sum() / gross * 100) if gross > 0 else np.nan,
        "time_in_market_pct": in_pos.mean() * 100,
        "round_trips": int(bt["transition"].sum() / 2),
        "mean_hold_hours": (in_pos.sum() / max(bt["transition"].sum() / 2, 1)),
        "pct_hours_paying": (hourly_pnl < 0).mean() * 100 if len(hourly_pnl) else np.nan,
        "max_drawdown_pct": drawdown.min() * 100,
    })


def sweep_costs(df: pd.DataFrame, profiles: dict[str, CostModel] | None = None,
                **kwargs) -> pd.DataFrame:
    """Run the same rule under every cost profile.

    Holding the rule fixed and varying only the cost isolates the contribution
    of execution quality, which is the variable an implementation controls.
    """
    profiles = profiles or COST_PROFILES
    rows = {}
    for name, model in profiles.items():
        bt = backtest_spread(df, model, **kwargs)
        rows[name] = backtest_summary(bt)
        rows[name]["round_trip_bps"] = model.round_trip_bps
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Cross-venue price basis - a measured execution cost, not an assumed one
# --------------------------------------------------------------------------- #

def price_basis(price_a: pd.DataFrame, price_b: pd.DataFrame,
                name_a: str, name_b: str) -> pd.DataFrame:
    """Hourly price difference between the two venues, in basis points.

    The two legs of a cross-venue pair are filled on two different order books.
    Whatever gap exists between them at that moment is paid immediately, and it
    is paid again on exit. Unlike slippage, this component can be measured from
    public data instead of assumed.
    """
    a = price_a.set_index("time")["close"].sort_index()
    b = price_b.set_index("time")["close"].sort_index()
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return pd.DataFrame()

    out = pd.DataFrame({
        "time": idx,
        f"price_{name_a}": a.loc[idx].to_numpy(),
        f"price_{name_b}": b.loc[idx].to_numpy(),
    })
    mid = (out[f"price_{name_a}"] + out[f"price_{name_b}"]) / 2
    out["basis_bps"] = (out[f"price_{name_a}"] - out[f"price_{name_b}"]) / mid * 10_000
    out["abs_basis_bps"] = out["basis_bps"].abs()
    return out


def basis_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Distribution of the cross-venue basis per coin, in basis points."""
    rows = {}
    for coin, df in frames.items():
        if df is None or df.empty:
            continue
        rows[coin] = pd.Series({
            "hours": len(df),
            "mean_abs_bps": df["abs_basis_bps"].mean(),
            "median_abs_bps": df["abs_basis_bps"].median(),
            "p90_abs_bps": df["abs_basis_bps"].quantile(0.90),
            "p99_abs_bps": df["abs_basis_bps"].quantile(0.99),
            "max_abs_bps": df["abs_basis_bps"].max(),
        })
    return pd.DataFrame(rows).T

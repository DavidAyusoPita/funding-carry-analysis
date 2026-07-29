"""Charts for the analysis, written to ``figures/`` for embedding in the README.

The palette is a validated categorical set: three hues assigned in a fixed
order and checked for colour-vision-deficiency separation against the chart
surface rather than chosen by eye. Every chart with more than one series
carries a legend, so identity never depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order - never cycled, never reassigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
})


def _save(fig, name: str) -> Path:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _titles(ax, title: str, subtitle: str) -> None:
    """Headline plus a muted deck above the axes.

    Matplotlib keeps one title object per ``loc``, so a centre title and a left
    title land on the same line and overlap. The deck takes the real title slot
    and the headline is offset above it.
    """
    ax.set_title(subtitle, fontsize=9, color=MUTED, loc="left", pad=10)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 30), textcoords="offset points",
                ha="left", va="bottom", fontsize=12.5, color=INK)


def _endpoint_label(ax, x, y, text: str) -> None:
    """Direct label at a line's end. Text stays in ink; the line carries identity."""
    ax.annotate(f" {text}", xy=(x, y), xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK_2)


# --------------------------------------------------------------------------- #

def plot_venue_funding(summaries: dict[str, pd.DataFrame],
                       name: str = "venue_funding") -> Path:
    """Mean annualised funding per venue, grouped by coin, on the common window."""
    coins = list(summaries)
    venues = list(summaries[coins[0]].index)
    x = np.arange(len(venues))
    width = 0.8 / len(coins)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, coin in enumerate(coins):
        vals = summaries[coin]["annualised_mean_pct"].reindex(venues).to_numpy()
        offset = (i - (len(coins) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=coin,
                      color=SERIES[i], linewidth=0)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK_2)

    ax.set_xticks(x, venues)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("Annualised mean funding (%)")
    ax.margins(y=0.18)
    _titles(ax, "The on-chain venue pays more for the same exposure",
            "Perpetual funding, annualised, over the window all four venues share")
    ax.legend(ncol=len(coins), loc="upper right")
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


def plot_spread_bias(cex_cex: pd.DataFrame, cex_dex: pd.DataFrame, coin: str,
                     name: str = "spread_bias") -> Path:
    """Signed spread distributions: one venue pair is noise, the other is biased.

    Drawn as cumulative distributions rather than histograms. Both series have
    a large atom at exactly zero — the hours when two venues print the same
    base rate — and in a density plot that single spike compresses everything
    else into the floor. A CDF absorbs the atom as a step and puts the median
    shift, which is the actual finding, on the axis.
    """
    fig, ax = plt.subplots(figsize=(9, 4.6))

    for i, (df, label) in enumerate(((cex_cex, "Binance vs Bybit   (exchange vs exchange)"),
                                     (cex_dex, "Binance vs Hyperliquid   (exchange vs DEX)"))):
        vals = np.sort(df["annualised_pct"].to_numpy())
        cdf = np.arange(1, len(vals) + 1) / len(vals) * 100
        ax.plot(vals, cdf, color=SERIES[i], label=label, linewidth=2.0)

        median = float(np.median(vals))
        ax.plot([median], [50], marker="o", markersize=7, color=SERIES[i],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(f"median {median:+.1f}%", xy=(median, 50),
                    xytext=(8, -14 if i else 8), textcoords="offset points",
                    fontsize=9, color=INK_2)

    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.axhline(50, color=GRID, linewidth=0.8)
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Funding spread, annualised (%)      negative = the second venue pays more")
    ax.set_ylabel("Share of hours below this spread")
    _titles(ax, "Between two exchanges the spread is noise. Against a DEX it is biased.",
            f"{coin} hourly funding spread, 2024-07 to 2026-07. A centred curve means no edge.")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


def plot_breakeven(edges: dict[str, float], profiles: dict,
                   name: str = "breakeven") -> Path:
    """Hours a position must survive before it has repaid its own execution."""
    profile_names = list(profiles)
    coins = list(edges)
    y = np.arange(len(profile_names))
    height = 0.8 / len(coins)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, coin in enumerate(coins):
        hours = [profiles[p].round_trip / edges[coin] for p in profile_names]
        offset = (i - (len(coins) - 1) / 2) * height
        bars = ax.barh(y + offset, hours, height * 0.92, label=coin,
                       color=SERIES[i], linewidth=0)
        ax.bar_label(bars, fmt="%.0f h", padding=3, fontsize=8, color=INK_2)

    labels = [f"{p}\n{profiles[p].round_trip_bps:.0f} bps round trip" for p in profile_names]
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Hours the position must be held to repay its own execution")
    ax.margins(x=0.12)
    _titles(ax, "At retail fees the trade has to survive over a week to break even",
            "Holding period needed at each venue pair's mean funding spread, 2024-07 to 2026-07.")
    ax.legend(loc="lower right", ncol=len(coins))
    ax.grid(axis="y", visible=False)
    return _save(fig, name)


def plot_strategies(runs: dict[str, dict[str, pd.DataFrame]], profile_label: str,
                    name: str = "strategies") -> Path:
    """Cumulative net return per coin, one panel each, one line per strategy."""
    coins = list(runs)
    fig, axes = plt.subplots(1, len(coins), figsize=(12.5, 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, coin in zip(axes, coins):
        for i, (label, bt) in enumerate(runs[coin].items()):
            ax.plot(bt["time"], bt["cumulative_pnl_pct"], color=SERIES[i],
                    label=label, linewidth=1.9)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_title(coin, loc="left", fontsize=11, color=INK, pad=6)
        ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=8.5)

    axes[0].set_ylabel("Cumulative net return (% of notional)")
    axes[0].legend(loc="lower left", fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.text(0.004, 0.995, "Trading the spread destroys it. Holding it collects it.",
             ha="left", va="top", fontsize=13, color=INK)
    fig.text(0.004, 0.925,
             f"Same signal, same data, three decision rules - net of costs at the "
             f"{profile_label} profile.",
             ha="left", va="top", fontsize=9.5, color=MUTED)
    return _save(fig, name)


def plot_cost_ladder(table: pd.DataFrame, name: str = "cost_ladder") -> Path:
    """Net annualised return by strategy across cost profiles.

    ``table`` is indexed by cost profile with one column per strategy.
    """
    profiles = list(table.index)
    strategies = list(table.columns)
    x = np.arange(len(profiles))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for i, strat in enumerate(strategies):
        offset = (i - (len(strategies) - 1) / 2) * width
        bars = ax.bar(x + offset, table[strat].to_numpy(), width * 0.92,
                      label=strat, color=SERIES[i], linewidth=0)
        ax.bar_label(bars, fmt="%+.1f", padding=3, fontsize=8, color=INK_2)

    ax.set_xticks(x, [p.replace(" (", "\n(") for p in profiles], fontsize=8.5)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylabel("Net annualised return (% of notional)")
    ax.margins(y=0.16)
    _titles(ax, "Cheaper execution does not improve the patient rule. It rescues the impatient one.",
            "Mean of BTC, ETH and SOL on the Binance/Hyperliquid spread, 2024-07 to 2026-07.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(strategies), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


def plot_decay(quarterly: dict[str, pd.DataFrame], name: str = "decay") -> Path:
    """Mean absolute spread per quarter - is the opportunity still there?"""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (coin, df) in enumerate(quarterly.items()):
        labels = df.index.astype(str)
        y = df["mean_abs_annualised_pct"].to_numpy()
        ax.plot(labels, y, color=SERIES[i], label=coin, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.4)

    ax.set_ylabel("Mean absolute spread, annualised (%)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.07)
    _titles(ax, "The opportunity is narrowing, not stable",
            "Binance vs Hyperliquid spread by quarter. The second year averaged 20-35% below the "
            "first; the last quarter is partial.")
    ax.legend(loc="lower left", ncol=3)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8.5)
    return _save(fig, name)


def plot_basis(frames: dict[str, pd.DataFrame], name: str = "basis") -> Path:
    """Distribution of the cross-venue price gap, a measured execution cost."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.linspace(0, 12, 97)
    for i, (coin, df) in enumerate(frames.items()):
        ax.hist(df["abs_basis_bps"].clip(0, 12), bins=bins, density=True,
                histtype="step", linewidth=2.0, color=SERIES[i], label=coin)

    ax.set_xlabel("Absolute price difference between the two venues (bps)")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    _titles(ax, "The two legs are filled on two different books, and the gap is not free",
            "Hourly close-to-close difference, Binance vs Hyperliquid perpetuals.")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return _save(fig, name)

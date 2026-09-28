"""Exploratory Data Analysis (EDA) for the Garena Brazil gun-skin case study.

This module loads the ``mobile_game_inapp_purchases`` dataset (XLSX), cleans it
and produces a set of **English-labelled** figures using the **Garena brand
palette** (red / dark-red / charcoal). It also writes the key summary tables
that are used to justify the pricing and selling strategy in the assignment.

Everything here is *exploratory*: the dataset is a small synthetic sample and
numbers describe that sample only -- they are not market estimates.

Usage (from the repository root)::

    python scripts/run_garena_eda.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT / "data" / "raw" / "archive" / "mobile_game_inapp_purchases.xlsx"
)
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

# ---------------------------------------------------------------------------
# Garena brand palette (inspired by Garena's red / charcoal identity)
# ---------------------------------------------------------------------------
GARENA_RED = "#D92A24"
GARENA_DARK_RED = "#B71C1C"
GARENA_CHARCOAL = "#4A4A4A"
GARENA_GRAY = "#8C8C8C"
GARENA_LIGHT = "#D9D9D9"
GARENA_ORANGE = "#F26522"  # accent, used sparingly

PALETTE = [GARENA_RED, GARENA_CHARCOAL, GARENA_DARK_RED, GARENA_GRAY, GARENA_LIGHT]

SEGMENT_ORDER = ["Whales", "Dolphins", "Minnows"]
SEGMENT_COLORS = {
    "Whales": GARENA_DARK_RED,
    "Dolphins": GARENA_RED,
    "Minnows": GARENA_CHARCOAL,
}

TARGET_GENRES = ["Battle Royale", "Action RPG"]

# Latin-American markets (Portuguese / Spanish speaking).
LATAM = {
    "argentina", "bolivia", "brazil", "chile", "colombia", "costa rica", "cuba",
    "dominican republic", "ecuador", "el salvador", "guatemala", "haiti",
    "honduras", "mexico", "nicaragua", "panama", "paraguay", "peru", "uruguay",
    "venezuela",
}

ORIGINAL_COLUMNS = [
    "UserID", "Age", "Gender", "Country", "Device", "GameGenre", "SessionCount",
    "AverageSessionLength", "SpendingSegment", "InAppPurchaseAmount",
    "FirstPurchaseDaysAfterInstall", "PaymentMethod", "LastPurchaseDate",
]

# Stated assumption for any BRL reference values (USD -> BRL).
USD_TO_BRL = 5.40

_SEGMENT_MAP = {
    "whale": "Whales", "whales": "Whales",
    "dolphin": "Dolphins", "dolphins": "Dolphins",
    "minnow": "Minnows", "minnows": "Minnows",
}


# ---------------------------------------------------------------------------
# Style & helpers
# ---------------------------------------------------------------------------
def set_style() -> None:
    """Apply a consistent Garena-flavoured plotting style."""
    sns.set_theme(style="whitegrid", font_scale=1.02)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": GARENA_CHARCOAL,
        "text.color": "#1F1F1F",
        "axes.labelcolor": "#1F1F1F",
        "xtick.color": GARENA_CHARCOAL,
        "ytick.color": GARENA_CHARCOAL,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    sns.set_palette(PALETTE)


def _savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def _annotate_bars(ax, values, fmt="{:.0f}", pad=0.01, horizontal=False):
    """Lightly annotate bar values."""
    span = (max(values) - min(0, min(values))) or 1
    for patch, value in zip(ax.patches, values):
        if horizontal:
            ax.text(patch.get_width() + span * pad, patch.get_y()
                    + patch.get_height() / 2, fmt.format(value),
                    va="center", fontsize=9, color=GARENA_CHARCOAL)
        else:
            ax.text(patch.get_x() + patch.get_width() / 2,
                    patch.get_height() + span * pad, fmt.format(value),
                    ha="center", fontsize=9, color=GARENA_CHARCOAL)


def _group_of(country_norm: str) -> str:
    if country_norm == "brazil":
        return "Brazil"
    if country_norm in LATAM:
        return "LATAM (ex-Brazil)"
    return "Rest of world"


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
def load_and_clean(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the XLSX and return a cleaned (but not imputed) DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(path)

    # Trim text, keep NaNs untouched.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

    for col in [
        "Age", "SessionCount", "AverageSessionLength",
        "InAppPurchaseAmount", "FirstPurchaseDaysAfterInstall",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["country_norm"] = df["Country"].str.casefold()
    df["genre_norm"] = df["GameGenre"].str.casefold()
    df["segment"] = df["SpendingSegment"].str.casefold().map(_SEGMENT_MAP)
    df["last_purchase_date"] = pd.to_datetime(df["LastPurchaseDate"], errors="coerce")
    df["purchase_month"] = df["last_purchase_date"].dt.to_period("M").astype(str)
    df["group"] = df["country_norm"].apply(_group_of)
    df["is_target_genre"] = df["GameGenre"].isin(TARGET_GENRES)
    return df


# ---------------------------------------------------------------------------
# Figures (all titles / labels in English)
# ---------------------------------------------------------------------------
def fig_missing_values(df: pd.DataFrame) -> None:
    miss = df[ORIGINAL_COLUMNS].isna().mean().mul(100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(miss.index, miss.values, color=GARENA_RED)
    # Highlight columns with any missing data in a darker tone.
    for bar, value in zip(bars, miss.values):
        if value == 0:
            bar.set_color(GARENA_LIGHT)
        ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=9, color=GARENA_CHARCOAL)
    ax.set_title("Data quality: missing values by column")
    ax.set_xlabel("Share of missing values (%)")
    ax.set_ylabel("")
    ax.axvline(0, color=GARENA_CHARCOAL, linewidth=0.8)
    _savefig("01_missing_values.png")


def fig_demographics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["Age"].dropna(), bins=21, color=GARENA_RED, kde=True,
                 ax=axes[0])
    axes[0].set_title("Age distribution")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Players")

    gender = df["Gender"].fillna("Unknown").value_counts()
    sns.barplot(x=gender.index, y=gender.values, ax=axes[1],
                hue=gender.index, palette=[GARENA_RED, GARENA_DARK_RED,
                                           GARENA_CHARCOAL, GARENA_GRAY],
                legend=False)
    _annotate_bars(axes[1], gender.values.tolist())
    axes[1].set_title("Gender split")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Players")

    fig.suptitle("Player demographics", y=1.02, fontweight="bold")
    fig.tight_layout()
    _savefig("02_demographics.png")


def fig_top_countries(df: pd.DataFrame) -> None:
    counts = df["Country"].dropna().value_counts().head(15).sort_values()
    colors = [GARENA_DARK_RED if c == "Brazil" else GARENA_RED for c in counts.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(counts.index, counts.values, color=colors)
    for value, y in zip(counts.values, range(len(counts))):
        ax.text(value + 2, y, str(value), va="center", fontsize=9,
                color=GARENA_CHARCOAL)
    ax.set_title("Records by country (Top 15) | Brazil highlighted")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    _savefig("03_top_countries.png")


def fig_genres(df: pd.DataFrame) -> None:
    counts = df["GameGenre"].dropna().value_counts().sort_values()
    colors = [GARENA_DARK_RED if g in TARGET_GENRES else GARENA_RED
              for g in counts.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(counts.index, counts.values, color=colors)
    for value, y in zip(counts.values, range(len(counts))):
        ax.text(value + 1, y, str(value), va="center", fontsize=9,
                color=GARENA_CHARCOAL)
    ax.set_title("Records by game genre | Target genres (dark red) highlighted")
    ax.set_xlabel("Records")
    ax.set_ylabel("")
    _savefig("04_genres.png")


def fig_device_payment(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    device = df["Device"].fillna("Unknown").value_counts()
    sns.barplot(x=device.index, y=device.values, ax=axes[0], hue=device.index,
                palette=[GARENA_RED, GARENA_DARK_RED, GARENA_GRAY], legend=False)
    _annotate_bars(axes[0], device.values.tolist())
    axes[0].set_title("Records by device")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Records")

    pay = df["PaymentMethod"].fillna("Unknown").value_counts()
    sns.barplot(x=pay.values, y=pay.index, ax=axes[1], hue=pay.index,
                palette=sns.light_palette(GARENA_DARK_RED, n_colors=len(pay),
                                          reverse=True), legend=False)
    axes[1].set_title("Records by payment method")
    axes[1].set_xlabel("Records")
    axes[1].set_ylabel("")

    fig.suptitle("Platform & payment preferences", y=1.02, fontweight="bold")
    fig.tight_layout()
    _savefig("05_device_payment.png")


def fig_engagement(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["SessionCount"], bins=range(1, 24), color=GARENA_RED,
                 ax=axes[0])
    axes[0].set_title("Number of sessions")
    axes[0].set_xlabel("Sessions per player")
    axes[0].set_ylabel("Players")

    sns.histplot(df["AverageSessionLength"], bins=30, color=GARENA_DARK_RED,
                 kde=True, ax=axes[1])
    axes[1].set_title("Average session length")
    axes[1].set_xlabel("Minutes per session")
    axes[1].set_ylabel("Players")

    fig.suptitle("Engagement distribution", y=1.02, fontweight="bold")
    fig.tight_layout()
    _savefig("06_engagement.png")


def fig_iap_distribution(df: pd.DataFrame) -> None:
    spend = df["InAppPurchaseAmount"].dropna()
    positive = spend[spend > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Panel A: log-scale histogram reveals the long tail.
    bins = np.logspace(np.log10(max(positive.min(), 0.1)),
                       np.log10(positive.max()), 40)
    axes[0].hist(positive, bins=bins, color=GARENA_RED)
    axes[0].set_xscale("log")
    axes[0].set_title("In-app purchases (log scale)")
    axes[0].set_xlabel("Purchase amount (USD, log)")
    axes[0].set_ylabel("Players")

    # Panel B: cumulative share stresses top spenders.
    sns.histplot(positive, bins=40, cumulative=True, stat="percent",
                 color=GARENA_CHARCOAL, ax=axes[1])
    axes[1].set_title("Cumulative share of purchasing players")
    axes[1].set_xlabel("Purchase amount (USD)")
    axes[1].set_ylabel("Cumulative % of players")

    fig.suptitle("Distribution of in-app purchase amounts", y=1.02,
                 fontweight="bold")
    fig.tight_layout()
    _savefig("07_iap_distribution.png")


def fig_revenue_concentration(df: pd.DataFrame) -> pd.DataFrame:
    spend = df["InAppPurchaseAmount"].dropna().sort_values(ascending=False)
    n = len(spend)
    cum_users = np.arange(1, n + 1) / n * 100
    cum_rev = np.cumsum(spend.values) / spend.sum() * 100

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(cum_users, cum_rev, color=GARENA_RED, linewidth=2.5,
            label="Cumulative revenue")
    ax.plot([0, 100], [0, 100], linestyle="--", color=GARENA_GRAY,
            label="Perfect equality")
    ax.fill_between(cum_users, cum_rev, cum_users, color=GARENA_RED, alpha=0.12)

    shares = {}
    for top in (1, 5, 10):
        k = int(n * top / 100)
        share = spend.head(k).sum() / spend.sum() * 100
        shares[f"top_{top}pct"] = share
        ax.scatter(cum_users[k - 1], cum_rev[k - 1], color=GARENA_DARK_RED, zorder=5)
        ax.annotate(f"Top {top}% spenders\n= {share:.0f}% of revenue",
                    (cum_users[k - 1], cum_rev[k - 1]),
                    textcoords="offset points", xytext=(12, -18), fontsize=9,
                    color=GARENA_DARK_RED)

    ax.set_title("Revenue concentration (Pareto / Lorenz curve)")
    ax.set_xlabel("Cumulative % of purchasing players (ranked by spend)")
    ax.set_ylabel("Cumulative % of revenue")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    _savefig("08_revenue_concentration.png")
    return shares


def fig_segment_users_vs_revenue(df: pd.DataFrame) -> pd.DataFrame:
    data = df.dropna(subset=["segment"])
    users = data.groupby("segment")["UserID"].nunique().reindex(SEGMENT_ORDER)
    revenue = (data.groupby("segment")["InAppPurchaseAmount"].sum(min_count=1)
               .reindex(SEGMENT_ORDER))
    summary = pd.DataFrame({
        "users": users,
        "user_share_pct": users / users.sum() * 100,
        "revenue_usd": revenue,
        "revenue_share_pct": revenue / revenue.sum() * 100,
        "arpu_usd": revenue / users,
    })
    summary.index.name = "segment"

    plot_df = summary.reset_index().melt(
        id_vars="segment",
        value_vars=["user_share_pct", "revenue_share_pct"],
        var_name="metric", value_name="share",
    )
    plot_df["metric"] = plot_df["metric"].map(
        {"user_share_pct": "% of users", "revenue_share_pct": "% of revenue"}
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=plot_df, x="segment", y="share", hue="metric",
                order=SEGMENT_ORDER, palette=[GARENA_CHARCOAL, GARENA_RED],
                errorbar=None, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, color=GARENA_CHARCOAL)
    ax.set_title("Player-country: user share vs revenue share by spending segment")
    ax.set_xlabel("Spending segment")
    ax.set_ylabel("Share (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(title="")
    _savefig("09_segment_users_vs_revenue.png")
    return summary


def fig_segment_spend(df: pd.DataFrame) -> pd.DataFrame:
    data = df.dropna(subset=["segment", "InAppPurchaseAmount"])
    data = data[data["InAppPurchaseAmount"] > 0]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=data, x="segment", y="InAppPurchaseAmount",
                order=SEGMENT_ORDER, hue="segment", palette=SEGMENT_COLORS,
                legend=False, ax=ax, showfliers=False, width=0.55)
    sns.stripplot(data=data, x="segment", y="InAppPurchaseAmount",
                  order=SEGMENT_ORDER, color=GARENA_CHARCOAL, size=2.2,
                  alpha=0.25, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Spending capacity by segment (willingness to pay)")
    ax.set_xlabel("Spending segment")
    ax.set_ylabel("Purchase amount (USD, log)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    medians = data.groupby("segment")["InAppPurchaseAmount"].median()
    stats = data.groupby("segment")["InAppPurchaseAmount"].agg(
        ["count", "mean", "median", "max"])
    for i, seg in enumerate(SEGMENT_ORDER):
        if seg in medians.index:
            ax.text(i, medians[seg], f"  median ${medians[seg]:,.0f}",
                    va="bottom", fontsize=9, color=GARENA_DARK_RED,
                    fontweight="bold")
    _savefig("10_segment_spend.png")
    stats.index.name = "segment"
    return stats


def fig_first_purchase(df: pd.DataFrame) -> pd.DataFrame:
    days = df["FirstPurchaseDaysAfterInstall"].dropna()
    days = days[days >= 0]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(-0.5, np.ceil(days.max()) + 1.5, 1)
    sns.histplot(days, bins=bins, color=GARENA_RED, ax=ax)
    median = days.median()
    mean = days.mean()
    ax.axvline(median, color=GARENA_DARK_RED, linestyle="--",
               label=f"Median = {median:.0f} days")
    ax.axvline(mean, color=GARENA_CHARCOAL, linestyle=":",
               label=f"Mean = {mean:.1f} days")
    ax.set_title("Conversion window: days from install to first purchase")
    ax.set_xlabel("Days after install")
    ax.set_ylabel("Players with a known first-purchase day")
    ax.legend()
    _savefig("11_first_purchase_window.png")
    return pd.Series({"mean": mean, "median": median, "mode": days.mode().iloc[0],
                      "p75": days.quantile(0.75), "n": len(days)})


def fig_engagement_vs_spend(df: pd.DataFrame) -> dict:
    fields = ["SessionCount", "AverageSessionLength"]
    titles = ["Number of sessions", "Average session length (min)"]
    corr = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, field, title in zip(axes, fields, titles):
        pairs = df.dropna(subset=[field, "InAppPurchaseAmount"])
        rho = pairs[field].corr(pairs["InAppPurchaseAmount"], method="spearman")
        corr[field] = rho
        sns.scatterplot(data=pairs, x=field, y="InAppPurchaseAmount",
                        color=GARENA_RED, alpha=0.45, s=18, ax=ax)
        ceiling = pairs["InAppPurchaseAmount"].quantile(0.99)
        if ceiling and ceiling > 0:
            ax.set_ylim(0, ceiling)
        ax.set_title(f"{title}\nSpearman rank correlation = {rho:.2f}")
        ax.set_xlabel(title)
        ax.set_ylabel("Purchase amount (USD, capped at P99)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    fig.suptitle("Engagement vs spending", y=1.03, fontweight="bold")
    fig.tight_layout()
    _savefig("12_engagement_vs_spend.png")
    return corr


def fig_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Age", "SessionCount", "AverageSessionLength",
            "InAppPurchaseAmount", "FirstPurchaseDaysAfterInstall"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds", vmin=-1, vmax=1,
                square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Spearman rho"}, ax=ax)
    ax.set_title("Spearman correlation between numeric variables")
    fig.tight_layout()
    _savefig("13_correlation_heatmap.png")
    return corr


def fig_brazil_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Brazil", "LATAM (ex-Brazil)", "Rest of world"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A: spend distribution by group (log).
    spend = df.dropna(subset=["InAppPurchaseAmount"])
    spend = spend[spend["InAppPurchaseAmount"] > 0]
    sns.boxplot(data=spend, x="group", y="InAppPurchaseAmount", order=order,
                hue="group", palette=[GARENA_DARK_RED, GARENA_RED,
                                      GARENA_CHARCOAL], legend=False, ax=axes[0],
                showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Purchase amount (USD, log)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("USD")
    axes[0].tick_params(axis="x", rotation=20)

    # Panel B: segment mix (% of users).
    seg = df.dropna(subset=["segment"])
    mix = (seg.groupby(["group", "segment"])["UserID"].nunique()
           .unstack(fill_value=0))
    mix = mix.reindex(order).reindex(columns=SEGMENT_ORDER, fill_value=0)
    mix_pct = mix.div(mix.sum(axis=1), axis=0) * 100
    mix_pct.reset_index().melt(id_vars="group", var_name="segment",
                               value_name="pct").pipe(
        lambda d: sns.barplot(data=d, x="group", y="pct", hue="segment",
                              order=order, hue_order=SEGMENT_ORDER,
                              palette=SEGMENT_COLORS, errorbar=None, ax=axes[1]))
    axes[1].set_title("Spending-segment mix (% of users)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of users (%)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="", fontsize=8)

    # Panel C: median days to first purchase.
    first = df.dropna(subset=["FirstPurchaseDaysAfterInstall"])
    first = first[first["FirstPurchaseDaysAfterInstall"] >= 0]
    med = first.groupby("group")["FirstPurchaseDaysAfterInstall"].median().reindex(order)
    sns.barplot(x=med.index, y=med.values, ax=axes[2], hue=med.index,
                palette=[GARENA_DARK_RED, GARENA_RED, GARENA_CHARCOAL],
                legend=False)
    for value, y in zip(med.values, range(len(med))):
        axes[2].text(y, value + 0.2, f"{value:.0f}", ha="center", fontsize=9,
                     color=GARENA_CHARCOAL)
    axes[2].set_title("Median days to first purchase")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Days")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Brazil benchmark vs LATAM and rest of world", y=1.03,
                 fontweight="bold")
    fig.tight_layout()
    _savefig("14_brazil_benchmark.png")

    summary = df.groupby("group").agg(
        records=("UserID", "nunique"),
        avg_spend=("InAppPurchaseAmount", "mean"),
        median_spend=("InAppPurchaseAmount", "median"),
        avg_sessions=("SessionCount", "mean"),
        avg_session_min=("AverageSessionLength", "mean"),
    ).reindex(order)
    summary.index.name = "group"
    return summary


def fig_price_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """Illustrative RPM tier anchors derived from sample spending capacity."""
    spend = df["InAppPurchaseAmount"].dropna()
    spend = spend[spend > 0]
    pct = {p: np.percentile(spend, p) for p in (25, 50, 90, 99)}

    tiers = [
        ("R  (Rare)", pct[25], "Entry price point"),
        ("SR (Super Rare)", pct[50], "Core recurring spend"),
        ("SSR (Super Super Rare)", pct[90], "Dolphin capacity"),
        ("UR (Ultra Rare)", pct[99], "Whale anchor / gacha ceiling"),
    ]
    anchors = pd.DataFrame(tiers, columns=["tier", "usd", "rationale"])
    anchors["brl"] = anchors["usd"] * USD_TO_BRL

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [GARENA_LIGHT, GARENA_GRAY, GARENA_RED, GARENA_DARK_RED]
    bars = ax.barh(anchors["tier"], anchors["usd"], color=colors)
    for bar, usd, brl in zip(bars, anchors["usd"], anchors["brl"]):
        ax.text(usd * 1.02, bar.get_y() + bar.get_height() / 2,
                f"US$ {usd:,.2f}  (~R$ {brl:,.0f})", va="center", fontsize=9,
                color=GARENA_CHARCOAL)
    ax.set_xscale("log")
    ax.set_title("Illustrative price anchors by gun-skin tier\n"
                 "(derived from sample spending percentiles)")
    ax.set_xlabel("Price anchor (USD, log scale)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    fig.text(0.5, -0.02,
             "ILLUSTRATIVE ONLY. Anchors come from this sample's spending "
             "distribution and are not validated market prices.",
             ha="center", fontsize=8, color=GARENA_GRAY)
    fig.tight_layout()
    _savefig("15_price_ladder.png")
    return anchors


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run() -> dict:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()

    print("=" * 68)
    print("GARENA EDA -- mobile game in-app purchases (Brazil gun-skin case)")
    print("=" * 68)
    print(f"Rows: {len(df):,} | Unique players: {df['UserID'].nunique():,} | "
          f"Columns: {len(ORIGINAL_COLUMNS)}")
    print(f"Players with a recorded purchase: {df['InAppPurchaseAmount'].notna().sum():,}"
          f" ({df['InAppPurchaseAmount'].notna().mean() * 100:.1f}%)")
    print(f"Brazil records: {df['country_norm'].eq('brazil').sum()} | "
          f"Target-genre records: {df['is_target_genre'].sum()}")

    shares = fig_revenue_concentration(df)
    segment_summary = fig_segment_users_vs_revenue(df)
    segment_spend = fig_segment_spend(df)
    first_purchase = fig_first_purchase(df)
    corr = fig_engagement_vs_spend(df)
    fig_missing_values(df)
    fig_demographics(df)
    fig_top_countries(df)
    fig_genres(df)
    fig_device_payment(df)
    fig_engagement(df)
    fig_iap_distribution(df)
    fig_correlation_heatmap(df)
    brazil_summary = fig_brazil_benchmark(df)
    anchors = fig_price_ladder(df)

    # ---- Persist tables ----
    segment_summary.round(2).to_csv(TABLE_DIR / "segment_summary.csv")
    segment_spend.round(2).to_csv(TABLE_DIR / "segment_spend_stats.csv")
    brazil_summary.round(2).to_csv(TABLE_DIR / "brazil_vs_world.csv")
    anchors.round(2).to_csv(TABLE_DIR / "price_anchors.csv", index=False)
    pd.DataFrame([shares]).to_csv(TABLE_DIR / "revenue_concentration.csv",
                                  index=False)

    # ---- Console summary ----
    print("\n-- Revenue share by segment (users vs revenue) --")
    print(segment_summary.round(1).to_string())
    print("\n-- Spending capacity by segment --")
    print(segment_spend.round(2).to_string())
    print("\n-- Conversion window --")
    print(first_purchase.round(1).to_string())
    print("\n-- Engagement vs spending (Spearman) --")
    for k, v in corr.items():
        print(f"   {k}: {v:.2f}")
    print("\n-- Revenue concentration --")
    for k, v in shares.items():
        print(f"   {k}: {v:.1f}%")
    print("\n-- Illustrative tier anchors (USD) --")
    print(anchors[["tier", "usd", "brl", "rationale"]].round(2).to_string(index=False))
    print(f"\nFigures written to: {FIG_DIR}")
    print(f"Tables written to:  {TABLE_DIR}")

    return {
        "revenue_concentration": shares,
        "segment_summary": segment_summary,
        "segment_spend": segment_spend,
        "first_purchase": first_purchase,
        "correlations": corr,
        "price_anchors": anchors,
        "brazil_summary": brazil_summary,
    }

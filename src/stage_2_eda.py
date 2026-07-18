"""
Stage 2 — Exploratory Data Analysis
Dataset Type: Synthetic
Generates comprehensive visualization suite for all locations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

from src.config import (DATA_CLEAN_DIR, EDA_DIR, COLORS, LOCATION_ORDER,
                         PLOT_STYLE, WATERMARK)

plt.rcParams.update(PLOT_STYLE)

def load_data():
    daily = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "daily_admissions.csv"), parse_dates=["Admission Date"])
    monthly = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "monthly_admissions.csv"), parse_dates=["Month"])
    dept = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "department_monthly.csv"), parse_dates=["Month"])
    hourly = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "hourly_profile.csv"))
    dow = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "dow_profile.csv"))
    return daily, monthly, dept, hourly, dow

def stamp(ax):
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes, fontsize=7,
            color="#64748B", ha="right", va="bottom", alpha=0.7)

# ── Chart 1: Monthly trends (all locations) ──
def plot_monthly_trends(monthly):
    fig, ax = plt.subplots(figsize=(14, 6))
    for loc in LOCATION_ORDER:
        d = monthly[monthly["Location"] == loc].sort_values("Month")
        ax.plot(d["Month"], d["Admissions"], color=COLORS[loc], linewidth=1.8,
                label=loc, alpha=0.9)
    ax.set_title("Monthly Admission Trends — All Locations", pad=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Admissions")
    ax.legend(loc="upper left", framealpha=0.3, fontsize=9)
    ax.grid(True, alpha=0.3)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "01_monthly_trends_all.png"))
    plt.close(fig)
    print("  [OK] 01_monthly_trends_all.png")

# ── Chart 2: Individual location monthly trends ──
def plot_individual_trends(monthly):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        d = monthly[monthly["Location"] == loc].sort_values("Month")
        ax.fill_between(d["Month"], d["Admissions"], alpha=0.3, color=COLORS[loc])
        ax.plot(d["Month"], d["Admissions"], color=COLORS[loc], linewidth=1.5)
        ax.set_title(loc, fontsize=12)
        ax.set_ylabel("Admissions")
        ax.grid(True, alpha=0.3)
        stamp(ax)
    fig.suptitle("Monthly Admission Trends — Individual Locations", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "02_monthly_trends_individual.png"))
    plt.close(fig)
    print("  [OK] 02_monthly_trends_individual.png")

# ── Chart 3: Day-of-week patterns ──
def plot_dow(dow):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.12
    x = np.arange(7)
    for i, loc in enumerate(LOCATION_ORDER):
        d = dow[dow["Location"] == loc].copy()
        d["DayOfWeek"] = pd.Categorical(d["DayOfWeek"], categories=day_order, ordered=True)
        d = d.sort_values("DayOfWeek")
        vals = d["Admissions"].values
        if len(vals) == 7:
            ax.bar(x + i * bar_width, vals, bar_width, label=loc, color=COLORS[loc], alpha=0.85)
    ax.set_xticks(x + bar_width * 2.5)
    ax.set_xticklabels(day_order, rotation=30)
    ax.set_title("Day-of-Week Admission Patterns", pad=15)
    ax.set_ylabel("Total Admissions")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "03_day_of_week.png"))
    plt.close(fig)
    print("  [OK] 03_day_of_week.png")

# ── Chart 4: Hourly heatmap ──
def plot_hourly_heatmap(hourly):
    pivot = hourly.pivot_table(index="Location", columns="Hour", values="Admissions", fill_value=0)
    pivot = pivot.reindex(LOCATION_ORDER)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(len(LOCATION_ORDER)))
    ax.set_yticklabels(LOCATION_ORDER)
    ax.set_xlabel("Hour of Day")
    ax.set_title("Hourly Admission Heatmap", pad=15)
    plt.colorbar(im, ax=ax, label="Admissions", shrink=0.8)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "04_hourly_heatmap.png"))
    plt.close(fig)
    print("  [OK] 04_hourly_heatmap.png")

# ── Chart 5: Seasonal decomposition ──
def plot_seasonal_decomposition(monthly):
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    titles = ["Observed", "Trend", "Seasonal", "Residual"]
    
    # Use largest location for decomposition demo
    loc = "Al_Reem"
    d = monthly[monthly["Location"] == loc].sort_values("Month").set_index("Month")
    
    if len(d) >= 24:
        result = seasonal_decompose(d["Admissions"], model="additive", period=12)
        components = [result.observed, result.trend, result.seasonal, result.resid]
        
        for ax, comp, title in zip(axes, components, titles):
            ax.plot(comp.index, comp.values, color=COLORS[loc], linewidth=1.5)
            ax.set_title(f"{title} — {loc}", fontsize=11)
            ax.grid(True, alpha=0.3)
            stamp(ax)
    
    fig.suptitle("Seasonal Decomposition (Additive)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "05_seasonal_decomposition.png"))
    plt.close(fig)
    print("  [OK] 05_seasonal_decomposition.png")

# ── Chart 6: Year-over-year comparison ──
def plot_yoy(monthly):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        d = monthly[monthly["Location"] == loc].copy()
        d["Year"] = d["Month"].dt.year
        d["MonthNum"] = d["Month"].dt.month
        
        for year in sorted(d["Year"].unique()):
            yd = d[d["Year"] == year].sort_values("MonthNum")
            alpha = 0.4 if year < d["Year"].max() else 1.0
            lw = 1.0 if year < d["Year"].max() else 2.0
            ax.plot(yd["MonthNum"], yd["Admissions"], label=str(year),
                    alpha=alpha, linewidth=lw)
        
        ax.set_title(loc, fontsize=11)
        ax.set_xlabel("Month")
        ax.set_ylabel("Admissions")
        ax.legend(fontsize=7, framealpha=0.3, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 13))
        stamp(ax)
    
    fig.suptitle("Year-over-Year Monthly Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "06_year_over_year.png"))
    plt.close(fig)
    print("  [OK] 06_year_over_year.png")

# ── Chart 7: Monthly box plots ──
def plot_monthly_boxplots(daily):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        d = daily[daily["Location"] == loc].copy()
        d["MonthNum"] = d["Admission Date"].dt.month
        data_by_month = [d[d["MonthNum"] == m]["Admissions"].values for m in range(1, 13)]
        bp = ax.boxplot(data_by_month, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[loc])
            patch.set_alpha(0.6)
        ax.set_title(loc, fontsize=11)
        ax.set_xlabel("Month")
        ax.set_ylabel("Daily Admissions")
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        stamp(ax)
    
    fig.suptitle("Monthly Distribution of Daily Admissions", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "07_monthly_boxplots.png"))
    plt.close(fig)
    print("  [OK] 07_monthly_boxplots.png")

# ── Chart 8: Correlation matrix across locations ──
def plot_correlation(monthly):
    pivot = monthly.pivot_table(index="Month", columns="Location", values="Admissions")
    pivot = pivot[LOCATION_ORDER]
    
    # Use only overlapping months
    pivot_clean = pivot.dropna()
    corr = pivot_clean.corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(LOCATION_ORDER)))
    ax.set_xticklabels(LOCATION_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(LOCATION_ORDER)))
    ax.set_yticklabels(LOCATION_ORDER, fontsize=9)
    
    for r in range(len(LOCATION_ORDER)):
        for c in range(len(LOCATION_ORDER)):
            ax.text(c, r, f"{corr.values[r, c]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(corr.values[r, c]) < 0.5 else "white")
    
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Cross-Location Monthly Admission Correlation", pad=15)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "08_correlation_matrix.png"))
    plt.close(fig)
    print("  ✓ 08_correlation_matrix.png")

# ── Chart 9: Department breakdown ──
def plot_department_breakdown(dept):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATION_ORDER):
        ax = axes[i]
        d = dept[dept["Location"] == loc]
        dept_totals = d.groupby("Admitting Department")["Admissions"].sum().sort_values(ascending=False)
        top_depts = dept_totals.head(8)
        colors_dept = plt.cm.Set2(np.linspace(0, 1, len(top_depts)))
        
        ax.barh(range(len(top_depts)), top_depts.values, color=colors_dept, alpha=0.85)
        ax.set_yticks(range(len(top_depts)))
        ax.set_yticklabels(top_depts.index, fontsize=8)
        ax.set_title(loc, fontsize=11)
        ax.set_xlabel("Total Admissions")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        stamp(ax)
    
    fig.suptitle("Top 8 Departments by Admission Volume", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "09_department_breakdown.png"))
    plt.close(fig)
    print("  ✓ 09_department_breakdown.png")

# ── Chart 10: Combined volume summary ──
def plot_volume_summary(monthly):
    total_by_loc = monthly.groupby("Location")["Admissions"].sum().reindex(LOCATION_ORDER)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    bars = ax1.bar(range(len(LOCATION_ORDER)), total_by_loc.values,
                   color=[COLORS[loc] for loc in LOCATION_ORDER], alpha=0.85)
    ax1.set_xticks(range(len(LOCATION_ORDER)))
    ax1.set_xticklabels(LOCATION_ORDER, rotation=30, ha="right", fontsize=9)
    ax1.set_title("Total Admissions by Location", fontsize=12)
    ax1.set_ylabel("Total Admissions")
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, total_by_loc.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=8, color="#E2E8F0")
    stamp(ax1)
    
    # Pie chart
    ax2.pie(total_by_loc.values, labels=LOCATION_ORDER,
            colors=[COLORS[loc] for loc in LOCATION_ORDER],
            autopct="%1.1f%%", textprops={"fontsize": 9, "color": "#E2E8F0"},
            wedgeprops={"edgecolor": "#1E293B", "linewidth": 1.5})
    ax2.set_title("Admission Share by Location", fontsize=12)
    stamp(ax2)
    
    fig.suptitle("Admission Volume Summary", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "10_volume_summary.png"))
    plt.close(fig)
    print("  ✓ 10_volume_summary.png")

def main():
    print("=" * 60)
    print(f"STAGE 2 — EXPLORATORY DATA ANALYSIS  |  {WATERMARK}")
    print("=" * 60)
    
    daily, monthly, dept, hourly, dow = load_data()
    
    print("\nGenerating charts ...")
    plot_monthly_trends(monthly)
    plot_individual_trends(monthly)
    plot_dow(dow)
    plot_hourly_heatmap(hourly)
    plot_seasonal_decomposition(monthly)
    plot_yoy(monthly)
    plot_monthly_boxplots(daily)
    plot_correlation(monthly)
    plot_department_breakdown(dept)
    plot_volume_summary(monthly)
    
    print(f"\n✓ Stage 2 complete — {len(os.listdir(EDA_DIR))} charts saved to {EDA_DIR}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""论文3 定稿图 (v3, 2026-08-07)

Fig 1  全球样本地图: 16 个处理国按改革年份着色, 对照池与未入样区分
Fig 2  各国走势小倍数 (原 Fig 1)
Fig 3  CS 事件研究 (原 Fig 2)
Fig 4  分国 ATT, 按"改革后是否残留法定进入门槛"着色 (原 Fig 3, 分类已按 16 国一手法条重编)
Fig 5  进入门槛梯度: A 面板 = 最低资本/人均GNI vs 国家ATT 散点; B 面板 = 按门槛状态的箱线图

输出 PNG(300dpi) + PDF(矢量) 至 paper3/figures/final/
地图边界: Natural Earth 110m admin-0 (data/ne_110m_countries.geojson)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GRAY = "#2563eb", "#d97706", "#9ca3af"
TEAL, ROSE = "#0f766e", "#be123c"

BASE = ROOT
OUT = os.path.join(BASE, "figures", "final")
os.makedirs(OUT, exist_ok=True)

panel = pd.read_csv(os.path.join(BASE, "data/panel_with_treatment.csv"))
theta = pd.read_csv(os.path.join(BASE, "data/cs_theta_main.csv"), index_col=0)
cohort = pd.read_csv(os.path.join(BASE, "data/cs_cohort_att.csv"), index_col=0)
bar = pd.read_csv(os.path.join(BASE, "data/barrier_analysis_out.csv"))

MAIN = ["Türkiye", "Brazil", "India", "Argentina", "Saudi Arabia", "Mexico",
        "Malaysia", "Egypt, Arab Rep.", "Philippines", "Ghana", "Uruguay", "Ecuador",
        "Bangladesh", "Ethiopia", "Indonesia", "Nigeria"]
CTRL_EXTRA = ["Viet Nam", "Sri Lanka", "Colombia", "Kenya"]
G = panel.drop_duplicates("economy").set_index("economy")["treat_year"].to_dict()
SHORT = {"Egypt, Arab Rep.": "Egypt", "Hong Kong SAR, China": "Hong Kong",
         "Korea, Rep.": "Korea", "Taiwan, China": "Taiwan",
         "Russian Federation": "Russia", "Slovak Republic": "Slovakia"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ============================ Fig 1: 全球样本地图 ============================
iso = pd.read_csv(os.path.join(BASE, "data/wb_iso3.csv"), sep="|")
NAME2ISO = dict(zip(iso["name"], iso["iso3"]))
NAME2ISO.update({"Türkiye": "TUR", "Taiwan, China": "TWN"})   # 世行名与 API 名不一致

code = panel.drop_duplicates("economy").set_index("economy")
ctrl_names = sorted(set(code[code["status"].isin(["always_treated", "never_treated"])].index)
                    | set(CTRL_EXTRA))

treat_iso = {NAME2ISO[e]: int(G[e]) for e in MAIN if e in NAME2ISO}
ctrl_iso = {NAME2ISO[e] for e in ctrl_names if e in NAME2ISO}
never_iso = {NAME2ISO[e] for e in code[code["status"] == "never_treated"].index if e in NAME2ISO}
ctrl_iso -= never_iso

gj = json.load(open(os.path.join(BASE, "data/ne_110m_countries.geojson"), encoding="utf-8"))

# 处理年份 → 离散色 (2012–2022)
years = sorted(set(treat_iso.values()))
cmap = LinearSegmentedColormap.from_list("cohort", ["#fde68a", "#f59e0b", "#b45309", "#7c2d12"])
bounds = np.arange(min(years) - 0.5, max(years) + 1.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N)

NOTIN, CTRLC, NEVERC = "#f3f4f6", "#c7d2e8", TEAL


def polys_of(geom):
    """GeoJSON geometry → 顶点数组列表 (经纬度直接当平面坐标, 等距圆柱)"""
    t, c = geom["type"], geom["coordinates"]
    rings = [c[0]] if t == "Polygon" else [p[0] for p in c]
    return [np.asarray(r) for r in rings]


verts, facecols, edgecols = [], [], []
for feat in gj["features"]:
    p = feat["properties"]
    a3 = p.get("ISO_A3")
    if not a3 or a3 == "-99":
        a3 = p.get("ADM0_A3")
    if p.get("NAME") == "Antarctica":
        continue
    if a3 in treat_iso:
        fc = cmap(norm(treat_iso[a3]))
    elif a3 in never_iso:
        fc = NEVERC
    elif a3 in ctrl_iso:
        fc = CTRLC
    else:
        fc = NOTIN
    for r in polys_of(feat["geometry"]):
        verts.append(r)
        facecols.append(fc)
        edgecols.append("white")

fig, ax = plt.subplots(figsize=(12.4, 6.0), dpi=150)
ax.add_collection(PolyCollection(verts, facecolors=facecols, edgecolors=edgecols, linewidths=0.35))
ax.set_xlim(-180, 180)
ax.set_ylim(-58, 84)
ax.set_aspect(1.0)
ax.axis("off")

# 处理国标注 (国名 + 年份), 手工微调避让
LBL = {
    "Brazil": (-52, -12, "Brazil 2012"), "Türkiye": (35, 40, "Türkiye 2012"),
    "India": (79, 22, "India 2014"), "Argentina": (-65, -36, "Argentina 2015"),
    "Saudi Arabia": (45, 24, "Saudi Arabia 2016"), "Mexico": (-102, 24, "Mexico 2016"),
    "Malaysia": (110, 3, "Malaysia 2017"), "Egypt, Arab Rep.": (30, 27, "Egypt 2018"),
    "Philippines": (122, 13, "Philippines 2019"), "Ghana": (-1, 8, "Ghana 2019"),
    "Uruguay": (-56, -33, "Uruguay 2020"), "Ecuador": (-78, -1, "Ecuador 2020"),
    "Bangladesh": (90, 24, "Bangladesh 2021"), "Ethiopia": (40, 9, "Ethiopia 2021"),
    "Indonesia": (118, -3, "Indonesia 2022"), "Nigeria": (8, 10, "Nigeria 2021"),
}
OFF = {  # 引线终点偏移, 避免压字与互相重叠
    "Ghana": (-30, -28), "Nigeria": (-2, -26), "Ethiopia": (26, -16),
    "Egypt, Arab Rep.": (-26, 8), "Bangladesh": (-26, 26), "Uruguay": (28, -12),
    "Ecuador": (-30, -14), "Malaysia": (-18, -22), "Philippines": (22, 14),
    "Saudi Arabia": (24, -14), "Türkiye": (0, 18), "India": (-16, -20),
    "Indonesia": (14, -22), "Brazil": (18, 8), "Argentina": (-30, -8), "Mexico": (-26, 12),
}
for e, (lon, lat, txt) in LBL.items():
    dx, dy = OFF.get(e, (0, 12))
    ax.annotate(txt, xy=(lon, lat), xytext=(lon + dx, lat + dy),
                fontsize=7.4, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.55,
                                shrinkA=0, shrinkB=1.5))
ax.annotate("Thailand\n(never treated)", xy=(101, 15), xytext=(112, 30),
            fontsize=7.4, color=TEAL, ha="center", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.55, shrinkA=0, shrinkB=1.5))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.032, pad=0.02,
                  ticks=years, aspect=42)
cb.set_label("Year of OPC legalization (treated countries)", fontsize=9, color=INK)
cb.ax.tick_params(labelsize=8, colors=MUTED, length=2)
cb.outline.set_visible(False)

ax.legend(handles=[mpatches.Patch(facecolor=CTRLC, edgecolor="white",
                                  label="Comparison pool (OPC status unchanged, 2006–2024)"),
                   mpatches.Patch(facecolor=NEVERC, edgecolor="white",
                                  label="Never treated (Thailand)"),
                   mpatches.Patch(facecolor=NOTIN, edgecolor="#d1d5db",
                                  label="Not in estimation sample")],
          frameon=False, fontsize=8.2, loc="lower left", bbox_to_anchor=(0.0, 0.02))
save(fig, "fig1_world_map")


# ==================== Fig 2: 各国走势小倍数 (原 Fig 1) ====================
order = sorted(MAIN, key=lambda e: (G[e], e))
fig, axes = plt.subplots(3, 6, figsize=(13, 6.2), dpi=150, sharex=True)
for ax in axes.flat[len(order):]:
    ax.axis("off")
for ax, e in zip(axes.flat, order):
    sub = panel[panel["economy"] == e].sort_values("year")
    ax.plot(sub["year"], np.log(sub["density"]), color=BLUE, lw=1.4)
    g = int(G[e])
    ax.axvline(g - 0.5, color=ORANGE, lw=1.1, ls="--")
    ax.set_title(f"{SHORT.get(e, e)} ({g})", fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xticks([2006, 2015, 2024])
fig.suptitle("Log new LLC registrations per 1,000 working-age adults; dashed line = OPC legalization",
             fontsize=9.5, color=MUTED, y=1.00)
fig.tight_layout()
save(fig, "fig2_country_trends")


# ==================== Fig 3: 事件研究 (原 Fig 2) ====================
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
es = theta.reset_index().rename(columns={"index": "e"})
es["e"] = es["e"].astype(int)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axvline(-0.5, color=MUTED, lw=0.8, ls="--")
ax.errorbar(es["e"], es["theta"], yerr=[es["theta"] - es["lo"], es["hi"] - es["theta"]],
            fmt="o", ms=5.5, color=BLUE, ecolor=BLUE, elinewidth=1.4, capsize=0)
ax.scatter([-1], [0], marker="o", s=32, facecolors="white", edgecolors=BLUE, zorder=5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_xticks(range(-5, 6))
ax.set_xlabel("Years relative to OPC legalization", fontsize=10, color=INK)
ax.set_ylabel("ATT, log new business density", fontsize=10, color=INK)
save(fig, "fig3_event_study")


# ============ Fig 4: 分国 ATT, 按"是否残留法定进入门槛"着色 ============
b = bar.set_index("economy")
cohort = cohort.copy()
cohort["gated"] = [bool(b.loc[e, "high_any"]) if e in b.index else False for e in cohort.index]
cohort["label"] = [f"{SHORT.get(e, e)} ({int(r.g)})" for e, r in cohort.iterrows()]
cohort = cohort.sort_values("att")

fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
ypos = np.arange(len(cohort))
cols = [ORANGE if g else BLUE for g in cohort["gated"]]
ax.barh(ypos, cohort["att"], height=0.62, color=cols, edgecolor="none")
ax.axvline(0, color=INK, lw=0.8)
for grp, c, ls in [(False, BLUE, "-"), (True, ORANGE, "--")]:
    m = cohort.loc[cohort["gated"] == grp, "att"].mean()
    ax.axvline(m, color=c, lw=1.1, ls=ls, alpha=0.85)
    ax.text(m, len(cohort) - 0.2, f"mean {m:+.2f}", fontsize=7.8, color=c,
            ha="center", va="bottom")
ax.set_yticks(ypos)
ax.set_yticklabels(cohort["label"], fontsize=8.5, color=INK)
ax.tick_params(axis="x", colors=MUTED, labelsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.set_ylim(-0.8, len(cohort) + 0.6)
ax.set_xlabel("Country-level post-reform ATT, log new business density", fontsize=10, color=INK)
ax.legend(handles=[mpatches.Patch(color=BLUE, label="No residual statutory entry barrier (6)"),
                   mpatches.Patch(color=ORANGE, label="Residual capital or non-capital barrier (10)")],
          frameon=False, fontsize=8.5, loc="lower right")
save(fig, "fig4_cohort_att")


# ============ Fig 5: 门槛梯度 (A 散点 + B 箱线) ============
d = bar.dropna(subset=["ratio"]).copy()
d["lab"] = [SHORT.get(e, e) for e in d["economy"]]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=150,
                               gridspec_kw={"width_ratios": [1.75, 1.0]})

# --- A: 散点 (x 用 symlog; 8 个零资本国横向抖动以便读出标签) ---
axA.axhline(0, color=MUTED, lw=0.8)
d = d.sort_values("att").reset_index(drop=True)
zero = d["ratio"] == 0
xplot = d["ratio"].astype(float).copy()
xplot[zero] = np.linspace(0.0, 0.040, int(zero.sum()))   # 真值均为 0, 仅作可读性抖动
cols = [ORANGE if g else BLUE for g in d["high_any"]]
axA.scatter(xplot, d["att"], s=64, c=cols, edgecolors="white", linewidths=0.8, zorder=3)
axA.set_xscale("symlog", linthresh=0.05, linscale=0.75)
axA.set_xticks([0, 0.1, 0.25, 0.5, 1, 2, 5, 10])
axA.set_xticklabels(["0", "0.1", "0.25", "0.5", "1", "2", "5", "10"])

for i, r in d.iterrows():
    above = (i % 2 == 0) if zero.iloc[i] else True
    dy, va = (11, "bottom") if above else (-11, "top")
    axA.annotate(r["lab"], xy=(xplot.iloc[i], r["att"]), xytext=(0, dy),
                 textcoords="offset points", fontsize=7.6, color=INK,
                 ha="center", va=va)

rho = pd.Series(d["ratio"]).rank().corr(pd.Series(d["att"]).rank())
axA.text(0.98, 0.03,
         f"Spearman $\\rho$ = {rho:.2f}  [95% CI −0.78, 0.29]",
         transform=axA.transAxes, ha="right", va="bottom", fontsize=8.8, color=MUTED)
axA.axvspan(-0.02, 0.047, color="#eef2f7", zorder=0)
axA.margins(y=0.18)
axA.set_xlim(-0.02, 22)
axA.text(0.012, axA.get_ylim()[1], "no statutory\nminimum capital\n(8 reforms)",
         fontsize=7.4, color=MUTED, style="italic", ha="center", va="top")
for s in ["top", "right"]:
    axA.spines[s].set_visible(False)
axA.grid(color=GRID, lw=0.5)
axA.set_axisbelow(True)
axA.tick_params(colors=MUTED, labelsize=9)
axA.set_xlabel("Statutory minimum capital at reform ÷ GNI per capita (same year, same currency)",
               fontsize=9.5, color=INK)
axA.set_ylabel("Country-level post-reform ATT", fontsize=9.5, color=INK)
axA.set_title("A. Capital requirement and effect size", fontsize=10.5, color=INK, loc="left")

# --- B: 箱线 + 散点 ---
g0 = d.loc[~d["high_any"].astype(bool), "att"].values
g1 = d.loc[d["high_any"].astype(bool), "att"].values
axB.axhline(0, color=MUTED, lw=0.8)
bp = axB.boxplot([g0, g1], widths=0.5, patch_artist=True, showfliers=False,
                 medianprops=dict(color=INK, lw=1.4),
                 whiskerprops=dict(color=MUTED, lw=1.0),
                 capprops=dict(color=MUTED, lw=1.0))
for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
    patch.set_facecolor(c)
    patch.set_alpha(0.20)
    patch.set_edgecolor(c)
    patch.set_linewidth(1.3)
rng = np.random.default_rng(7)
for i, (vals, c) in enumerate([(g0, BLUE), (g1, ORANGE)], start=1):
    axB.scatter(rng.normal(i, 0.055, len(vals)), vals, s=42, color=c,
                edgecolors="white", linewidths=0.7, zorder=3)
    axB.scatter([i], [vals.mean()], marker="D", s=46, color=c,
                edgecolors=INK, linewidths=0.8, zorder=4)
axB.set_xticks([1, 2])
axB.set_xticklabels([f"No residual\nbarrier (n={len(g0)})", f"Residual\nbarrier (n={len(g1)})"],
                    fontsize=9, color=INK)
axB.text(1.30, g0.mean(), f"{g0.mean():+.2f}", fontsize=8.6, color=BLUE, va="center")
axB.text(2.30, g1.mean(), f"{g1.mean():+.2f}", fontsize=8.6, color=ORANGE, va="center")
axB.text(0.5, 0.97, "difference +0.25  [95% CI −0.04, 0.53]",
         transform=axB.transAxes, ha="center", va="top", fontsize=8.8, color=MUTED)
axB.set_xlim(0.45, 2.62)
axB.margins(y=0.16)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)
axB.grid(axis="y", color=GRID, lw=0.5)
axB.set_axisbelow(True)
axB.tick_params(axis="y", colors=MUTED, labelsize=9)
axB.set_ylabel("Country-level post-reform ATT", fontsize=9.5, color=INK)
axB.set_title("B. Any residual statutory entry barrier", fontsize=10.5, color=INK, loc="left")

fig.tight_layout()
save(fig, "fig5_barrier_gradient")

print("\n全部 5 张图已输出 →", OUT)

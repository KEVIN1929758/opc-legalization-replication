# -*- coding: utf-8 -*-
"""论文3 第一版事件研究：OPC 立法 × log 新企业注册密度
TWFE 事件研究（诊断用，非正式估计——正式版须用 Callaway–Sant'Anna）。
样本：主力+可用处理国 vs 始终处理/never-treated 对照池。聚类稳健 SE（国家层面）。
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(ROOT + "/data/panel_with_treatment.csv")

MAIN = ["Türkiye", "Brazil", "India", "Argentina", "Saudi Arabia", "Mexico",
        "Malaysia", "Egypt, Arab Rep.", "Philippines", "Ghana", "Uruguay", "Ecuador",
        "Bangladesh", "Ethiopia", "Indonesia", "Nigeria"]
# 卡塔尔剔除(2002年已合法化); 孟加拉处理年改2021 — 见 opc_law_coding.csv
# 观测期内状态不变者做对照：始终处理 + never-treated + 数据起点在处理之后的四国
CTRL_EXTRA = ["Viet Nam", "Sri Lanka", "Colombia", "Kenya"]
ctrl_mask = df["status"].isin(["always_treated", "never_treated"]) | df["economy"].isin(CTRL_EXTRA)
d = df[df["economy"].isin(MAIN) | ctrl_mask].copy()
d["y"] = np.log(d["density"])

d["treat_year"] = d["treat_year"].where(d["economy"].isin(MAIN))
d["k"] = (d["year"] - d["treat_year"]).clip(-7, 6)  # 端点分箱

K_OMIT = -1
ks = sorted(x for x in d["k"].dropna().unique() if x != K_OMIT)

def run_es(dat, label):
    X_parts, names = [], []
    for k in ks:
        X_parts.append(((dat["k"] == k) & dat["economy"].isin(MAIN)).astype(float).values)
        names.append(int(k))
    E = pd.get_dummies(dat["economy"], drop_first=True).astype(float)
    Y = pd.get_dummies(dat["year"], drop_first=True).astype(float)
    X = np.column_stack(X_parts + [E.values, Y.values, np.ones(len(dat))])
    y = dat["y"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # 国家聚类稳健 SE
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in dat["economy"].unique():
        m = (dat["economy"] == g).values
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = dat["economy"].nunique()
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se = np.sqrt(np.diag(V))[: len(ks)]
    out = pd.DataFrame({"k": names, "b": beta[: len(ks)], "se": se})
    n_treat = dat.loc[dat["economy"].isin(MAIN)].groupby("k").size()
    out["n_treated"] = out["k"].map(n_treat).fillna(0).astype(int)
    print(f"\n== {label} ==  (N={len(dat)}, 国家数={G}, 处理国={dat.loc[dat['economy'].isin(MAIN),'economy'].nunique()})")
    print(out.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    return out

es_full = run_es(d, "全样本")
es_nocovid = run_es(d[~d["year"].isin([2020, 2021])], "剔除 2020–2021")

# ---- 图 ----
INK, MUTED, GRID = "#1a1a2e", "#6b7280", "#e5e7eb"
BLUE, GRAY = "#2563eb", "#9ca3af"
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for es, color, dx, label in [(es_full, BLUE, -0.08, "全样本"),
                             (es_nocovid, GRAY, 0.08, "剔除 2020–21")]:
    x = es["k"] + dx
    ax.errorbar(x, es["b"], yerr=1.96 * es["se"], fmt="o", ms=5, color=color,
                ecolor=color, elinewidth=1.2, capsize=0, label=label)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axvline(-0.5, color=MUTED, lw=0.8, ls="--")
ax.scatter([K_OMIT], [0], marker="o", s=25, facecolors="white", edgecolors=BLUE, zorder=5)
ax.set_xticks(list(range(-7, 7)))
ax.set_xticklabels(["≤-7"] + [str(i) for i in range(-6, 6)] + ["≥6"])
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("事件时间（相对 OPC 立法生效年）", fontsize=10, color=INK)
ax.set_ylabel("log 新注册密度（TWFE 系数，95% CI）", fontsize=10, color=INK)
ax.set_title("OPC 立法与新企业注册密度：事件研究（诊断版）", fontsize=11, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
import os
os.makedirs(ROOT + "/figures", exist_ok=True)
fig.savefig(ROOT + "/figures/event_study_v1.png", bbox_inches="tight")
print("\n图已保存: paper3/figures/event_study_v1.png")

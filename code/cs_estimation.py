# -*- coding: utf-8 -*-
"""论文3 正式估计：Callaway–Sant'Anna 风格 ATT(g,t)（手写实现，无外部估计包依赖）

设计
----
- 结果变量: y = log(新注册 LLC / 千名 15–64 岁人口)
- 处理: OPC(一人公司)合法化, 队列 g = 首个处理年
- 对照池: 观测窗口内处理状态不变者(始终允许 45 国 + never-treated 泰国 +
  数据起点晚于处理的越南/斯里兰卡/哥伦比亚/肯尼亚) + not-yet-treated
- ATT(g,t) = E[y_t − y_{g−1} | 队列 g] − E[y_t − y_{g−1} | 对照(t 期未处理)]
  基期统一为 g−1 (universal base period)
- 聚合: 事件时间 θ(e) 与总体 ATT, 国家等权
- 推断: 国家层面成对聚类自助法 (999 次), 百分位 CI
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd

rng = np.random.default_rng(20260806)
N_BOOT = 999
E_MIN, E_MAX = -5, 5

df = pd.read_csv(ROOT + "/data/panel_with_treatment.csv")
df["y"] = np.log(df["density"])

MAIN = ["Türkiye", "Brazil", "India", "Argentina", "Saudi Arabia", "Mexico",
        "Malaysia", "Egypt, Arab Rep.", "Philippines", "Ghana", "Uruguay", "Ecuador",
        "Bangladesh", "Ethiopia", "Indonesia", "Nigeria"]
# 卡塔尔剔除: 2002年已合法化(Law 5/2002), 2015年改革实为取消最低资本(法条核实 2026-08-06)
CTRL_EXTRA = ["Viet Nam", "Sri Lanka", "Colombia", "Kenya"]

code = df.drop_duplicates("economy").set_index("economy")
G = {e: int(code.loc[e, "treat_year"]) for e in MAIN}
BARRIER = {e: code.loc[e, "barrier"] for e in MAIN}

is_const_ctrl = df["status"].isin(["always_treated", "never_treated"]) | df["economy"].isin(CTRL_EXTRA)
CTRL = sorted(df.loc[is_const_ctrl, "economy"].unique())

# 宽表: economy × year → y
Y = df[df["economy"].isin(MAIN + CTRL)].pivot(index="economy", columns="year", values="y")


def att_components(Y, gmap, ctrl_pool):
    """返回逐(处理国 i, 年 t)的 D_it = (y_it − y_i,base) − 对照均值差分, 及事件时间 e"""
    rows = []
    for i, g in gmap.items():
        if i not in Y.index:
            continue
        base = g - 1
        if base not in Y.columns or pd.isna(Y.loc[i, base]):
            continue
        for t in Y.columns:
            e = t - g
            if e < E_MIN or e > E_MAX or t == base or pd.isna(Y.loc[i, t]):
                continue
            # 对照: 状态不变者 + t 期尚未处理的其他处理国
            notyet = [j for j, gj in gmap.items() if j != i and gj > max(t, base) and j in Y.index]
            pool = [c for c in ctrl_pool if c in Y.index] + notyet
            dc = (Y.loc[pool, t] - Y.loc[pool, base]).dropna()
            if len(dc) < 5:
                continue
            rows.append({"economy": i, "g": g, "t": t, "e": e,
                         "D": (Y.loc[i, t] - Y.loc[i, base]) - dc.mean()})
    return pd.DataFrame(rows)


def aggregate(comp):
    """事件时间 θ(e)(国家等权) 与总体 ATT(先国家内平均后国家间平均)"""
    theta = comp.groupby("e")["D"].mean()
    post = comp[comp["e"] >= 0].groupby("economy")["D"].mean()
    return theta, post.mean()


def bootstrap(Y, gmap, ctrl_pool, n=N_BOOT):
    """国家层面成对自助: 同时重抽处理国与对照国"""
    treated = [i for i in gmap if i in Y.index]
    ctrl = [c for c in ctrl_pool if c in Y.index]
    thetas, overalls = [], []
    for _ in range(n):
        bt = list(rng.choice(treated, len(treated), replace=True))
        bc = list(rng.choice(ctrl, len(ctrl), replace=True))
        # 重命名以保留重复抽样
        gm, frames = {}, []
        for k, i in enumerate(bt):
            nm = f"T{k}"
            gm[nm] = gmap[i]
            frames.append(Y.loc[[i]].rename(index={i: nm}))
        cp = []
        for k, c in enumerate(bc):
            nm = f"C{k}"
            cp.append(nm)
            frames.append(Y.loc[[c]].rename(index={c: nm}))
        Yb = pd.concat(frames)
        comp = att_components(Yb, gm, cp)
        if comp.empty:
            continue
        th, ov = aggregate(comp)
        thetas.append(th)
        overalls.append(ov)
    TH = pd.DataFrame(thetas)
    return TH, np.array(overalls)


def run(label, Y, gmap, ctrl_pool, boot=True):
    comp = att_components(Y, gmap, ctrl_pool)
    theta, overall = aggregate(comp)
    print(f"\n===== {label} =====")
    print(f"处理国 {comp['economy'].nunique()}，对照池 {len([c for c in ctrl_pool if c in Y.index])} 国 + not-yet")
    if boot:
        TH, OV = bootstrap(Y, gmap, ctrl_pool)
        out = pd.DataFrame({"theta": theta})
        out["se"] = TH.std()
        out["lo"] = TH.quantile(0.025)
        out["hi"] = TH.quantile(0.975)
        out["n_cty"] = comp.groupby("e")["economy"].nunique()
        print(out.to_string(float_format=lambda v: f"{v: .4f}"))
        print(f"总体 ATT (post 0..{E_MAX}) = {overall: .4f}"
              f"  se={OV.std():.4f}  95%CI=[{np.quantile(OV,0.025): .4f},{np.quantile(OV,0.975): .4f}]")
        return comp, out, (overall, OV.std(), np.quantile(OV, 0.025), np.quantile(OV, 0.975))
    print(f"总体 ATT = {overall: .4f}")
    return comp, pd.DataFrame({"theta": theta}), (overall, None, None, None)


# ---- 主结果 ----
comp_main, theta_main, ov_main = run("主结果: CS 风格, 17 处理国", Y, G, CTRL)

# 分队列 ATT
print("\n-- 分国家(=分队列)post平均 D --")
cq = comp_main[comp_main["e"] >= 0].groupby("economy").agg(g=("g", "first"), att=("D", "mean"), n=("D", "size"))
cq["barrier"] = [BARRIER[i] if pd.notna(BARRIER[i]) else "" for i in cq.index]
print(cq.sort_values("g").to_string(float_format=lambda v: f"{v: .4f}"))

# ---- 异质性: 进入门槛 ----
low = [i for i in MAIN if BARRIER[i] == "low"]
high = [i for i in MAIN if BARRIER[i] == "high"]
post = comp_main[comp_main["e"] >= 0].groupby("economy")["D"].mean()
print(f"\n-- 门槛异质性 (国家 post 平均 D) --")
print(f"低门槛 ({len(low)}): {post[post.index.isin(low)].mean(): .4f}   国家: {', '.join(low)}")
print(f"高门槛 ({len(high)}): {post[post.index.isin(high)].mean(): .4f}   国家: {', '.join(high)}")
print(f"未分类 ({len(MAIN)-len(low)-len(high)}): {post[~post.index.isin(low+high)].mean(): .4f}")

# ---- 稳健性 ----
# R1: 巴西按 2019 (零门槛 SLU) 编码
G_br19 = dict(G, Brazil=2019)
run("R1: 巴西处理年=2019", Y, G_br19, CTRL)

# R2: 剔除 COVID 年份 (2020–2021 列删除)
Y_nc = Y.drop(columns=[c for c in (2020, 2021) if c in Y.columns])
run("R2: 剔除 2020–2021", Y, G, CTRL) if False else None
comp_nc, theta_nc, ov_nc = run("R2: 剔除 2020–2021", Y_nc, G, CTRL)

# R3: 对照仅 not-yet + never (剔除始终处理池)
CTRL_NY = [c for c in CTRL if code.loc[c, "status"] == "never_treated"]
comp_ny, theta_ny, ov_ny = run("R3: 对照=never+not-yet(无始终处理池)", Y, G, CTRL_NY)

# R5: 剔除埃及 (2017-18 投资法 72/2017 一揽子改革 + 汇率并轨, 捆绑最严重;
#     且为逐国剔除中影响最大者, 注册量在改革年近乎翻倍)
G_noeg = {k: v for k, v in G.items() if k != "Egypt, Arab Rep."}
comp_ne, theta_ne, ov_ne = run("R5: 剔除埃及(捆绑改革)", Y, G_noeg, CTRL)

# R6: 剔除加纳 (法条核实: Act 992 s.171(1) 强制两名董事,
#     故加纳只创设"一股东公司"而非"一人公司", 处理定义不同质)
G_nogh = {k: v for k, v in G.items() if k != "Ghana"}
comp_ng, theta_ng, ov_ng = run("R6: 剔除加纳(非真正一人公司)", Y, G_nogh, CTRL)

# R7: 同时剔除埃及与加纳
G_core = {k: v for k, v in G.items() if k not in ("Egypt, Arab Rep.", "Ghana")}
comp_co, theta_co, ov_co = run("R7: 剔除埃及+加纳", Y, G_core, CTRL)

# R4: 逐国剔除 (leave-one-out) — 只报总体 ATT 范围
print("\n===== R4: 逐国剔除 =====")
loo = {}
for drop in MAIN:
    gm = {k: v for k, v in G.items() if k != drop}
    comp = att_components(Y, gm, CTRL)
    _, ov = aggregate(comp)
    loo[drop] = ov
s = pd.Series(loo).sort_values()
print(f"总体 ATT 范围: [{s.min():.4f} (剔除 {s.idxmin()}), {s.max():.4f} (剔除 {s.idxmax()})]")
print(s.to_string(float_format=lambda v: f"{v: .4f}"))

# ---- 保存 ----
theta_main.to_csv(ROOT + "/data/cs_theta_main.csv", encoding="utf-8-sig")
comp_main.to_csv(ROOT + "/data/cs_components_main.csv", index=False, encoding="utf-8-sig")
cq.to_csv(ROOT + "/data/cs_cohort_att.csv", encoding="utf-8-sig")
theta_nc.to_csv(ROOT + "/data/cs_theta_nocovid.csv", encoding="utf-8-sig")
theta_ny.to_csv(ROOT + "/data/cs_theta_notyet.csv", encoding="utf-8-sig")
print("\n结果已保存到 paper3/data/cs_*.csv")

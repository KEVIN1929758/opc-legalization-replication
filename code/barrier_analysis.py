# -*- coding: utf-8 -*-
"""论文3 §4.3 升级版：进入门槛(最低资本)与处理效应的关系, 含推断

动机
----
v2 手稿的门槛异质性只有 7 低 vs 1 高的组均值, 无区间估计, 且 8 国未分类;
稿中 §4.3/§5.2 自己把"系统编码全部 16 国的法定进入成本"列为首要扩展。
本脚本完成该扩展并补上推断。

测度
----
barrier_ratio = 该国一人公司形式法定最低(实缴)资本 / 同年人均 GNI(本币, NY.GNP.PCAP.CN)
  —— 分子分母同为当年本币, 无需汇率; 与 Djankov et al.(2002)、Klapper et al.(2006)
     "最低资本占人均收入比重"的口径一致。
零最低资本者 ratio = 0。另设 extra_restriction 标记非资本类法定门槛(董事人数、
营业额上限、强制转换阈值等), 供稳健性用。

推断
----
国家层面自助(重抽 16 个处理国的 post 平均 D), 999 次, 报告:
  ① 低/高组均值及其差的百分位 CI
  ② ratio 与国家 ATT 的 Spearman 秩相关及其 CI
  ③ 二分切点的敏感性
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd


def spearman(x, y):
    """秩相关(手写, 无 scipy 依赖; 平均秩处理并列)"""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


rng = np.random.default_rng(20260807)
N_BOOT = 999
BASE = ROOT

att = pd.read_csv(f"{BASE}/data/cs_cohort_att.csv")          # 国家 post 平均 D(来自 cs_estimation.py)
bar = pd.read_csv(f"{BASE}/data/barrier_coding_v2.csv")       # 本次新建: 法定最低资本(本币)
gni = pd.read_csv(f"{BASE}/data/gni_pc_lcu.csv", sep="|")   # 世行 API NY.GNP.PCAP.CN, ISO3 键

d = att.merge(bar, on="economy", how="left", validate="1:1")
d = d.merge(gni, left_on=["iso3", "g"], right_on=["country", "year"], how="left")

d["ratio"] = d["min_capital_lcu"] / d["gni_pc_lcu"]
missing = d[d["ratio"].isna()]
if len(missing):
    print("!! 缺 ratio 的国家(须补):", ", ".join(missing["economy"]))

d = d.sort_values("ratio")
d_all = d.copy()
d = d.dropna(subset=["ratio"]).reset_index(drop=True)   # 未编码国不进推断
if len(d) < len(d_all):
    print(f"!! 仅对已编码的 {len(d)}/{len(d_all)} 国做门槛分析")
print("\n===== 16 国法定进入成本与处理效应 =====")
print(d_all[["economy", "g", "att", "min_capital_lcu", "currency", "gni_pc_lcu",
             "ratio", "cap_confidence"]].to_string(
    index=False, float_format=lambda v: f"{v: .4f}"))


def summarize(sub, label):
    print(f"{label:<28} n={len(sub):<3} mean ATT={sub['att'].mean(): .4f}")


def boot_stat(fn, n=N_BOOT):
    """国家层面自助: 有放回重抽 16 个处理国, 重算统计量"""
    out = []
    idx = np.arange(len(d))
    for _ in range(n):
        b = d.iloc[rng.choice(idx, len(idx), replace=True)]
        v = fn(b)
        if v is not None and np.isfinite(v):
            out.append(v)
    a = np.array(out)
    if a.size == 0:
        return np.nan, np.nan, np.nan
    return a.mean(), np.quantile(a, 0.025), np.quantile(a, 0.975)


# ---- ① 连续测度: 秩相关 ----
rho = spearman(d["ratio"], d["att"])
_, rlo, rhi = boot_stat(lambda b: spearman(b["ratio"], b["att"])
                        if b["ratio"].nunique() > 2 else None)
print(f"\n-- ① Spearman(最低资本/人均GNI, 国家ATT) --")
print(f"rho = {rho: .3f}   自助 95%CI = [{rlo: .3f}, {rhi: .3f}]")

# ---- ② 二分: 切点敏感性 ----
print(f"\n-- ② 低/高二分与组差(切点敏感性) --")
print(f"{'切点(ratio)':<14}{'n低':>4}{'n高':>4}{'低均值':>10}{'高均值':>10}{'差(低-高)':>12}{'95%CI':>26}")
for cut in [0.05, 0.10, 0.25, 0.50, 1.00]:
    lo_m = d.loc[d["ratio"] <= cut, "att"].mean()
    hi_m = d.loc[d["ratio"] > cut, "att"].mean()

    def diff(b, c=cut):
        a1, a2 = b.loc[b["ratio"] <= c, "att"], b.loc[b["ratio"] > c, "att"]
        return a1.mean() - a2.mean() if len(a1) and len(a2) else None

    _, dlo, dhi = boot_stat(diff)
    n_lo = int((d["ratio"] <= cut).sum())
    print(f"{cut:<14.2f}{n_lo:>4}{len(d)-n_lo:>4}{lo_m:>10.4f}{hi_m:>10.4f}"
          f"{lo_m-hi_m:>12.4f}{f'[{dlo: .3f}, {dhi: .3f}]':>26}")

# ---- ③ 含非资本类门槛的稳健性 ----
# nonkap_barrier = 1 仅当存在"真正的独资创办人必须满足才能设立"的非资本法定要求
# (强制第二名董事、前一年营业额门槛、强制公证/持牌秘书等);
# 仅约束重复创办或仅适用于放松版形式者记 0。
print(f"\n-- ③ 稳健性: 资本门槛高 或 存在非资本类法定门槛 → '高' --")
d["high_any"] = (d["ratio"] > 0.25) | (d["nonkap_barrier"] == 1)
summarize(d[~d["high_any"]], "低(资本低且无非资本门槛)")
summarize(d[d["high_any"]], "高(资本高或有非资本门槛)")


def diff_any(b):
    a1, a2 = b.loc[~b["high_any"], "att"], b.loc[b["high_any"], "att"]
    return a1.mean() - a2.mean() if len(a1) and len(a2) else None


m, lo, hi = boot_stat(diff_any)
print(f"差(低-高) = {d.loc[~d['high_any'],'att'].mean()-d.loc[d['high_any'],'att'].mean(): .4f}"
      f"   自助 95%CI = [{lo: .3f}, {hi: .3f}]")

# ---- ④ 与 v2 手稿口径的对照(原 7 低 vs 1 高) ----
print(f"\n-- ④ 对照 v2 手稿原口径 --")
V2_LOW = ["Indonesia", "Uruguay", "Ecuador", "Mexico", "Argentina", "Philippines", "India"]
V2_HIGH = ["Brazil"]
print(f"原低门槛 7 国均值 = {d.loc[d['economy'].isin(V2_LOW),'att'].mean(): .4f}")
print(f"原高门槛(巴西)     = {d.loc[d['economy'].isin(V2_HIGH),'att'].mean(): .4f}")
print(f"原未分类 8 国均值 = {d.loc[~d['economy'].isin(V2_LOW+V2_HIGH),'att'].mean(): .4f}")

# ---- ⑤ 稳健性: 印度门槛只存续 13 个月(2015-05-29 废除 s.11), 改记 0 ----
if "India" in set(d["economy"]):
    d2 = d.copy()
    d2.loc[d2["economy"] == "India", "ratio"] = 0.0
    print(f"\n-- ⑤ 稳健性: 印度资本门槛仅存续 13 个月, 改记 ratio=0 --")
    print(f"Spearman = {spearman(d2['ratio'], d2['att']): .3f}  (主编码 {rho: .3f})")
    for cut in (0.25, 0.50):
        a1 = d2.loc[d2["ratio"] <= cut, "att"]
        a2 = d2.loc[d2["ratio"] > cut, "att"]
        print(f"  切点 {cut:.2f}: 低({len(a1)}) {a1.mean(): .4f}  高({len(a2)}) {a2.mean(): .4f}"
              f"  差 {a1.mean()-a2.mean(): .4f}")

d.to_csv(f"{BASE}/data/barrier_analysis_out.csv", index=False, encoding="utf-8-sig")
print(f"\n结果已保存 → paper3/data/barrier_analysis_out.csv")

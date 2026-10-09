# -*- coding: utf-8 -*-
"""合并处理编码与世行面板，输出队列诊断与未编码优先清单"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import pandas as pd

panel = pd.read_csv(ROOT + "/data/new_llcs_long.csv")
code = pd.read_csv(ROOT + "/data/opc_law_coding.csv")

# CSV 中为避免引号问题去掉了世行名里的逗号，这里映射回去
name_map = {"Korea Rep.": "Korea, Rep.",
            "Hong Kong SAR China": "Hong Kong SAR, China",
            "Taiwan China": "Taiwan, China",
            "Egypt": "Egypt, Arab Rep."}
code["economy"] = code["economy"].replace(name_map)

miss = set(code["economy"]) - set(panel["economy"])
if miss:
    print("!! 编码表中有名称匹配不上面板:", miss)

df = panel.merge(code, on="economy", how="left")
df["status"] = df["status"].fillna("uncoded")

cov = panel.groupby("economy")["year"].agg(["min", "max", "count"])

print("== 队列构成（经济体数）==")
st = code.set_index("economy")["status"]
econ = pd.DataFrame({"status": [st.get(e, "uncoded") for e in cov.index]}, index=cov.index)
print(econ["status"].value_counts().to_string())

print("\n== 窗口内处理组：可用性核查 ==")
tw = code[code["status"] == "treated_in_window"].copy()
print(f"{'国家':<24}{'处理年':<7}{'数据覆盖':<12}{'预处理':<7}{'后处理':<7}{'判定'}")
for _, r in tw.sort_values("treat_year").iterrows():
    e = r["economy"]
    if e not in cov.index:
        print(f"{e:<24}{int(r['treat_year']):<7}{'无数据':<12}")
        continue
    c = cov.loc[e]
    ty = int(r["treat_year"])
    sub = panel[panel["economy"] == e]
    pre = (sub["year"] < ty).sum()
    post = (sub["year"] >= ty).sum()
    if pre >= 4 and post >= 3:
        verdict = "主力"
    elif pre >= 2 and post >= 2:
        verdict = "可用"
    elif pre == 0:
        verdict = "无预处理→归已处理池"
    else:
        verdict = "边缘"
    print(f"{e:<24}{ty:<7}{f'{c['min']}-{c['max']}':<12}{pre:<7}{post:<7}{verdict}")

print("\n== 未编码经济体（按覆盖年数排序，前 40 ——下一轮编码优先级）==")
unc = econ[econ["status"] == "uncoded"].join(cov).sort_values("count", ascending=False)
for e, r in unc.head(40).iterrows():
    print(f"  {e:<34}{r['min']}-{r['max']}  {r['count']}年")
print(f"  ……共 {len(unc)} 个未编码")

out = ROOT + "/data/panel_with_treatment.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n已保存合并面板: {out}")

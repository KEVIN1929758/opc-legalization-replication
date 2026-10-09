# -*- coding: utf-8 -*-
"""论文3 覆盖诊断：世界银行 Entrepreneurship Database (ED2025 New LLCs)"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import pandas as pd

XLSX = ROOT + "/data/ED2025_New_LLCs.xlsx"

df = pd.read_excel(XLSX, sheet_name="NEW LLCs")
df.columns = ["economy", "adult_pop", "year", "n_new", "density"]
dropped = df[df["year"].isna() | df["economy"].isna()]
if len(dropped):
    print(f"剔除 {len(dropped)} 行非数据行（脚注/空行）:")
    for v in dropped["economy"].dropna().unique():
        print(f"  {str(v)[:100]}")
df = df.dropna(subset=["year", "economy"]).copy()
df["year"] = df["year"].astype(int)

# 名称清洗：脚注星号与首尾空格；* = Eurostat 离岸金融中心名单
df["offshore"] = df["economy"].str.match(r"^[^*]+\*$")  # 恰好一个星号
df["economy"] = df["economy"].str.replace(r"\*+$", "", regex=True).str.strip()
dup = df.duplicated(subset=["economy", "year"], keep=False)
if dup.any():
    print(f"清洗后出现重复 economy-year {dup.sum()} 行（如 'Brazil'/'Brazil '），逐条检查:")
    print(df[dup].sort_values(["economy", "year"])[["economy", "year", "n_new"]].to_string())
    df = df.drop_duplicates(subset=["economy", "year"], keep="first")

print(f"总行数 {len(df)}，经济体数 {df['economy'].nunique()}，年份 {df['year'].min()}–{df['year'].max()}")

# 每个经济体的覆盖区间与年数
cov = df.groupby("economy")["year"].agg(["min", "max", "count"])
cov["span"] = cov["max"] - cov["min"] + 1
cov["gaps"] = cov["span"] - cov["count"]

print("\n== 面板平衡度 ==")
for k in [19, 15, 10, 5]:
    print(f"覆盖 >= {k} 年的经济体: {(cov['count'] >= k).sum()}")
print(f"有内部缺口的经济体: {(cov['gaps'] > 0).sum()}")

# 处理组国家：名称候选（世行命名可能不同）
treat = {
    "Vietnam 2006-07": ["Vietnam", "Viet Nam"],
    "Sri Lanka 2007": ["Sri Lanka"],
    "Colombia 2008": ["Colombia"],
    "Brazil 2012-01": ["Brazil"],
    "Turkiye 2012-07": ["Turkey", "Türkiye", "Turkiye"],
    "India 2014-04": ["India"],
    "Malaysia 2017-01": ["Malaysia"],
    "Philippines 2019": ["Philippines"],
    "Bangladesh 2020": ["Bangladesh"],
    "Nigeria 2021-01": ["Nigeria"],
    "Indonesia 2021": ["Indonesia"],
}
treat_year = {"Vietnam 2006-07": 2006, "Sri Lanka 2007": 2007, "Colombia 2008": 2008,
              "Brazil 2012-01": 2012, "Turkiye 2012-07": 2012, "India 2014-04": 2014,
              "Malaysia 2017-01": 2017, "Philippines 2019": 2019, "Bangladesh 2020": 2020,
              "Nigeria 2021-01": 2021, "Indonesia 2021": 2021}

print("\n== 处理组国家覆盖（标签含生效年月）==")
print(f"{'处理国':<22}{'世行名':<14}{'覆盖':<12}{'年数':<5}{'缺口':<5}{'预处理年':<9}{'后处理年'}")
for label, names in treat.items():
    hit = [n for n in names if n in cov.index]
    if not hit:
        print(f"{label:<22}{'!! 不在库中':<14}")
        continue
    n = hit[0]
    r = cov.loc[n]
    ty = treat_year[label]
    sub = df[df["economy"] == n]
    pre = (sub["year"] < ty).sum()
    post = (sub["year"] >= ty).sum()
    print(f"{label:<22}{n:<14}{f'{r['min']}-{r['max']}':<12}{r['count']:<5}{r['gaps']:<5}{pre:<9}{post}")

# 中国及主要"已处理对照"
print("\n== 已处理对照池样例 ==")
for n in ["China", "Hong Kong SAR, China", "Taiwan, China", "United States", "Germany", "France", "United Kingdom", "Japan",
          "Korea, Rep.", "Singapore", "Pakistan", "Chile", "Peru", "Thailand", "Egypt, Arab Rep.",
          "Mexico", "Argentina", "Saudi Arabia", "South Africa", "Kenya", "Italy", "Spain"]:
    if n in cov.index:
        r = cov.loc[n]
        print(f"  {n:<22}{r['min']}-{r['max']}  {r['count']}年  缺口{r['gaps']}")
    else:
        print(f"  {n:<22}!! 不在库中")

# 缺口最大的国家（数据质量红旗）
print("\n== 内部缺口 >= 3 年的经济体 ==")
bad = cov[cov["gaps"] >= 3].sort_values("gaps", ascending=False)
for n, r in bad.iterrows():
    print(f"  {n:<30}{r['min']}-{r['max']}  {r['count']}年  缺口{r['gaps']}")

# 保存整理后的长表
out = ROOT + "/data/new_llcs_long.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n已保存整理后长表: {out}")

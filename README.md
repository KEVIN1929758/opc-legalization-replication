# Replication package: One-person company legalization and business formation

Replication materials for the manuscript *Does Legalizing the One-Person Company Spur Business Formation? Evidence from Staggered Reforms across 16 Countries*.

The package reproduces every estimate, table and figure in the paper from public data and an original legal coding of one-person-company (OPC) reforms.

## Contents

```
code/
  coverage_check.py     1. Cleans the World Bank Entrepreneurship Database (ED2025) into a country-year panel
  merge_treatment.py    2. Merges the OPC legal coding onto the panel
  cs_estimation.py      3. Callaway–Sant'Anna event study, pair-cluster bootstrap, robustness (Tables 2–3)
  barrier_analysis.py   4. Statutory entry-cost analysis (Section 4.3, Figure 5)
  make_figures.py       5. Figures 1–5
  event_study_v1.py     Conventional two-way fixed-effects event study (Section 4.2, comparison only)
data/
  ED2025_New_LLCs.xlsx         Source: World Bank Entrepreneurship Database, 2025 vintage (new LLC registrations)
  gni_pc_lcu.csv               Source: World Bank WDI, GNI per capita in current local currency (NY.GNP.PCAP.CN)
  wb_iso3.csv                  Economy name ↔ ISO3 crosswalk
  ne_110m_countries.geojson    Source: Natural Earth 1:110m country boundaries (public domain), Figure 1 only
  opc_law_coding.csv           Original coding: OPC legal status and effective dates for 71 economies
  barrier_coding_v2.csv        Original coding: statutory minimum capital and non-capital entry gates, 16 reforms
  new_llcs_long.csv            Derived by step 1
  panel_with_treatment.csv     Derived by step 2
  cs_*.csv                     Derived by step 3 (event-time estimates, country-level effects, components)
  barrier_analysis_out.csv     Derived by step 4
figures/final/                 Figures 1–5 as PNG (300 dpi) and PDF (vector)
docs/data_appendix.md          Legal sources for every coded reform, with per-claim URLs and verification tiers
```

## Requirements

Python 3.12 with `pandas`, `numpy`, `matplotlib` and `openpyxl`. The results in the paper were produced with pandas 3.0.5, numpy 2.5.1, matplotlib 3.11.1 and openpyxl 3.1.5 (`pip install -r requirements.txt`).

## How to reproduce

Run the scripts in order from any working directory; paths are resolved relative to the package root.

```
python code/coverage_check.py
python code/merge_treatment.py
python code/cs_estimation.py      # about 10 minutes: 999 bootstrap replications per specification
python code/barrier_analysis.py
python code/make_figures.py
```

The bootstrap uses a fixed seed (`numpy.random.default_rng(20260806)`), so the confidence intervals reproduce exactly.

## Mapping to the paper

| Paper | Output |
|---|---|
| Table 1 | `data/barrier_coding_v2.csv`, `data/barrier_analysis_out.csv`; sources in `docs/data_appendix.md` |
| Table 2, Figure 3 | `data/cs_theta_main.csv` |
| Table 3 | console output of `cs_estimation.py`; `data/cs_theta_nocovid.csv`, `data/cs_theta_notyet.csv` |
| Figure 4 | `data/cs_cohort_att.csv` |
| Section 4.3, Figure 5 | console output of `barrier_analysis.py`; `data/barrier_analysis_out.csv` |

## Notes on the coding files

- `confidence` in `opc_law_coding.csv` records the verification tier of each coded date. `docs/data_appendix.md` gives the source for each of the sixteen treated reforms.
- The free-text `notes` column in the coding files and the console messages printed by the scripts are in Chinese; they are working notes and are not used in any computation.

## Licence

Code: MIT License (see `LICENSE`). Original coding files (`opc_law_coding.csv`, `barrier_coding_v2.csv`, `docs/data_appendix.md`): CC BY 4.0. Third-party data remain under their providers' terms: the World Bank Entrepreneurship Database and WDI (please cite the World Bank as the source), and Natural Earth (public domain).

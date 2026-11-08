# Urban–Rural Gaps in Household Living Conditions in Nepal — replication code

Code and aggregate results for the working paper *Urban–Rural Gaps in Household Living Conditions
in Nepal: Evidence from the 2022 Nepal Demographic and Health Survey* (Porokmi Sambahamphe,
November 2026).

**[Read the working paper (PDF)](outputs/paper/nepal_housing_working_paper.pdf)**

**No survey microdata are included.** The NDHS 2022 data may not be redistributed; you need your
own access from The DHS Program (instructions below).

## What is here
| File | What it is |
|---|---|
| `nepal_housing_analysis.py` | The whole analysis, in order, as `# %%` cells (opens as a notebook in VS Code). Every scoring rule is written out explicitly; the script stops on any unrecognised category. |
| `nepal_housing_analysis.html` | The script run from start to finish, with every table, chart, finding and interpretation shown. **Read this if you don't have the data.** Household-level rows are hidden. |
| `outputs/tables/` | Every table the script produces (CSV, aggregate results only). |
| `outputs/figures/` | Every chart (PNG). Figure numbers in file names follow the script, not the paper. |
| `data_external/` | Census 2021 district absentee counts (Annex 5 of NSO, *Internal Migration in Nepal*, 2025), transcribed and checked against the published national total. |
| `outputs/paper/` | The working paper (PDF). |

## How to reproduce
1. **Get the data.** Register at [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm),
   request the Nepal 2022 DHS, and download the household recode in Stata format (`NPHR82DT.ZIP`).
   Unzip and place `NPHR82FL.DTA` at `data/raw/NPHR82FL.DTA` (or set `NDHS_HR_FILE` to its path).
2. **Set up Python 3.11.**
   ```
   python3.11 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt
   ```
3. **Run.** `MPLBACKEND=Agg ./.venv/bin/python nepal_housing_analysis.py` (about 5–10 minutes,
   mostly bootstrap resamples). It recreates the analysis tables and figures in `outputs/`. Or open the script in VS Code
   and run the cells one at a time.

## Main methods
Household-weighted estimates (`hv005`/1,000,000); weighted least squares with standard errors
clustered by primary sampling unit, plus a design-based check that adds the 14 strata; an asset
index from the first principal component of eleven items that do not enter the outcomes
(Filmer and Pritchett 2001); cluster-bootstrap intervals for the joint-deprivation and targeting
comparisons. Details and every result are in the HTML file.

## Use of AI tools
Claude (Anthropic) and ChatGPT (OpenAI) assisted with writing and checking this code.

## Suggested citation
Sambahamphe, Porokmi. 2026. “Urban–Rural Gaps in Household Living Conditions in Nepal: Evidence from the 2022 Nepal Demographic and Health Survey.” Working paper, November. [PDF](outputs/paper/nepal_housing_working_paper.pdf).

## Contact and license
Porokmi Sambahamphe, psambahamphe@fordham.edu, ORCID [0009-0005-9446-7676](https://orcid.org/0009-0005-9446-7676). The code is licensed under the MIT License. The paper is © the author.

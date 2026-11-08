# %% [markdown]
# # Urban–Rural Gaps in Household Living Conditions in Nepal
# Analysis code for the working paper (Porokmi Sambahamphe, November 2026).
#
# **Data:** Nepal DHS 2022 household file (`NPHR82FL.DTA`) and Census 2021 Annex 5
# (absentees abroad by district).
#
# **Figure numbers** in this script differ from the paper's (the paper numbers figures in order of
# first mention).
#
# **Conventions**
# - Percentages use the DHS household weight (`hv005 / 1,000,000`) and describe households.
# - Regressions: weighted least squares, standard errors clustered by survey cluster (`hv021`).
#   Section 10.1b adds a design-based check that also uses the 14 sampling strata.
# - Charts: urban = blue, rural = orange.

# %% [markdown]
# ## 0. Setup
# ### 0.1 Libraries

# %%
# pyright: reportMissingImports=false
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
from patsy import dmatrix
import matplotlib.pyplot as plt
from IPython.display import display

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 150)

# %% [markdown]
# ### 0.2 File locations
# The DHS household file may not be shared. Put it (or a link to it) at `data/raw/NPHR82FL.DTA`,
# or set the environment variable `NDHS_HR_FILE` to its location. `data/` is never committed.

# %%
DHS_FILE = Path(os.environ.get("NDHS_HR_FILE", Path("data") / "raw" / "NPHR82FL.DTA"))
CENSUS_FILE = Path("data_external") / "annex5_raw.txt"
TABLES = Path("outputs") / "tables"
FIGURES = Path("outputs") / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

print("DHS file found:", DHS_FILE.exists())
print("Census file found:", CENSUS_FILE.exists())

# Public render: household-level rows are never shown (DHS data may not be redistributed).
PUBLIC_RENDER = os.environ.get("PUBLIC_RENDER") == "1"


def peek(df):
    """Show the first rows while working; in the public render show only the shape and columns."""
    if PUBLIC_RENDER:
        print(f"[household rows hidden in public version] {df.shape[0]:,} rows × {df.shape[1]} columns:",
              ", ".join(map(str, df.columns)))
    else:
        display(df.head())


# %% [markdown]
# ### 0.3 Four small helper functions and chart colours

# %%
URBAN_COLOR, RURAL_COLOR, GREY = "#2a78d6", "#eb6834", "#8a8984"
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # light = low, dark = high

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False})


def wpct(x, w):
    """Weighted percentage of a 0/1 variable (missing values ignored)."""
    keep = x.notna()
    return 100 * (x[keep] * w[keep]).sum() / w[keep].sum()


def save_table(table, name):
    """Save a table as outputs/tables/<name>.csv."""
    table.to_csv(TABLES / f"{name}.csv")


def save_figure(fig, name):
    """Save a chart as outputs/figures/<name>.png and show it."""
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.show()


def classify(column, scores):
    """Turn text labels into 1 / 0 / missing using a dictionary.
    Stops with an error if the data contain a label the dictionary does not list."""
    labels = column.astype(object)
    unknown = set(labels.dropna().astype(str)) - set(scores)
    if unknown:
        raise ValueError(f"No score for: {unknown}")
    return labels.map(lambda v: np.nan if pd.isna(v) or scores[str(v)] is None else scores[str(v)])


# %% [markdown]
# ## 1. Load the data
# ### 1.1 DHS household file
# `read_stata` converts DHS codes into text labels (e.g. `hv025` → "urban"/"rural").
# One row = one household.

# %%
raw = pd.read_stata(DHS_FILE)
print(raw.shape[0], "households,", raw.shape[1], "variables")
peek(raw[["hv021", "hv023", "hv005", "hv025", "hv024", "shecoreg", "shdist"]])

# %% [markdown]
# ### 1.2 DHS variables used
# | Purpose | Variables |
# |---|---|
# | Survey design | `hv021` cluster, `hv023` stratum, `hv005` weight |
# | Geography | `hv025` urban/rural, `hv024` province, `shecoreg` zone, `shdist` district |
# | Basic services | `hv206` electricity, `hv201` water source, `hv205` toilet, `hv225` toilet shared |
# | Cooking and water | `hv226` cooking fuel, `hv204` time to water |
# | Housing | `hv213` floor, `hv214` walls, `hv215` roof, `hv009` members, `hv216` sleeping rooms |
# | Assets | `hv207` radio, `hv208` television, `hv209` refrigerator, `hv210` bicycle, `hv211` motorcycle, `hv212` car/truck/tractor, `hv221` landline, `hv243a` mobile phone, `hv243b` watch, `hv243e` computer, `hv247` bank account |
# | Household head | `hv219` sex, `hv220` age, `hv101_xx`/`hv106_xx` relationship and education |
# | Comparison only | `hv270` DHS wealth quintile, `hv012` household members (de jure) |

# %% [markdown]
# ## 2. Scoring rules
# Each label gets **1**, **0** or **None** (= missing). `classify` stops if a label is not listed.
#
# ### 2.1 Drinking water (`hv201`) — WHO/UNICEF "improved source"
# "Improved" is a type of source; it does not mean the water is safe.

# %%
WATER = {
    "piped into dwelling": 1, "piped to yard/plot": 1, "piped to neighbor": 1,
    "public tap/standpipe": 1, "tube well or borehole": 1, "protected well": 1,
    "protected spring": 1, "tanker truck": 1, "bottled water": 1,
    "unprotected well": 0, "unprotected spring": 0,
    "river/dam/lake/ponds/stream/canal/irrigation channel": 0,
    "other": None,
}

# %% [markdown]
# ### 2.2 Toilet (`hv205`) — improved types
# Biogas toilets count as improved (Nepal DHS report); "flush, don't know where" is improved (DHS guidance).

# %%
TOILET = {
    "flush to piped sewer system": 1, "flush to septic tank": 1, "flush to pit latrine": 1,
    "flush, don't know where": 1, "ventilated improved pit latrine (vip)": 1,
    "pit latrine with slab": 1, "composting toilet with slab": 1, "biogas attached toilet": 1,
    "flush to somewhere else": 0, "pit latrine without slab/open pit": 0,
    "no facility/bush/field": 0, "bucket toilet": 0, "composting toilet without slab": 0,
    "other": None,
}

# %% [markdown]
# ### 2.3 Cooking fuel (`hv226`) — WHO clean fuels
# Households that do not cook at home are missing.

# %%
FUEL = {
    "electricity": 1, "solar energy": 1, "liquefied petroleum gas (lpg)/cooking gas": 1,
    "piped natural gas": 1, "biogas": 1,
    "gasoline/diesel": 0, "kerosene/paraffin": 0, "coal/lignite": 0, "charcoal": 0, "wood": 0,
    "straw/shrubs/grass": 0, "agricultural crop": 0, "animal dung/waste": 0,
    "processed biomass (pellets) or woodchips": 0, "sawdust": 0,
    "no food cooked in house": None,
}

# %% [markdown]
# ### 2.4 Building materials (`hv213`, `hv214`, `hv215`) — DHS "finished" materials

# %%
FLOOR = {
    "parquet or polished wood": 1, "vinyl or asphalt strips": 1, "ceramic tiles": 1,
    "cement": 1, "carpet": 1,
    "earth/sand": 0, "dung": 0, "wood planks": 0, "palm/bamboo": 0, "other": None,
}
WALL = {
    "cement": 1, "stone with lime/cement": 1, "bricks": 1, "cement blocks": 1,
    "wood planks/shingles": 1,
    "no walls": 0, "cane/palm/trunks": 0, "mud/sand": 0, "bamboo with mud": 0,
    "stone with mud": 0, "metal/galvanized sheet": 0, "plywood": 0, "cardboard": 0,
    "reused wood": 0, "other": None,
}
ROOF = {
    "metal/galvanized sheet": 1, "wood": 1, "calamine/cement fiber": 1, "ceramic tiles": 1,
    "cement": 1, "roofing shingles": 1,
    "no roof": 0, "thatch/palm leaf": 0, "mud": 0, "rustic mat": 0, "palm/bamboo": 0,
    "wood planks": 0, "other": None,
}

# Robustness: the first notebook's shorter lists of "finished" materials
ORIGINAL_FLOOR = ["cement", "ceramic tiles", "vinyl or asphalt strips", "parquet or polished wood", "wood planks"]
ORIGINAL_WALL = ["cement", "bricks", "cement blocks", "stone with lime/cement"]
ORIGINAL_ROOF = ["metal/galvanized sheet", "cement", "ceramic tiles", "roofing shingles"]

# %% [markdown]
# ### 2.5 Review: every label, its score and how many households have it

# %%
review = []
for name, var, scores in [("water", "hv201", WATER), ("toilet", "hv205", TOILET), ("fuel", "hv226", FUEL),
                          ("floor", "hv213", FLOOR), ("wall", "hv214", WALL), ("roof", "hv215", ROOF)]:
    counts = raw[var].value_counts()
    for label, n in counts.items():
        if n > 0:
            review.append({"indicator": name, "label": label, "score": scores.get(label, "NOT LISTED"),
                           "households": n})
review = pd.DataFrame(review)
save_table(review, "classification")
display(review.head(15))

missing_labels = review[review["score"].isna()]
print(f"Finding: {len(missing_labels)} labels are scored missing ({missing_labels['households'].sum()} answers); "
      "every other label is scored 1 or 0.")

# %% [markdown]
# **Interpretation.** The scores follow standard definitions: WHO/UNICEF JMP for water and
# sanitation, WHO for clean cooking fuel, DHS for finished building materials. Only a few labels
# ("other", "no food cooked in house") are set to missing, so the results are driven by the
# definitions, not by missing data.
# *To change a definition* (for example, metal walls as finished), edit the dictionary in Section 2
# and rerun; every table and chart updates.

# %% [markdown]
# ## 3. Household basics
# ### 3.1 Survey design and geography

# %%
hh = pd.DataFrame()
hh["cluster"] = raw["hv021"].astype(int)            # survey cluster (for standard errors)
hh["stratum"] = raw["hv022"].astype(str)            # sampling stratum (14: province × urban/rural)
hh["weight"] = raw["hv005"] / 1_000_000             # DHS weight has 6 implied decimals
hh["urban"] = classify(raw["hv025"], {"urban": 1, "rural": 0})
hh["province"] = raw["hv024"].astype("string")      # "string" keeps missing values missing
hh["eco_zone"] = raw["shecoreg"].astype("string")   # mountain / hill / terai
hh["district"] = raw["shdist"].astype("string")
hh["wealth_dhs"] = raw["hv270"].astype("string")    # DHS wealth quintile (comparison only)
peek(hh)

# %% [markdown]
# ### 3.2 Finding: weighting changes the urban share

# %%
urban_sample = 100 * hh["urban"].mean()
urban_weighted = wpct(hh["urban"], hh["weight"])

fig, ax = plt.subplots(figsize=(6, 1.6))
ax.barh(["Survey sample", "All households (weighted)"], [urban_sample, urban_weighted], color=URBAN_COLOR)
for y, value in enumerate([urban_sample, urban_weighted]):
    ax.text(value + 1, y, f"{value:.1f}%", va="center")
ax.set_xlim(0, 100)
ax.set_xlabel("Urban share of households (%)")
ax.set_title("Fig 1. Weighting changes the urban share")
save_figure(fig, "fig01_urban_share")

print(f"Finding: urban households are {urban_sample:.1f}% of the sample but {urban_weighted:.1f}% "
      "of households once weighted, so all percentages use the weights.")

# %% [markdown]
# **Interpretation.** The DHS sampled rural areas more heavily than their share of households, so
# the raw sample under-represents urban households. Unweighted averages would therefore lean
# towards rural conditions. This is why every percentage and regression in this script uses the
# survey weights. *In the paper:* say that percentages are household-weighted.

# %% [markdown]
# ### 3.3 Household head: education, sex, age
# Education comes from the household roster: find the member whose relationship (`hv101_xx`) is "head".

# %%
head_edu = pd.Series(pd.NA, index=raw.index, dtype="object")
for line in range(1, 100):
    rel_col, edu_col = f"hv101_{line:02d}", f"hv106_{line:02d}"
    if rel_col not in raw.columns:
        break
    is_head = (raw[rel_col] == "head") & head_edu.isna()
    head_edu[is_head] = raw.loc[is_head, edu_col]

EDUCATION = {"no education, preschool/early childhood education": "none", "basic": "basic",
             "secondary": "secondary", "higher": "higher", "don't know": None}
hh["head_education"] = classify(head_edu, EDUCATION)
hh["head_female"] = classify(raw["hv219"], {"female": 1, "male": 0})
hh["head_age"] = pd.to_numeric(raw["hv220"], errors="coerce")
display(hh["head_education"].value_counts(dropna=False))

# %% [markdown]
# ## 4. Indicators
# ### 4.1 Basic services: electricity, improved water, basic sanitation
# Basic sanitation = improved toilet that is **not shared**. Sharing (`hv225`) is only asked of
# households with a toilet, so a missing answer is normal when there is no improved toilet.

# %%
hh["electricity"] = classify(raw["hv206"], {"yes": 1, "no": 0})
hh["water_improved"] = classify(raw["hv201"], WATER)

toilet_improved = classify(raw["hv205"], TOILET)
toilet_shared = classify(raw["hv225"], {"yes": 1, "no": 0})
hh["toilet_basic"] = toilet_improved
hh.loc[toilet_shared == 1, "toilet_basic"] = 0                               # shared: not basic
hh.loc[(toilet_improved == 1) & toilet_shared.isna(), "toilet_basic"] = np.nan  # improved, sharing unknown

display(pd.crosstab(toilet_improved, toilet_shared.isna(), dropna=False,
                    rownames=["improved toilet"], colnames=["sharing answer missing"]))

# %% [markdown]
# **Interpretation.** Every household with a missing sharing answer has no improved toilet (the
# question was not asked), so the missing-answer rule changes nothing in this dataset. It is a
# safeguard: an improved toilet with an unknown sharing status would be missing, not "basic".

# %% [markdown]
# ### 4.2 Cooking and water conditions: clean fuel, water on premises
# `hv204` holds minutes to the water source, or "on premises" / "don't know".

# %%
hh["clean_fuel"] = classify(raw["hv226"], FUEL)

water_time = raw["hv204"].astype(object)
hh["water_on_premises"] = 0.0                                   # any number of minutes
hh.loc[water_time == "on premises", "water_on_premises"] = 1.0
hh.loc[(water_time == "don't know") | water_time.isna(), "water_on_premises"] = np.nan
peek(hh[["clean_fuel", "water_on_premises"]])

# %% [markdown]
# ### 4.3 Housing materials and crowding
# Overcrowded = more than 3 people per sleeping room. Invalid counts (0 or missing) are missing.

# %%
hh["floor_dhs"] = classify(raw["hv213"], FLOOR)
hh["wall_dhs"] = classify(raw["hv214"], WALL)
hh["roof_dhs"] = classify(raw["hv215"], ROOF)

# Original lists: 1 if listed, 0 otherwise, missing where the DHS rules say missing
hh["floor_original"] = raw["hv213"].isin(ORIGINAL_FLOOR).astype(float).where(hh["floor_dhs"].notna())
hh["wall_original"] = raw["hv214"].isin(ORIGINAL_WALL).astype(float).where(hh["wall_dhs"].notna())
hh["roof_original"] = raw["hv215"].isin(ORIGINAL_ROOF).astype(float).where(hh["roof_dhs"].notna())

members = pd.to_numeric(raw["hv009"], errors="coerce")
sleeping_rooms = pd.to_numeric(raw["hv216"], errors="coerce")
valid = (members > 0) & (sleeping_rooms > 0)
hh["not_overcrowded"] = (members / sleeping_rooms <= 3).astype(float).where(valid)
peek(hh[["floor_dhs", "wall_dhs", "roof_dhs", "not_overcrowded"]])

# %% [markdown]
# ### 4.4 Review: missing values

# %%
COMPONENTS = {"electricity": "Electricity", "water_improved": "Improved water source",
              "toilet_basic": "Basic sanitation", "clean_fuel": "Clean cooking fuel",
              "water_on_premises": "Water on premises", "floor_dhs": "Finished floor",
              "wall_dhs": "Finished walls", "roof_dhs": "Finished roof",
              "not_overcrowded": "Not overcrowded"}
missing = hh[list(COMPONENTS)].isna().sum().rename("missing households")
save_table(missing, "missing_values")
display(missing)

# %% [markdown]
# **Interpretation.** Missing values are rare (at most 40 households, about 0.3%, for clean fuel),
# so dropping them does not affect the results.

# %% [markdown]
# ## 5. Indices and profiles
# ### 5.1 Three indices
# - **BSI** Basic Services Index (0–3) = electricity + improved water + basic sanitation
# - **CWI** Cooking and Water Conditions Index (0–2) = clean fuel + water on premises
# - **HMCI** Housing Materials and Crowding Index (0–4) = finished floor + walls + roof + not overcrowded
#
# `min_count` makes an index missing if any part is missing.

# %%
hh["BSI"] = hh[["electricity", "water_improved", "toilet_basic"]].sum(axis=1, min_count=3)
hh["CWI"] = hh[["clean_fuel", "water_on_premises"]].sum(axis=1, min_count=2)
hh["HMCI_dhs"] = hh[["floor_dhs", "wall_dhs", "roof_dhs", "not_overcrowded"]].sum(axis=1, min_count=4)
hh["HMCI_original"] = hh[["floor_original", "wall_original", "roof_original", "not_overcrowded"]].sum(axis=1, min_count=4)
display(hh[["BSI", "CWI", "HMCI_dhs", "HMCI_original"]].describe().round(2))

# %% [markdown]
# **Interpretation.** The Basic Services Index sits close to its maximum of 3 for most households,
# while the Cooking and Water Conditions Index and the Housing Materials and Crowding Index are spread out. The
# three indices measure different things, which is the starting point of the paper.

# %% [markdown]
# ### 5.2 Household profiles
# | Profile | Definition |
# |---|---|
# | Basic Services and Living Conditions Met | BSI = 3, CWI = 2 and HMCI ≥ 3 (at least three of the four housing conditions) |
# | Basic Services Only | BSI = 3 and CWI = 0 |
# | Limited Basic Services | BSI ≤ 1 (at most one basic service) |
# | Intermediate | all other households with complete index information |

# %%
TOP_PROFILE = "Basic Services and Living Conditions Met"


def household_profile(row):
    if pd.isna(row["BSI"]) or pd.isna(row["CWI"]) or pd.isna(row["HMCI_dhs"]):
        return np.nan
    if row["BSI"] == 3 and row["CWI"] == 2 and row["HMCI_dhs"] >= 3:
        return TOP_PROFILE
    if row["BSI"] == 3 and row["CWI"] == 0:
        return "Basic Services Only"
    if row["BSI"] <= 1:
        return "Limited Basic Services"
    return "Intermediate"


PROFILES = ["Limited Basic Services", "Basic Services Only", "Intermediate", TOP_PROFILE]
hh["profile"] = hh.apply(household_profile, axis=1)
display(hh["profile"].value_counts(dropna=False))

# %% [markdown]
# **Interpretation.** Most households are "Intermediate". The informative groups are "Basic Services
# and Living Conditions Met" and "Basic Services Only" (all basic services but neither clean fuel
# nor water on premises). "Limited Basic Services" describes basic services only; it says nothing
# about cooking, water or housing conditions. The cut-offs are a descriptive typology, not a
# standard measure; say so in the paper.

# %% [markdown]
# ### 5.3 Do the components move together?
# If the three indices measure different aspects of living conditions, components *across* indices
# should be only weakly related. This checks that claim, and also how closely the components
# *within* each index go together. (Simple correlations of 0/1 indicators, unweighted.)

# %%
corr = hh[list(COMPONENTS)].corr()
corr.index = corr.columns = [COMPONENTS[c] for c in corr.columns]
save_table(corr.round(3), "component_correlations")
display(corr.round(2))

index_corr = hh[["BSI", "CWI", "HMCI_dhs"]].corr()
save_table(index_corr.round(3), "index_correlations")
display(index_corr.round(2))

# %%
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(6.8, 5.6))
blue_scale = LinearSegmentedColormap.from_list("blues", ["#ffffff"] + BLUES)
image = ax.imshow(corr.clip(lower=0), cmap=blue_scale, vmin=0, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        value = corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if value > 0.5 else "black")
ax.grid(False)
# Outline the three indices (BSI = first 3, CWI = next 2, HMCI = last 4)
for start, size in [(0, 3), (3, 2), (5, 4)]:
    ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), size, size, fill=False, edgecolor="black", linewidth=1.5))
fig.colorbar(image, ax=ax, shrink=0.7, label="Correlation")
ax.set_title("Fig 6b. Correlation between components (boxes = the three indices)")
save_figure(fig, "fig06b_component_correlations")

print(f"Finding: index correlations are {index_corr.loc['BSI', 'CWI']:.2f} (BSI–CWI), {index_corr.loc['BSI', 'HMCI_dhs']:.2f} "
      f"(BSI–HMCI) and {index_corr.loc['CWI', 'HMCI_dhs']:.2f} (CWI–HMCI). The strongest component pairs are finished floor–walls "
      f"({corr.loc['Finished floor', 'Finished walls']:.2f}) and clean fuel–finished floor ({corr.loc['Clean cooking fuel', 'Finished floor']:.2f}).")

# %% [markdown]
# **Interpretation.**
# - **Basic services move separately from the rest.** The Basic Services Index correlates only
#   weakly with the other two indices (about 0.2). Having electricity, improved water and a basic
#   toilet says little about cooking fuel or housing materials: the core argument of the paper.
# - **The indices are summaries, not scales.** Components *within* the Basic Services Index barely
#   correlate with each other (each service is near-universal and is supplied separately), and the
#   two parts of the Cooking and Water Conditions Index correlate only weakly (about 0.17). The
#   indices group conditions by *what they are*, not because they move together; describe them as
#   transparent counts with equal weights, and keep the component results central.
# - **Clean fuel goes with building materials.** Clean cooking fuel correlates more with finished
#   floors and walls (about 0.5) than with water on premises. Both reflect household resources and
#   access to modern markets; this is a pattern to describe, not evidence that they belong in one index.
# - A finished roof and adequate sleeping space are only weakly related to the other housing items
#   (most roofs are already finished, and crowding depends on household size).

# %% [markdown]
# ## 6. Asset measure (wealth without housing items)
# The DHS wealth index (`hv270`) is partly built from housing, water and sanitation, which are our
# outcomes. So we build our own index from durable goods only.
#
# ### 6.1 Asset items: DHS variables and scoring
# Each item is a yes/no question in the DHS household questionnaire. As in Section 2, the labels
# are scored explicitly: "yes" = 1, "no" = 0 (anything else would stop the code).
#
# | DHS variable | DHS question | Item name used here |
# |---|---|---|
# | `hv207` | has radio | radio |
# | `hv208` | has television | television |
# | `hv209` | has refrigerator | refrigerator |
# | `hv210` | has bicycle/rickshaw | bicycle |
# | `hv211` | has motorcycle/scooter | motorcycle |
# | `hv212` | has car/truck/tractor | car/truck/tractor |
# | `hv221` | has telephone (land-line) | landline |
# | `hv243a` | has mobile telephone | mobile phone |
# | `hv243b` | has watch | watch |
# | `hv243e` | has a computer | computer |
# | `hv247` | has bank account | bank account |

# %%
ASSET_SCORE = {"yes": 1, "no": 0}
ASSETS = {"hv207": "radio", "hv208": "television", "hv209": "refrigerator", "hv210": "bicycle",
          "hv211": "motorcycle", "hv212": "car/truck/tractor", "hv221": "landline",
          "hv243a": "mobile phone", "hv243b": "watch", "hv243e": "computer", "hv247": "bank account"}

assets = pd.DataFrame()
for var, name in ASSETS.items():
    assets[name] = classify(raw[var], ASSET_SCORE)

# Review table: one row per item, with its DHS variable, answers and weighted ownership
asset_review = pd.DataFrame({
    "dhs_variable": list(ASSETS),
    "item": list(ASSETS.values()),
    "yes": [int((raw[var] == "yes").sum()) for var in ASSETS],
    "no": [int((raw[var] == "no").sum()) for var in ASSETS],
    "missing": [int(raw[var].isna().sum()) for var in ASSETS],
    "weighted % owning": [wpct(assets[name], hh["weight"]) for name in ASSETS.values()],
}).set_index("item")
save_table(asset_review.round(1), "asset_items")
display(asset_review.round(1))
peek(assets)

# %% [markdown]
# ### 6.2 Principal component index
# Standardize each item, then take the first principal component (the DHS method).
# The weighted version is for robustness only.

# %%
def asset_index(items, weights):
    """First principal component of standardized items. Returns (score, loadings)."""
    w = weights / weights.sum()
    items = items.fillna(items.mean())                     # rare missing answers -> item mean
    mean = (items.mul(w, axis=0)).sum()
    sd = np.sqrt(((items - mean) ** 2).mul(w, axis=0).sum())
    items = items.loc[:, sd > 0]                           # drop items nobody (or everybody) has
    z = (items - mean[items.columns]) / sd[items.columns]
    covariance = z.mul(w, axis=0).T @ z
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    loadings = eigenvectors[:, -1]                         # largest eigenvalue is last
    if loadings.sum() < 0:
        loadings = -loadings                               # more assets = higher score
    return z @ loadings, pd.Series(loadings, index=items.columns)


equal_weights = pd.Series(1.0, index=assets.index)
hh["asset_score"], loadings = asset_index(assets, equal_weights)
hh["asset_score_wpca"], loadings_w = asset_index(assets, hh["weight"])

loading_table = pd.DataFrame({"dhs_variable": asset_review["dhs_variable"],
                              "loading": loadings, "loading (weighted PCA)": loadings_w}).sort_values("loading")
save_table(loading_table.round(3), "asset_loadings")
display(loading_table.round(3))

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.barh(loading_table.index, loading_table["loading"], color=URBAN_COLOR)
ax.scatter(loading_table["loading (weighted PCA)"], loading_table.index, color="black", s=12,
           zorder=3, label="weighted PCA")
ax.set_xlabel("Loading on first principal component")
ax.legend(loc="lower right")
ax.set_title("Fig 2. Asset index: weight of each item")
save_figure(fig, "fig02_asset_loadings")

print("Finding: the index is driven most by", ", ".join(loading_table.index[-3:][::-1]),
      "and is almost identical with survey weights.")

# %% [markdown]
# **Interpretation.** Items that distinguish better-off households (refrigerator, motorcycle,
# television, computer) carry the most weight; items that many households already own (radio,
# bicycle) carry little. This is the expected pattern for a wealth index. Weighted and unweighted
# PCA give almost the same loadings, so that choice does not matter.

# %% [markdown]
# ### 6.3 Asset groups (like quintiles, but households with the same score stay together)

# %%
def asset_groups(score, weights, n=5):
    """Cumulative weighted share at each distinct score -> group 1..n (ties share a group)."""
    share_up_to = weights.groupby(score).sum().sort_index().cumsum() / weights.sum()
    group_of_score = np.ceil(share_up_to * n - 1e-9).clip(1, n).astype(int)
    return score.map(group_of_score)


hh["asset_group"] = asset_groups(hh["asset_score"], hh["weight"])
hh["asset_group_wpca"] = asset_groups(hh["asset_score_wpca"], hh["weight"])

group_share = 100 * hh.groupby("asset_group")["weight"].sum() / hh["weight"].sum()
save_table(group_share.round(1).rename("weighted % of households"), "asset_group_shares")

fig, ax = plt.subplots(figsize=(5.5, 2.5))
ax.bar(group_share.index, group_share, color=URBAN_COLOR)
ax.axhline(20, color=GREY, linestyle="--", linewidth=1)
for g, value in group_share.items():
    ax.text(g, value + 0.5, f"{value:.1f}%", ha="center")
ax.set_xlabel("Asset group (1 = fewest assets)")
ax.set_ylabel("% of households")
ax.set_title("Fig 3. Asset group sizes (dashed line = exact quintile)")
save_figure(fig, "fig03_asset_groups")

print(f"Finding: groups hold {group_share.min():.1f}%–{group_share.max():.1f}% of households, close to quintiles.")

# %% [markdown]
# **Interpretation.** Many households own few or none of these items and therefore share the same
# score. Keeping tied households together makes the lowest group smaller (about 15%) and the second
# larger (about 24%) than exact quintiles. *In the paper:* call them "asset groups".

# %% [markdown]
# ## 7. Descriptive results
# ### 7.1 Table 1: each condition among urban and rural households

# %%
urban_hh = hh[hh["urban"] == 1]
rural_hh = hh[hh["urban"] == 0]

table1 = pd.DataFrame(index=list(COMPONENTS) + ["floor_original", "wall_original", "roof_original"])
table1["all"] = [wpct(hh[c], hh["weight"]) for c in table1.index]
table1["urban"] = [wpct(urban_hh[c], urban_hh["weight"]) for c in table1.index]
table1["rural"] = [wpct(rural_hh[c], rural_hh["weight"]) for c in table1.index]
table1["gap"] = table1["urban"] - table1["rural"]
save_table(table1.round(3), "table1_components")
display(table1.round(1))

# %%
t = table1.loc[list(COMPONENTS)].sort_values("gap")
labels = [COMPONENTS[c] for c in t.index]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hlines(labels, t["rural"], t["urban"], color="#d0cfca", linewidth=3)
ax.scatter(t["rural"], labels, color=RURAL_COLOR, s=50, label="Rural", zorder=3)
ax.scatter(t["urban"], labels, color=URBAN_COLOR, s=50, label="Urban", zorder=3)
for label, (_, row) in zip(labels, t.iterrows()):
    ax.text(max(row["urban"], row["rural"]) + 2, label, f"+{row['gap']:.0f}" if row["gap"] >= 0 else f"{row['gap']:.0f}",
            va="center", color=GREY)
ax.set_xlim(0, 110)
ax.set_xlabel("% of households (weighted)")
ax.legend(loc="lower left")
ax.set_title("Fig 4. Each condition among urban and rural households (gap in points)")
save_figure(fig, "fig04_components_urban_rural")

top3 = t["gap"].sort_values(ascending=False).head(3)
print("Finding: the largest urban advantages are",
      ", ".join(f"{COMPONENTS[c]} ({g:.1f} pts)" for c, g in top3.items()),
      f"; electricity differs by only {table1.loc['electricity', 'gap']:.1f} pts.")

# %% [markdown]
# **Interpretation.** Basic services are close to universal in both areas (gaps of 2–4 points).
# Basic sanitation is slightly *more* common in rural areas because urban households share toilets
# more often (about 27% vs 22% of households with a toilet), and shared toilets do not count as
# basic. The large urban advantages are in clean cooking fuel and building materials: this is where
# urban–rural inequality remains.
# *What this does not show:* these are raw differences. Section 9 shows how much remains once
# province, zone, assets and household head are taken into account.

# %% [markdown]
# ### 7.2 How each index is distributed, urban vs rural

# %%
def weighted_distribution(data, column, by):
    """Weighted % of households at each value of `column`, within each group of `by`."""
    data = data.dropna(subset=[column])
    totals = data.groupby([by, column])["weight"].sum().unstack(fill_value=0)
    return 100 * totals.div(totals.sum(axis=1), axis=0)


INDICES = {"BSI": "Basic Services Index (0–3)", "CWI": "Cooking and Water Conditions Index (0–2)",
           "HMCI_dhs": "Housing Materials and Crowding Index (0–4)"}
dist = {}
for index in list(INDICES) + ["HMCI_original"]:
    dist[index] = weighted_distribution(hh, index, "urban").rename(index={0: "Rural", 1: "Urban"})
    save_table(dist[index].round(2), f"dist_{index}")
    print(index)
    display(dist[index].round(1))

# %%
fig, axes = plt.subplots(3, 1, figsize=(7, 5))
for ax, (index, title) in zip(axes, INDICES.items()):
    d = dist[index]
    colors = BLUES[-len(d.columns):]
    d.plot.barh(stacked=True, ax=ax, color=colors, width=0.6, edgecolor="white", legend=False)
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylabel("")
    ax.legend([int(c) for c in d.columns], title="score", ncol=len(d.columns),
              loc="center left", bbox_to_anchor=(1, 0.5))
axes[-1].set_xlabel("% of households (weighted); darker = higher score")
fig.suptitle("Fig 5. Index scores, urban vs rural", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig05_index_distributions")

print(f"Finding: {dist['BSI'].loc['Urban', 3]:.0f}% of urban and {dist['BSI'].loc['Rural', 3]:.0f}% of rural households "
      f"have all three basic services, but {dist['HMCI_dhs'].loc['Urban', 4]:.0f}% vs {dist['HMCI_dhs'].loc['Rural', 4]:.0f}% "
      "have the top housing materials and crowding score.")

# %% [markdown]
# **Interpretation.** Urban and rural households look almost the same on the Basic Services Index
# but very different on the other two: more than half of urban households have the top housing
# score, compared with about a quarter of rural households. The indices separate *access to
# services* from *how households live*.

# %% [markdown]
# ### 7.3 Household profiles by residence and by asset group

# %%
profile_by_residence = weighted_distribution(hh, "profile", "urban").rename(index={0: "Rural", 1: "Urban"})[PROFILES]
profile_by_assets = weighted_distribution(hh, "profile", "asset_group")[PROFILES]
save_table(profile_by_residence.round(2), "profiles_by_residence")
save_table(profile_by_assets.round(2), "profiles_by_asset_group")
display(profile_by_residence.round(1))
display(profile_by_assets.round(1))

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1, 1.4]})
profile_by_residence.plot.barh(stacked=True, ax=axes[0], color=BLUES[1:], width=0.6, edgecolor="white", legend=False)
profile_by_assets.plot.barh(stacked=True, ax=axes[1], color=BLUES[1:], width=0.6, edgecolor="white")
axes[0].set_title("By residence")
axes[1].set_title("By asset group (1 = fewest assets)")
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of households")
    ax.set_ylabel("")
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
fig.suptitle("Fig 6. Household profiles", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig06_profiles")

print(f"Finding: '{TOP_PROFILE}' households rise from {profile_by_assets.loc[1, TOP_PROFILE]:.1f}% in the lowest "
      f"asset group to {profile_by_assets.loc[5, TOP_PROFILE]:.1f}% in the highest.")

# %% [markdown]
# **Interpretation.** The share of households meeting basic services and living conditions rises
# steeply with assets, as expected. "Basic Services Only" households are more common in rural areas
# and in the lower asset groups: they have every basic service but cook with traditional fuels and
# fetch water. They illustrate that access alone does not mean better living conditions. These are
# descriptive shares, without controls.

# %% [markdown]
# ### 7.3b Do deprivations cluster more than the separate rates imply?
# If conditions were unrelated within a group, the share of households lacking several of them
# would equal the product of the separate rates. We compare the observed share with that
# "independence" benchmark, within each residence and asset group:
# - lacking **all three** of clean fuel, finished floor and finished walls (the largest gaps);
# - lacking **four or more** of the nine conditions (benchmark: each condition lacked independently
#   at its group rate, computed exactly).
# A ratio above 1 means deprivations occur together more often than the separate rates suggest.

# %%
complete = hh.dropna(subset=list(COMPONENTS)).copy()
lacks = 1 - complete[list(COMPONENTS)]
CORE = ["clean_fuel", "floor_dhs", "wall_dhs"]


def share_lacking_k_or_more(rates, k):
    """P(at least k of the conditions lacked) if each is lacked independently with its own rate."""
    dist = np.array([1.0])
    for r in rates:
        dist = np.convolve(dist, [1 - r, r])
    return dist[k:].sum()


rows = []
groups = [("Rural", complete["urban"] == 0), ("Urban", complete["urban"] == 1)] + \
         [(f"Asset group {g}", complete["asset_group"] == g) for g in range(1, 6)]
for name, mask in groups:
    w = complete.loc[mask, "weight"]
    rate = lacks[mask].mul(w, axis=0).sum() / w.sum()
    core_obs = (lacks.loc[mask, CORE].sum(axis=1) == 3).mul(w).sum() / w.sum()
    many_obs = (lacks[mask].sum(axis=1) >= 4).mul(w).sum() / w.sum()
    rows.append({"group": name,
                 "lack fuel+floor+walls, observed %": 100 * core_obs,
                 "lack fuel+floor+walls, if independent %": 100 * rate[CORE].prod(),
                 "lack 4+ of 9, observed %": 100 * many_obs,
                 "lack 4+ of 9, if independent %": 100 * share_lacking_k_or_more(rate.values, 4)})
cluster = pd.DataFrame(rows).set_index("group")

# Uncertainty: resample PSUs (clusters) with replacement, 300 times, and recompute
# observed minus independence benchmark (percentage points) for each group.
rng = np.random.default_rng(2022)
psus = complete["cluster"].unique()
by_psu = {c: idx for c, idx in complete.groupby("cluster").indices.items()}
boot = {name: {"core": [], "many": []} for name, _ in groups}
for _ in range(300):
    take = np.concatenate([by_psu[c] for c in rng.choice(psus, size=len(psus), replace=True)])
    b, lb = complete.iloc[take], lacks.iloc[take]
    for name, mask_fn in [(n, None) for n, _ in groups]:
        if name == "Rural": m = (b["urban"] == 0).values
        elif name == "Urban": m = (b["urban"] == 1).values
        else: m = (b["asset_group"] == int(name[-1])).values
        w = b.loc[m, "weight"]
        rate = lb[m].mul(w.values, axis=0).sum() / w.sum()
        core = ((lb[m][CORE].sum(axis=1) == 3) * w.values).sum() / w.sum() - rate[CORE].prod()
        many = ((lb[m].sum(axis=1) >= 4) * w.values).sum() / w.sum() - share_lacking_k_or_more(rate.values, 4)
        boot[name]["core"].append(100 * core)
        boot[name]["many"].append(100 * many)
for name in cluster.index:
    for key, label in [("core", "fuel+floor+walls"), ("many", "4+ of 9")]:
        cluster.loc[name, f"excess {label}, 95% low"] = np.percentile(boot[name][key], 2.5)
        cluster.loc[name, f"excess {label}, 95% high"] = np.percentile(boot[name][key], 97.5)
cluster["ratio fuel+floor+walls"] = cluster.iloc[:, 0] / cluster.iloc[:, 1]
cluster["ratio 4+ of 9"] = cluster.iloc[:, 2] / cluster.iloc[:, 3]
save_table(cluster.round(2), "deprivation_clustering")
display(cluster.round(1))

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, (obs, exp, title) in zip(axes, [(0, 1, "Lack clean fuel, floor and walls"),
                                         (2, 3, "Lack 4 or more of the 9 conditions")]):
    y = np.arange(len(cluster))
    ax.barh(y - 0.2, cluster.iloc[:, obs], height=0.4, color=URBAN_COLOR, label="Observed")
    ax.barh(y + 0.2, cluster.iloc[:, exp], height=0.4, color="#c9c8c3", label="If conditions were independent")
    ax.set_yticks(y, cluster.index)
    ax.set_xlabel("% of households")
    ax.set_title(title, fontsize=10)
axes[0].invert_yaxis()                      # shared axis: invert once, Rural on top
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.suptitle("Fig 6c. Do deprivations occur together more than the separate rates imply?",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig06c_deprivation_clustering")

print("Finding: observed/independent ratio, fuel+floor+walls —",
      "; ".join(f"{g} {r:.2f}" for g, r in cluster["ratio fuel+floor+walls"].items()))
print("Finding: observed/independent ratio, 4+ of 9 —",
      "; ".join(f"{g} {r:.2f}" for g, r in cluster["ratio 4+ of 9"].items()))

# %% [markdown]
# **Interpretation.** Deprivations occur together far more often than the separate rates imply.
# - **Urban:** 23.5% of urban households lack clean fuel, a finished floor and finished walls
#   together; if these were unrelated, only 4.7% would. Urban disadvantage is concentrated in a
#   minority of households rather than spread thinly. 16.6% lack four or more of the nine conditions
#   (5.4% if independent).
# - **Rural:** 50.3% lack all three (31.9% if independent); 36.2% lack four or more (29.0%).
# - **Within asset groups:** the share lacking all three exceeds the benchmark in every group, and
#   the cluster-bootstrap 95% interval for the difference excludes zero in each. For "4+ of 9" the
#   lowest asset group's interval includes zero (most households there lack these conditions anyway).
# - Ratios are large where the separate rates are small (asset group 5), so read the percentages,
#   not the ratios. Intervals come from 300 cluster (PSU) bootstrap resamples.
#
# This is what the joint analysis adds beyond the separate rates: the marginal rates would suggest
# that very few urban households face all three deprivations, when about one in four does.

# %% [markdown]
# ### 7.4 Check against the DHS 2022 report
# The report gives percentages of the **de jure household population** (usual residents), so here
# each household's weight is multiplied by its number of usual residents (`hv012`). The rest of the
# analysis describes **households**.

# %%
usual_residents = pd.to_numeric(raw["hv012"], errors="coerce").fillna(0)   # 4 households have none
hh["people_weight"] = hh["weight"] * usual_residents

check = pd.DataFrame(index=["electricity", "clean_fuel", "toilet_basic"])
check["households, urban"] = [wpct(urban_hh[c], urban_hh["weight"]) for c in check.index]
check["people, urban"] = [wpct(hh.loc[hh.urban == 1, c], hh.loc[hh.urban == 1, "people_weight"]) for c in check.index]
check["people, rural"] = [wpct(hh.loc[hh.urban == 0, c], hh.loc[hh.urban == 0, "people_weight"]) for c in check.index]
check["people, all"] = [wpct(hh[c], hh["people_weight"]) for c in check.index]
save_table(check.round(2), "check_vs_dhs_report")
display(check.round(1))

print(f"Finding: electricity reaches {check.loc['electricity', 'people, urban']:.1f}% of the urban and "
      f"{check.loc['electricity', 'people, rural']:.1f}% of the rural household population (report: 98 / 94). "
      f"Basic sanitation is {check.loc['toilet_basic', 'people, all']:.2f}% (report Table 16.7: 72.6).")

# Where the small sanitation difference comes from: the report appears to count
# "flush, don't know where" toilets (11 households) as unimproved.
toilet_report_rule = dict(TOILET)
toilet_report_rule["flush, don't know where"] = 0
toilet_alt = classify(raw["hv205"], toilet_report_rule)
toilet_alt[toilet_shared == 1] = 0
toilet_alt[(toilet_alt == 1) & toilet_shared.isna()] = np.nan
print(f"With 'flush, don't know where' as unimproved: {wpct(toilet_alt, hh['people_weight']):.2f}% overall, "
      f"{wpct(toilet_alt[hh.urban == 1], hh.loc[hh.urban == 1, 'people_weight']):.2f}% urban, "
      f"{wpct(toilet_alt[hh.urban == 0], hh.loc[hh.urban == 0, 'people_weight']):.2f}% rural.")

# %% [markdown]
# **Interpretation.** Population-weighted figures reproduce the official DHS report: electricity
# exactly, and basic sanitation to within 0.1 point. The remaining sanitation difference comes from
# 11 households with "flush, don't know where" toilets: counting them as unimproved reproduces the
# report's 72.6% overall, urban and rural. This analysis keeps them as improved (DHS/JMP guidance);
# the effect is about 0.05 points. Household and population figures differ because larger
# households use clean fuel less often.
# *In the paper:* the analysis reports percentages of **households**; label the report's figures as
# percentages of the **household population**, and mention the "flush, don't know where" difference
# in a footnote.

# %% [markdown]
# ## 8. Regressions: the urban gap
# ### 8.1 Five specifications
# All use province and ecological zone. Specifications 2–4 differ **only** in the wealth measure.
# `C(...)` makes dummies; `Treatment('...')` names the reference category.

# %%
GEO = "C(province, Treatment('koshi province')) + C(eco_zone, Treatment('terai'))"
HEAD = "C(head_education, Treatment('none')) + head_female + head_age"
SPECS = {
    "(1) Province + zone": GEO,
    "(2) + DHS wealth index": GEO + " + C(wealth_dhs, Treatment('poorest'))",
    "(3) + asset groups": GEO + " + C(asset_group)",
    "(4) + asset score": GEO + " + asset_score",
    "(5) + asset groups + head": GEO + " + C(asset_group) + " + HEAD,
}
FULL = "(5) + asset groups + head"

# Variables used by any specification: households missing any of them are dropped,
# so all five specifications use exactly the same households.
MODEL_VARS = ["urban", "province", "eco_zone", "wealth_dhs", "asset_group", "asset_score",
              "head_education", "head_female", "head_age", "weight", "cluster"]


def urban_gap(outcome, controls, data, weighted=True, cluster="cluster"):
    """Regress outcome on urban + controls. Returns the fitted model."""
    formula = f"{outcome} ~ urban + {controls}"
    if weighted:
        model = smf.wls(formula, data=data, weights=data["weight"])
    else:
        model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})


# %% [markdown]
# ### 8.2 Table 2: urban coefficient by specification

# %%
rows = []
for outcome in INDICES:
    sample = hh.dropna(subset=MODEL_VARS + [outcome])
    for spec, controls in SPECS.items():
        fit = urban_gap(outcome, controls, sample)
        rows.append({"outcome": outcome, "specification": spec, "N": int(fit.nobs),
                     "urban_coef": fit.params["urban"], "se": fit.bse["urban"], "p": fit.pvalues["urban"]})
table2 = pd.DataFrame(rows)
save_table(table2.round(4), "table2_urban_gap")
display(table2.round(3))

# Full regression output for the main model (housing materials and crowding, specification 5)
sample = hh.dropna(subset=MODEL_VARS + ["HMCI_dhs"])
print(urban_gap("HMCI_dhs", SPECS[FULL], sample).summary().tables[1])

# %%
SHORT_NAMES = {"BSI": "Basic services", "CWI": "Cooking & water", "HMCI_dhs": "Housing materials & crowding"}
fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for ax, (outcome, title) in zip(axes, SHORT_NAMES.items()):
    t = table2[table2["outcome"] == outcome].iloc[::-1]
    colors = [GREY if "DHS" in s else URBAN_COLOR for s in t["specification"]]
    ax.errorbar(t["urban_coef"], t["specification"], xerr=1.96 * t["se"], fmt="none", ecolor=colors, elinewidth=2)
    ax.scatter(t["urban_coef"], t["specification"], color=colors, s=40, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Urban coefficient (index points)")
fig.suptitle("Fig 7. Urban gap by specification (grey = DHS wealth index; bars = 95% CI)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig07_urban_gap_by_specification")

hmci = table2[table2["outcome"] == "HMCI_dhs"].set_index("specification")["urban_coef"]
print(f"Finding: the urban coefficient on housing materials and crowding is {hmci.iloc[0]:.2f} with geography only, "
      f"{hmci.iloc[1]:.2f} with the DHS wealth index, but {hmci.iloc[2]:.2f} with asset groups "
      f"({hmci.iloc[3]:.2f} with the asset score).")

# %% [markdown]
# **Interpretation.**
# - **(1)** is the urban gap among households in the same province and ecological zone.
# - **(2)** Adding the DHS wealth index shrinks the cooking/water and housing gaps almost to zero.
#   But that index is partly built from the same housing, water and sanitation items, so it absorbs
#   part of the outcome itself; it is not a suitable control here.
# - **(3)–(4)** A wealth measure built only from durable goods leaves roughly two-thirds of the gap.
#   Groups and the continuous score give almost the same answer.
# - **(5)** Adding the household head's education, sex and age changes little.
# - For basic services the gap is small and disappears once assets are included.
#
# *What this does not show:* these are associations, not the causal effect of living in a town. The
# asset measure is an imperfect measure of income, so the remaining gap is the gap *after
# adjustment for observed assets*.

# %% [markdown]
# ### 8.3 Matched-index diagnostic: does adding outcome items to the index shrink the gap?
# The DHS index differs from ours in many ways (items, method, coverage). To isolate the overlap,
# we keep everything else fixed — same PCA method, same 11 assets, same households, same
# regression (specification 3: geography + 5 groups) — and only **add outcome items** to the index:
# - **A** 11 assets (our index)
# - **B** A + housing materials (finished floor, walls, roof)
# - **C** A + services and cooking (electricity, improved water, basic sanitation, clean fuel,
#   water on premises)
# - **D** A + all nine outcome indicators (B + C + not overcrowded)
#
# This is a diagnostic of how sensitive the estimate is to what goes into the index, not a test of
# which index measures "true" resources.

# %%
MATERIAL_ITEMS = ["floor_dhs", "wall_dhs", "roof_dhs"]
SERVICE_ITEMS = ["electricity", "water_improved", "toilet_basic", "clean_fuel", "water_on_premises"]
INDEX_VERSIONS = {
    "A: 11 assets": [],
    "B: + housing materials": MATERIAL_ITEMS,
    "C: + services and cooking": SERVICE_ITEMS,
    "D: + all nine outcome items": MATERIAL_ITEMS + SERVICE_ITEMS + ["not_overcrowded"],
}
for version, extra in INDEX_VERSIONS.items():
    items = pd.concat([assets, hh[extra]], axis=1)
    score, _ = asset_index(items, equal_weights)
    hh["group_" + version[0]] = asset_groups(score, hh["weight"])

rows = []
for outcome in INDICES:
    sample = hh.dropna(subset=MODEL_VARS + [outcome])
    for version in INDEX_VERSIONS:
        fit = urban_gap(outcome, GEO + f" + C(group_{version[0]})", sample)
        rows.append({"outcome": outcome, "index": version, "N": int(fit.nobs), "urban_coef": fit.params["urban"],
                     "se": fit.bse["urban"], "p": fit.pvalues["urban"]})
    dhs = table2[(table2.outcome == outcome) & (table2.specification == "(2) + DHS wealth index")].iloc[0]
    rows.append({"outcome": outcome, "index": "DHS wealth index (reference)", "N": int(dhs.N),
                 "urban_coef": dhs.urban_coef, "se": dhs.se, "p": dhs.p})
matched = pd.DataFrame(rows)
save_table(matched.round(4), "matched_index_diagnostic")
display(matched.pivot(index="index", columns="outcome", values="urban_coef").round(3))

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for ax, (outcome, title) in zip(axes, SHORT_NAMES.items()):
    t = matched[matched["outcome"] == outcome].iloc[::-1]
    colors = [GREY if "DHS" in s else URBAN_COLOR for s in t["index"]]
    ax.errorbar(t["urban_coef"], t["index"], xerr=1.96 * t["se"], fmt="none", ecolor=colors, elinewidth=2)
    ax.scatter(t["urban_coef"], t["index"], color=colors, s=40, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Urban coefficient (index points)")
fig.suptitle("Fig 7b. Urban gap when outcome items are added to the asset index (95% CI)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig07b_matched_index")

m = matched.pivot(index="index", columns="outcome", values="urban_coef")
print(f"Finding: HMCI urban coefficient {m.loc['A: 11 assets', 'HMCI_dhs']:.3f} (A), "
      f"{m.loc['B: + housing materials', 'HMCI_dhs']:.3f} (B), {m.loc['C: + services and cooking', 'HMCI_dhs']:.3f} (C), "
      f"{m.loc['D: + all nine outcome items', 'HMCI_dhs']:.3f} (D); DHS index {m.loc['DHS wealth index (reference)', 'HMCI_dhs']:.3f}.")

# %% [markdown]
# **Interpretation.** Holding the method, the 11 assets, the households and the regression fixed,
# adding the outcome items to the index is enough to reproduce most of the DHS-index result.
# - **Housing materials and crowding:** 0.357 with assets only, 0.122 when floor, wall and roof are
#   added, 0.071 when all nine outcome items are added, close to the DHS index (0.080). Adding the
#   service items alone moves it much less (0.279): the drop comes mainly from the materials items
#   that are part of this outcome.
# - **Cooking and water:** 0.245 → 0.112 with all items (DHS index 0.081).
# - **Basic services:** turns slightly negative as items are added.
#
# So the overlap with the outcome, not the other differences between the indices, accounts for most
# of the gap between 0.357 and 0.080. This is a diagnostic of sensitivity to index content; it does
# not show that either index measures economic resources correctly.

# %% [markdown]
# ### 8.4 Overlap and a more flexible asset adjustment
# Do urban and rural households actually overlap in assets? And does the adjusted gap depend on
# modelling assets as 5 groups with the same effect in towns and villages?
# - Distribution of asset deciles by residence (weighted %, and unweighted counts).
# - Households by province × zone × residence (unweighted counts).
# - Flexible model: 10 asset groups, each allowed its own urban gap (`urban * C(asset_decile)`),
#   plus geography and household head. We then average the urban gap over the **rural**
#   households' asset distribution: the gap for households like rural households.

# %%
hh["asset_decile"] = asset_groups(hh["asset_score"], hh["weight"], n=10)
w_dist = (hh.groupby(["asset_decile", "urban"])["weight"].sum().unstack()
          / hh.groupby("urban")["weight"].sum() * 100).rename(columns={0: "rural %", 1: "urban %"})
counts = hh.groupby(["asset_decile", "urban"]).size().unstack().rename(columns={0: "rural n", 1: "urban n"})
overlap = pd.concat([w_dist, counts], axis=1)
save_table(overlap.round(2), "asset_overlap_by_residence")
display(overlap.round(1))

cells = hh.groupby(["province", "eco_zone", "urban"]).size().unstack(fill_value=0).rename(columns={0: "rural n", 1: "urban n"})
save_table(cells, "sample_by_province_zone")
display(cells)

fig, ax = plt.subplots(figsize=(6.5, 3))
x = np.arange(len(overlap))
ax.bar(x - 0.2, overlap["rural %"], width=0.4, color=RURAL_COLOR, label="Rural")
ax.bar(x + 0.2, overlap["urban %"], width=0.4, color=URBAN_COLOR, label="Urban")
ax.set_xticks(x, overlap.index)
ax.set_xlabel("Asset group of 10 (1 = fewest assets; ties kept together, so sizes differ)")
ax.set_ylabel("% of households")
ax.legend()
ax.set_title("Fig 7c. Asset distribution by residence (weighted)")
save_figure(fig, "fig07c_asset_overlap")

# %%
FLEX = GEO + " + " + HEAD


def gap_on_rural_assets(outcome, sample):
    """Urban gap averaged over rural households' asset deciles, from urban * C(asset_decile)."""
    fit = smf.wls(f"{outcome} ~ urban * C(asset_decile) + {FLEX}", data=sample, weights=sample["weight"]).fit(
        cov_type="cluster", cov_kwds={"groups": sample["cluster"]})
    rural = sample[sample["urban"] == 0]
    share = rural.groupby("asset_decile")["weight"].sum() / rural["weight"].sum()
    r = pd.Series(0.0, index=fit.params.index)
    r["urban"] = 1
    for decile, s in share.items():
        name = f"urban:C(asset_decile)[T.{decile}]"
        if name in r.index:
            r[name] = s
    test = fit.t_test(r.values)
    return float(test.effect[0]), float(test.sd[0][0]), float(test.pvalue)


rows = []
FLEX_OUTCOMES = {"BSI": 1, "CWI": 1, "HMCI_dhs": 1, "clean_fuel": 100, "floor_dhs": 100, "wall_dhs": 100}
for outcome, scale in FLEX_OUTCOMES.items():
    sample = hh.dropna(subset=MODEL_VARS + ["asset_decile", outcome])
    main = urban_gap(outcome, SPECS[FULL], sample)
    gap, se, p = gap_on_rural_assets(outcome, sample)
    rows.append({"outcome": outcome, "N": len(sample), "main_spec5": scale * main.params["urban"],
                 "main_se": scale * main.bse["urban"], "flexible_rural_assets": scale * gap,
                 "flexible_se": scale * se, "flexible_p": p})
flexible = pd.DataFrame(rows)
save_table(flexible.round(4), "flexible_asset_adjustment")
display(flexible.round(3))

print("Finding: smallest decile cell —", int(counts.min().min()), "households; flexible vs main gap:",
      "; ".join(f"{r.outcome} {r.flexible_rural_assets:.3f} vs {r.main_spec5:.3f}" for r in flexible.itertuples()))

# %% [markdown]
# **Interpretation.**
# - **Overlap:** urban households are concentrated in the upper asset groups and rural households in
#   the lower ones, but every one of the 10 groups contains at least 155 rural and 361 urban
#   households. Urban and rural asset distributions overlap across the whole range, so no
#   common-support trimming is needed. Geographic cells are thinner: Gandaki Mountain has no urban
#   households and several Mountain cells have under 100 urban households (geography enters the
#   model additively, so these cells still contribute).
# - **Flexible model:** letting each of 10 asset groups have its own urban gap, and averaging over
#   rural households' assets, gives almost the same gaps as specification 5: clean fuel 16.8 vs 17.8
#   points, floors 14.7 vs 15.1, walls 12.9 vs 13.3; HMCI 0.320 vs 0.324; CWI 0.209 vs 0.220.
#   The "about half remains" result does not depend on the 5-group, same-slope specification.

# %% [markdown]
# ## 9. Where the gaps are
# ### 9.1 Each condition, before and after adjustment (specification 5)
# With a 0/1 outcome the coefficient × 100 is in percentage points.

# %%
rows = []
for col, label in COMPONENTS.items():
    sample = hh.dropna(subset=MODEL_VARS + [col])
    fit = urban_gap(col, SPECS[FULL], sample)
    raw_same_sample = urban_gap(col, "1", sample)          # no controls, same households and weights
    rows.append({"condition": label, "raw_gap": table1.loc[col, "gap"],
                 "raw_gap_regression_sample": 100 * raw_same_sample.params["urban"],
                 "adjusted_gap": 100 * fit.params["urban"], "se": 100 * fit.bse["urban"], "p": fit.pvalues["urban"]})
gaps = pd.DataFrame(rows).sort_values("raw_gap")
save_table(gaps.round(3), "gaps_by_component")
display(gaps.round(2))

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.hlines(gaps["condition"], gaps["adjusted_gap"], gaps["raw_gap"], color="#d0cfca", linewidth=3)
ax.scatter(gaps["raw_gap"], gaps["condition"], facecolors="white", edgecolors=GREY, s=45, label="Before adjustment", zorder=3)
ax.errorbar(gaps["adjusted_gap"], gaps["condition"], xerr=1.96 * gaps["se"], fmt="o", color=URBAN_COLOR,
            label="After adjustment (95% CI)", zorder=4)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Urban minus rural (percentage points)")
ax.legend(loc="lower right")
ax.set_title("Fig 8. Urban gap in each condition, before and after adjustment")
save_figure(fig, "fig08_component_gaps")

g = gaps.set_index("condition")["adjusted_gap"]
print(f"Finding: after adjustment the gaps are {g['Clean cooking fuel']:.1f} pts (clean fuel), "
      f"{g['Finished floor']:.1f} (floor) and {g['Finished walls']:.1f} (walls), about half the raw gaps; "
      f"basic sanitation is {g['Basic sanitation']:.1f} (rural ahead).")

# %% [markdown]
# **Interpretation.** Adjustment roughly halves the large gaps but does not remove them. Comparing
# urban and rural households in the same province, zone and asset group, with similar household
# heads, urban households remain about 18, 15 and 13 points more likely to have clean fuel,
# finished floors and finished walls. The smaller gaps also shrink: electricity (about 2 points)
# and water on premises (about 4) remain statistically significant, while improved water and
# overcrowding are not distinguishable from zero. The rural advantage in basic sanitation grows to
# about 4 points and is significant.
# *Possible explanations (not tested here):* fuel supply and prices, construction markets, or
# income not captured by assets. The data cannot separate these.

# %% [markdown]
# ### 9.2 Gaps within each province and ecological zone (no controls)
# Within each region we regress the condition on `urban` alone (household weights, standard errors
# clustered by PSU). The coefficient x 100 equals the weighted urban-rural difference in that region;
# the regression adds a cluster-robust 95% confidence interval.

# %%
REGION_CONDITIONS = {"clean_fuel": "Clean cooking fuel", "floor_dhs": "Finished floor", "wall_dhs": "Finished walls"}


def gap_within(group_col):
    rows = []
    for name, d in hh.groupby(group_col):
        for col, label in REGION_CONDITIONS.items():
            sample = d.dropna(subset=[col])
            fit = urban_gap(col, "1", sample)
            gap, se = 100 * fit.params["urban"], 100 * fit.bse["urban"]
            rows.append({group_col: name.replace(" province", "").title(), "condition": label,
                         "gap": gap, "se": se, "low": gap - 1.96 * se, "high": gap + 1.96 * se,
                         "p": fit.pvalues["urban"], "households": len(sample),
                         "urban_clusters": sample.loc[sample["urban"] == 1, "cluster"].nunique(),
                         "rural_clusters": sample.loc[sample["urban"] == 0, "cluster"].nunique()})
    return pd.DataFrame(rows)


province_ci = gap_within("province")
zone_ci = gap_within("eco_zone")
save_table(province_ci.round(3), "gaps_by_province")
save_table(zone_ci.round(3), "gaps_by_zone")
display(province_ci.round(1))
display(zone_ci.round(1))

# Wide tables of point estimates, sorted by the clean-fuel gap (used for the chart order and findings)
province_gaps = province_ci.pivot(index="province", columns="condition", values="gap")
province_gaps = province_gaps[list(REGION_CONDITIONS.values())].sort_values("Clean cooking fuel")
zone_gaps = zone_ci.pivot(index="eco_zone", columns="condition", values="gap")
zone_gaps = zone_gaps[list(REGION_CONDITIONS.values())].sort_values("Clean cooking fuel")

# %%
# One panel per condition (columns); provinces on top, zones below. Dot = gap, line = 95% CI.
fig, axes = plt.subplots(2, 3, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [7, 3]})
panels = [(province_gaps, province_ci, "province"), (zone_gaps, zone_ci, "eco_zone")]
for row, (order, ci, group_col) in enumerate(panels):
    for col, condition in enumerate(order.columns):
        ax = axes[row, col]
        d = ci[ci["condition"] == condition].set_index(group_col).loc[order.index]
        y = np.arange(len(d))
        ax.errorbar(d["gap"], y, xerr=1.96 * d["se"], fmt="o", color=URBAN_COLOR, elinewidth=1.5, markersize=5)
        for yi, value in zip(y, d["gap"]):
            ax.annotate(f"{value:.0f}", (value, yi), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#444444")
        ax.set_yticks(y, d.index if col == 0 else [""] * len(d))
        ax.set_ylim(-0.6, len(d) - 0.2)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
for col, condition in enumerate(province_gaps.columns):
    axes[0, col].set_title(condition)
    axes[1, col].set_xlabel("Urban minus rural (pts)")
axes[0, 0].set_ylabel("Province")
axes[1, 0].set_ylabel("Zone")
fig.suptitle("Fig 9. Urban–rural gaps by province and zone (weighted, no controls; 95% CI)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig09_gaps_by_region")

fuel = province_gaps["Clean cooking fuel"]
print(f"Finding: the clean-fuel gap ranges from {fuel.min():.0f} pts ({fuel.idxmin()}) to {fuel.max():.0f} pts "
      f"({fuel.idxmax()}); among zones the Hill has the largest gaps.")

# %% [markdown]
# **Interpretation.** The point estimates vary widely across the country. The clean-fuel gap is
# largest in Bagmati (which includes the Kathmandu Valley) and smallest in Madhesh. The Hill zone has
# the largest point estimates on all three conditions. The Mountain zone has only 11 urban clusters,
# so its intervals are wide. Overlapping or non-overlapping intervals are not a test of whether two
# regions differ; that needs a direct interaction test. Read the rankings as point estimates. These
# are unadjusted differences within each region: they describe where gaps are, not why.

# %% [markdown]
# ### 9.3 Do the gaps differ between regions? (interaction tests)
# One regression per condition with a separate urban gap for each region:
# `condition ~ C(region) + C(region):urban - 1` (household weights, PSU-clustered errors).
# - **Joint test**: are all regional gaps equal? (Wald test)
# - **Top vs next**: is the largest gap clearly larger than the second largest?

# %%
def region_tests(group_col, ci):
    rows = []
    for col, label in REGION_CONDITIONS.items():
        sample = hh.dropna(subset=[col])
        fit = smf.wls(f"{col} ~ C({group_col}) + C({group_col}):urban - 1", data=sample,
                      weights=sample["weight"]).fit(cov_type="cluster", cov_kwds={"groups": sample["cluster"]})
        names = list(fit.params.index)
        slopes = [i for i, n in enumerate(names) if n.endswith(":urban")]

        # Joint test: each regional gap minus the first one equals zero
        R = np.zeros((len(slopes) - 1, len(names)))
        for row, i in enumerate(slopes[1:]):
            R[row, slopes[0]], R[row, i] = -1, 1
        joint = fit.wald_test(R, scalar=True)

        # Top vs next: difference between the two largest point estimates
        ranked = ci[ci["condition"] == label].sort_values("gap", ascending=False)
        top, second = ranked[group_col].iloc[0], ranked[group_col].iloc[1]
        index_of = {n.split("[")[1].split("]")[0].replace(" province", "").title(): i
                    for i, n in enumerate(names) if n.endswith(":urban")}
        r = np.zeros(len(names))
        r[index_of[top]], r[index_of[second]] = 1, -1
        diff = fit.t_test(r)
        rows.append({"region_type": group_col, "condition": label,
                     "joint_chi2": float(joint.statistic), "joint_df": len(slopes) - 1, "joint_p": float(joint.pvalue),
                     "top": top, "second": second, "top_minus_second": 100 * float(diff.effect[0]),
                     "top_minus_second_se": 100 * float(diff.sd[0][0]), "top_minus_second_p": float(diff.pvalue)})
    return pd.DataFrame(rows)


region_test = pd.concat([region_tests("province", province_ci), region_tests("eco_zone", zone_ci)], ignore_index=True)
save_table(region_test.round(4), "gaps_region_tests")
display(region_test.round(3))

for _, t in region_test.iterrows():
    print(f"Finding: {t['condition']} by {t['region_type']}: equal-gaps test p = {t['joint_p']:.3f}; "
          f"{t['top']} minus {t['second']} = {t['top_minus_second']:.1f} pts (p = {t['top_minus_second_p']:.3f}).")

# %% [markdown]
# **Interpretation.** The tests separate real regional differences from noise.
# - **Clean fuel:** the gap differs across provinces (joint p < 0.001), and Bagmati's gap is larger
#   than the next largest, Gandaki's, by about 21 points (p = 0.02).
# - **Floors and walls by province:** the joint tests do not reject equal gaps (p = 0.18 and 0.14).
#   The province rankings for building materials are therefore not statistically supported.
# - **Zones:** all three gaps differ across zones (p ≤ 0.002), and the Hill gap exceeds the Terai gap
#   by about 19–29 points on each condition (p ≤ 0.001). The Mountain zone is too imprecise to rank.

# %% [markdown]
# ## 9b. Targeting: which households would different rules reach?
# Two groups of deprived households (all nine conditions observed):
# - **core-deprived:** lack clean fuel, a finished floor and finished walls together;
# - **multiply deprived:** lack four or more of the nine conditions.
#
# Four simple targeting rules, each covering roughly a third to two-fifths of households:
# rural residence; lacking any basic service (BSI < 3); the bottom two asset groups (our index);
# the bottom two DHS wealth quintiles. For each rule: the share of all households it selects
# (coverage), the share of deprived households it reaches, and the share of selected households
# that are deprived. Intervals: 300 PSU-cluster bootstrap resamples.
#
# *Note:* the DHS index contains floor, wall, water and sanitation items, so it partly selects on
# these conditions directly. That overlap is a problem for regression controls (Section 8) but can
# help a targeting rule.

# %%
complete["core_deprived"] = (lacks[CORE].sum(axis=1) == 3).astype(int)
complete["multi_deprived"] = (lacks.sum(axis=1) >= 4).astype(int)
RULES = {
    "Rural residence": lambda d: d["urban"] == 0,
    "Lacks a basic service": lambda d: d["BSI"] < 3,
    "Bottom two asset groups": lambda d: d["asset_group"] <= 2,
    "Bottom two DHS wealth quintiles": lambda d: d["wealth_dhs"].isin(["poorest", "poorer"]),
}


def targeting_stats(d):
    w = d["weight"]
    out = {"urban share of core-deprived": 100 * (w * d["core_deprived"] * d["urban"]).sum() / (w * d["core_deprived"]).sum(),
           "urban share of multiply deprived": 100 * (w * d["multi_deprived"] * d["urban"]).sum() / (w * d["multi_deprived"]).sum()}
    for rule, select in RULES.items():
        s = select(d).astype(int)
        out[f"{rule} | coverage"] = 100 * (w * s).sum() / w.sum()
        out[f"{rule} | reaches core"] = 100 * (w * s * d["core_deprived"]).sum() / (w * d["core_deprived"]).sum()
        out[f"{rule} | reaches multi"] = 100 * (w * s * d["multi_deprived"]).sum() / (w * d["multi_deprived"]).sum()
        out[f"{rule} | selected who are core"] = 100 * (w * s * d["core_deprived"]).sum() / (w * s).sum()
    return pd.Series(out)


point = targeting_stats(complete)
rng = np.random.default_rng(2022)
psu_index = complete.groupby("cluster").indices
psu_ids = np.array(list(psu_index))
draws = pd.DataFrame([targeting_stats(complete.iloc[np.concatenate([psu_index[c] for c in rng.choice(psu_ids, len(psu_ids))])])
                      for _ in range(300)])
targeting = pd.DataFrame({"estimate": point, "low": draws.quantile(0.025), "high": draws.quantile(0.975)})
save_table(targeting.round(2), "targeting")
display(targeting.round(1))

# Where do core-deprived households live? (weighted % of all core-deprived households)
core_hh = complete[complete["core_deprived"] == 1]
where = (core_hh.groupby(["province", "urban"])["weight"].sum().unstack() / core_hh["weight"].sum() * 100)
where = where.rename(columns={0: "rural", 1: "urban"})
where.index = where.index.str.replace(" province", "").str.title()
where["total"] = where.sum(axis=1)
where = where.sort_values("total", ascending=False)
save_table(where.round(2), "core_deprived_by_province")
display(where.round(1))

# %%
rules = list(RULES)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, (key, title) in zip(axes, [("reaches core", "Lack clean fuel, floor and walls"),
                                   ("reaches multi", "Lack 4 or more of the 9 conditions")]):
    est = [targeting.loc[f"{r} | {key}", "estimate"] for r in rules]
    lo = [targeting.loc[f"{r} | {key}", "low"] for r in rules]
    hi = [targeting.loc[f"{r} | {key}", "high"] for r in rules]
    cov = [targeting.loc[f"{r} | coverage", "estimate"] for r in rules]
    y = np.arange(len(rules))
    ax.barh(y, est, color=URBAN_COLOR, height=0.55)
    ax.errorbar(est, y, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="none", ecolor="black", elinewidth=1)
    for yi, e, h in zip(y, est, hi):
        ax.text(h + 2, yi, f"{e:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y, [f"{r}\n(selects {c:.0f}% of households)" for r, c in zip(rules, cov)])
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of deprived households reached")
    ax.set_title(title, fontsize=10)
axes[0].invert_yaxis()
fig.suptitle("Fig 9b. Share of deprived households each targeting rule would reach (95% CI)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig09b_targeting")

print(f"Finding: {point['urban share of core-deprived']:.1f}% of core-deprived households live in urban municipalities. "
      "Share of core-deprived reached: " + "; ".join(f"{r} {point[f'{r} | reaches core']:.1f}%" for r in rules))

# %% [markdown]
# **Interpretation.**
# - About half of the households lacking clean fuel, a finished floor and finished walls live in
#   **urban** municipalities: rural households are far more likely to be deprived, but most
#   households are urban, so in numbers the two areas are similar.
# - A **rural-only** rule reaches about half of them; a rule based on **lacking a basic service**
#   reaches fewer still, because most of these households have electricity, improved water and
#   sanitation.
# - **Asset-based** rules reach more, and the **DHS wealth quintiles** reach the most — because the
#   DHS index already includes housing and water items. The overlap that makes it a poor regression
#   control makes it an effective screen for these particular deprivations.
# - These rules are illustrations with similar coverage, not proposals; real programmes target on
#   other criteria too. The table shows which households each simple rule would overlook.

# %% [markdown]
# ## 10. Robustness
# ### 10.1 One change at a time (specification 5)
# - **Unweighted**: no survey weights.
# - **Weighted-PCA asset groups**: the asset index built with survey weights.
# - **Population weights**: household weight × usual residents, so results describe people.
# - **Excluding Kathmandu Valley**: drops Kathmandu, Lalitpur and Bhaktapur districts.

# %%
VALLEY = ["kathmandu", "lalitpur", "bhaktapur"]
rows = []
for outcome in list(INDICES) + ["HMCI_original"]:
    sample = hh.dropna(subset=MODEL_VARS + ["asset_group_wpca", outcome])
    no_valley = sample[~sample["district"].str.lower().isin(VALLEY)]
    versions = {
        "Main (weighted)": (urban_gap(outcome, SPECS[FULL], sample), sample),
        "Unweighted": (urban_gap(outcome, SPECS[FULL], sample, weighted=False), sample),
        "Weighted-PCA asset groups": (urban_gap(outcome, GEO + " + C(asset_group_wpca) + " + HEAD, sample), sample),
        "Population weights": (urban_gap(outcome, SPECS[FULL], sample.assign(weight=sample["people_weight"])), sample),
        "Excluding Kathmandu Valley": (urban_gap(outcome, SPECS[FULL], no_valley), no_valley),
    }
    for version, (fit, data) in versions.items():
        rows.append({"outcome": outcome, "version": version, "N": len(data), "urban_coef": fit.params["urban"],
                     "se": fit.bse["urban"], "p": fit.pvalues["urban"]})
robust = pd.DataFrame(rows)
save_table(robust.round(4), "robustness")
display(robust.round(3))

# %%
names = {"BSI": "Basic services", "CWI": "Cooking & water", "HMCI_dhs": "Housing materials & crowding",
         "HMCI_original": "Housing materials & crowding (original lists)"}
fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
for ax, (outcome, title) in zip(axes, names.items()):
    t = robust[robust["outcome"] == outcome].iloc[::-1]
    ax.errorbar(t["urban_coef"], t["version"], xerr=1.96 * t["se"], fmt="o", color=URBAN_COLOR)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=9)
fig.suptitle("Fig 10. Urban coefficient across robustness checks (95% CI)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig10_robustness")

print("Finding: the urban gap in cooking/water and housing materials and crowding stays positive and significant in every version; "
      "basic services stay close to zero.")

# %% [markdown]
# **Interpretation.** The main result does not depend on the survey weights, on how the asset index
# is built, on which list of finished materials is used, on describing people instead of households
# (population weights), or on the Kathmandu Valley (812 households in 28 clusters, about 16% of
# weighted urban households). The urban coefficient changes only slightly (CWI 0.19–0.22, HMCI
# 0.30–0.33) and stays clearly positive for cooking/water and housing materials and crowding, and
# close to zero for basic services. The national gap is therefore not driven by the capital.

# %% [markdown]
# ### 10.1b Full survey design: weights, PSUs **and** strata
# The main standard errors cluster by PSU but ignore the 14 sampling strata. Here we compute
# design-based (Taylor linearization) standard errors for the same weighted regressions: PSU totals
# of each household's score contribution are centred within their stratum, as in Stata's `svy`.
# Coefficients are unchanged; only the standard errors differ.

# %%
def design_se(fit, data):
    """Linearization SE of a WLS fit using weights, PSUs and strata (with-replacement PSU sampling)."""
    X = fit.model.exog
    e = fit.model.endog - X @ fit.params.values
    w = data["weight"].values
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    scores = pd.DataFrame(X * (w * e)[:, None])
    totals = scores.groupby([data["stratum"].values, data["cluster"].values]).sum()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, t in totals.groupby(level=0):
        n_h = len(t)
        d = t.values - t.values.mean(axis=0)
        meat += n_h / (n_h - 1) * d.T @ d
    V = bread @ meat @ bread
    return pd.Series(np.sqrt(np.diag(V)), index=fit.params.index)


rows = []
DESIGN_OUTCOMES = {"BSI": 1, "CWI": 1, "HMCI_dhs": 1, "clean_fuel": 100, "floor_dhs": 100, "wall_dhs": 100}
for outcome, scale in DESIGN_OUTCOMES.items():
    sample = hh.dropna(subset=MODEL_VARS + ["stratum", outcome]).reset_index(drop=True)
    fit = urban_gap(outcome, SPECS[FULL], sample)
    se_design = design_se(fit, sample)["urban"]
    rows.append({"outcome": outcome, "urban_coef": scale * fit.params["urban"],
                 "se_psu_clustered": scale * fit.bse["urban"], "se_design_strata": scale * se_design,
                 "p_design": 2 * (1 - stats.norm.cdf(abs(fit.params["urban"] / se_design)))})
design = pd.DataFrame(rows)
save_table(design.round(4), "design_based_se")
display(design.round(3))
print("Finding: SE ratio (design with strata / PSU-clustered):",
      "; ".join(f"{r.outcome} {r.se_design_strata / r.se_psu_clustered:.2f}" for r in design.itertuples()))

# %% [markdown]
# **Interpretation.** Adding the strata changes the standard errors by at most 2% (ratios 0.98–1.01),
# and no conclusion about statistical significance changes. The PSU-clustered errors used in the
# paper are a good approximation to full design-based inference here.

# %% [markdown]
# ### 10.2 Ordered logit (supplementary)
# Treats each index as ordered categories. It cannot use survey weights, so read it as a check only.

# %%
rows = []
for outcome in INDICES:
    sample = hh.dropna(subset=MODEL_VARS + [outcome])
    X = dmatrix("urban + " + SPECS[FULL], sample, return_type="dataframe").drop(columns="Intercept")
    fit = OrderedModel(sample[outcome].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False, maxiter=2000, cov_type="cluster", cov_kwds={"groups": sample["cluster"]})
    b, se = fit.params["urban"], fit.bse["urban"]
    rows.append({"outcome": outcome, "odds_ratio": np.exp(b), "low": np.exp(b - 1.96 * se),
                 "high": np.exp(b + 1.96 * se), "p": fit.pvalues["urban"]})
ordered = pd.DataFrame(rows)
save_table(ordered.round(3), "ordered_logit")
display(ordered.round(2))

fig, ax = plt.subplots(figsize=(6, 2))
labels = [INDICES[o].split(" (")[0] for o in ordered["outcome"]]
ax.errorbar(ordered["odds_ratio"], labels,
            xerr=[ordered["odds_ratio"] - ordered["low"], ordered["high"] - ordered["odds_ratio"]],
            fmt="o", color=URBAN_COLOR)
ax.axvline(1, color="black", linewidth=0.8)
ax.set_xlabel("Odds ratio, urban vs rural (1 = no difference)")
ax.set_title("Fig 11. Ordered logit: odds of a higher score")
save_figure(fig, "fig11_ordered_logit")

o = ordered.set_index("outcome")["odds_ratio"]
print(f"Finding: urban households have about {o['CWI']:.1f}x the odds of a higher cooking/water score and "
      f"{o['HMCI_dhs']:.1f}x for housing materials and crowding; {o['BSI']:.2f} for basic services (no difference).")

# %% [markdown]
# **Interpretation.** A different model gives the same picture: urban residence is associated with
# roughly twice the odds of a higher cooking/water or housing score, and no difference in basic
# services. Because this model cannot use the survey weights, it is a supplementary check only.

# %% [markdown]
# ## 11. Migration (supplementary)
# District absentee-abroad rates (Census 2021) measure **how common migration is**, not whether a
# household receives remittances. The results are district-level correlations.
#
# ### 11.1 Read and check the census table

# %%
rows = []
for line in CENSUS_FILE.read_text().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split()
    numbers = [float(x.replace(",", "")) for x in parts[-8:]]   # last 8 entries are numbers
    district = " ".join(parts[:-8])                             # the rest is the district name
    rows.append([district] + numbers)
census = pd.DataFrame(rows, columns=["district", "native_pop", "in_migrants", "in_rate", "out_migrants",
                                     "out_rate", "net_migration", "net_rate", "absentee_abroad"])

# Checks against the published table
assert len(census) == 77, "expected 77 districts"
assert (census["in_migrants"] - census["out_migrants"] == census["net_migration"]).all()
assert census["absentee_abroad"].sum() == 2_190_592, "national total does not match the census"

census["absentee_rate"] = 100 * census["absentee_abroad"] / census["native_pop"]   # per 100 residents
save_table(census.set_index("district"), "census2021_migration")
print("Census checks passed.")
display(census.sort_values("absentee_rate", ascending=False).head(10))

# %% [markdown]
# **Interpretation.** The table reproduces the official national total (2,190,592 absentees abroad)
# and its own in-, out- and net-migration arithmetic, so the transcription is correct. Absentee
# rates vary widely, from about 1 to 20 per 100 residents.

# %% [markdown]
# ### 11.2 Link each DHS household to its district's rate
# Two names are spelled differently; Manang and Rukum East are not in the DHS sample.

# %%
def district_key(name):
    key = str(name).lower().replace("(", "").replace(")", "").replace(" ", "")
    return {"dhanusha": "dhanusa", "terhathum": "tehrathum"}.get(key, key)


rate = dict(zip(census["district"].map(district_key), census["absentee_rate"]))
hh["absentee_rate"] = hh["district"].map(lambda d: rate.get(district_key(d)))
assert hh["absentee_rate"].notna().all(), "a district did not match"

hh["migration"] = hh["absentee_rate"] / 5          # results per 5 absentees per 100 residents
hh["district_id"] = hh["district"].astype("category").cat.codes
print(hh["district"].nunique(), "DHS districts matched")
display(hh[["district", "absentee_rate"]].drop_duplicates().head())

# %% [markdown]
# ### 11.3 Picture: rural living conditions vs district migration

# %%
rows = []
for district, d in hh[hh["urban"] == 0].groupby("district"):
    rows.append({"district": district, "absentee_rate": d["absentee_rate"].iloc[0], "households": len(d),
                 "rural_HMCI": (d["HMCI_dhs"] * d["weight"]).sum() / d.loc[d["HMCI_dhs"].notna(), "weight"].sum(),
                 "rural_clean_fuel": wpct(d["clean_fuel"], d["weight"])})
rural_districts = pd.DataFrame(rows).set_index("district")
save_table(rural_districts.round(3), "migration_rural_districts")
display(rural_districts.sort_values("absentee_rate", ascending=False).head())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, col, label in [(axes[0], "rural_HMCI", "Rural housing materials & crowding (mean HMCI)"),
                       (axes[1], "rural_clean_fuel", "Rural households with clean fuel (%)")]:
    ax.scatter(rural_districts["absentee_rate"], rural_districts[col], s=rural_districts["households"] / 3,
               color=URBAN_COLOR, alpha=0.5)
    slope, intercept = np.polyfit(rural_districts["absentee_rate"], rural_districts[col], 1)
    x = np.linspace(rural_districts["absentee_rate"].min(), rural_districts["absentee_rate"].max(), 20)
    ax.plot(x, intercept + slope * x, color=GREY, linestyle="--")
    corr = rural_districts["absentee_rate"].corr(rural_districts[col])
    ax.set_title(f"{label}\n(correlation {corr:+.2f})", fontsize=9)
    ax.set_xlabel("Absentees abroad per 100 residents (district)")
fig.suptitle("Fig 12. Rural living conditions by district migration (dot size = households)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig12_migration_districts")

# %% [markdown]
# **Interpretation.** Across districts there is no clear relationship between migration and rural
# housing materials and crowding (correlation about +0.14) or rural clean-fuel use (about +0.02); the fitted lines
# are nearly flat.

# %% [markdown]
# ### 11.4 Regressions with district migration
# `urban * migration` adds urban, migration and their interaction:
# - `migration` = association among **rural** households
# - `urban:migration` = change in the urban gap
#
# Standard errors are clustered by **district**, because migration varies by district.

# %%
rows = []
for outcome in ["BSI", "CWI", "HMCI_dhs", "clean_fuel", "floor_dhs", "wall_dhs"]:
    scale = 100 if outcome in COMPONENTS else 1        # 0/1 outcomes in percentage points
    sample = hh.dropna(subset=MODEL_VARS + [outcome, "migration"])
    for controls_name, controls in [("geography", GEO), ("assets + head", SPECS[FULL])]:
        fit = smf.wls(f"{outcome} ~ urban * migration + {controls}", data=sample, weights=sample["weight"]).fit(
            cov_type="cluster", cov_kwds={"groups": sample["district_id"]})
        rows.append({"outcome": outcome, "controls": controls_name,
                     "migration": scale * fit.params["migration"], "migration_se": scale * fit.bse["migration"],
                     "migration_p": fit.pvalues["migration"],
                     "urban_x_migration": scale * fit.params["urban:migration"],
                     "urban_x_migration_se": scale * fit.bse["urban:migration"],
                     "urban_x_migration_p": fit.pvalues["urban:migration"]})
migration = pd.DataFrame(rows)
save_table(migration.round(4), "migration_regressions")
display(migration.round(3))

# %%
READABLE = {"BSI": "Basic services", "CWI": "Cooking & water", "HMCI_dhs": "Housing materials & crowding",
            "clean_fuel": "Clean fuel", "floor_dhs": "Finished floor", "wall_dhs": "Finished walls"}
t = migration[migration["controls"] == "assets + head"].copy()
t["name"] = t["outcome"].map(READABLE)
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
for ax, outcomes, units in [(axes[0], ["BSI", "CWI", "HMCI_dhs"], "index points"),
                            (axes[1], ["clean_fuel", "floor_dhs", "wall_dhs"], "percentage points")]:
    d = t[t["outcome"].isin(outcomes)]
    ax.errorbar(d["migration"], d["name"] + ": rural", xerr=1.96 * d["migration_se"], fmt="o",
                color=URBAN_COLOR, label="Rural households")
    ax.errorbar(d["urban_x_migration"], d["name"] + ": urban gap", xerr=1.96 * d["urban_x_migration_se"],
                fmt="o", mfc="white", color=URBAN_COLOR, label="Change in urban gap")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Per 5 absentees per 100 residents ({units})")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
fig.suptitle("Fig 13. District migration and living conditions (95% CI; asset and head controls)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save_figure(fig, "fig13_migration_coefficients")

districts = hh.drop_duplicates("district")
province_r2 = smf.ols("absentee_rate ~ C(province)", data=districts).fit().rsquared
smallest_p = migration[["migration_p", "urban_x_migration_p"]].min().min()
print(f"Finding: no estimate is statistically significant (smallest p = {smallest_p:.2f}). Province explains "
      f"{100 * province_r2:.0f}% of the differences in district absentee rates. The district-level analysis does "
      "not detect an association; it does not establish whether household remittances affect housing.")

# %% [markdown]
# **Interpretation.** None of the migration estimates can be distinguished from zero, with or
# without asset controls. The district rate measures how common migration is, not whether a
# household receives remittances, and much of its variation is between provinces. The conclusion is
# limited: *the district-level analysis does not detect an association*. It does not show whether
# household remittances affect housing; that needs household-level remittance data (NLSS-IV).
# Keep this section supplementary in the paper.

# %% [markdown]
# ## 12. Summary of findings
# Every number is read from the tables above.

# %%
summary = pd.DataFrame([
    ["Weighting matters", f"urban share {urban_sample:.1f}% sample vs {urban_weighted:.1f}% weighted", "Fig 1"],
    ["Basic services near-universal", f"electricity {table1.loc['electricity', 'all']:.1f}%, improved water {table1.loc['water_improved', 'all']:.1f}%", "Fig 4"],
    ["Largest raw gaps", f"clean fuel {table1.loc['clean_fuel', 'gap']:.1f}, floor {table1.loc['floor_dhs', 'gap']:.1f}, walls {table1.loc['wall_dhs', 'gap']:.1f} pts", "Fig 4"],
    ["Gaps after adjustment", f"clean fuel {g['Clean cooking fuel']:.1f}, floor {g['Finished floor']:.1f}, walls {g['Finished walls']:.1f} pts", "Fig 8"],
    ["Wealth measure matters", f"HMCI: {hmci.iloc[0]:.2f} -> {hmci.iloc[1]:.2f} (DHS wealth) vs {hmci.iloc[2]:.2f} (asset groups)", "Fig 7"],
    ["Profiles", f"basic services and living conditions met: {profile_by_assets.loc[1, TOP_PROFILE]:.1f}% to {profile_by_assets.loc[5, TOP_PROFILE]:.1f}% across asset groups", "Fig 6"],
    ["Regional variation", f"clean-fuel gap {fuel.min():.0f}-{fuel.max():.0f} pts across provinces", "Fig 9"],
    ["Robust", "same result across weighting, PCA and material lists", "Figs 10-11"],
    ["Migration (supplementary)", "no association detected at district level", "Figs 12-13"],
], columns=["finding", "evidence", "figure"])
save_table(summary.set_index("finding"), "summary_of_findings")
display(summary)

# %% [markdown]
# **Overall interpretation.** In 2022, urban and rural households in Nepal differed little in access
# to electricity, improved water and basic sanitation, but differed substantially in clean cooking
# fuel and in finished floors and walls. About half of these differences remain after adjustment for
# province, ecological zone, household assets and household-head characteristics. How much of the
# gap a wealth control "explains" depends heavily on the measure: the DHS wealth index shares items
# with the outcomes and absorbs most of the gap, while an asset-only measure leaves most of it.
#
# **Limitations to state in the paper**
# - One cross-section: no causal effect of urban residence, and no statement about change over time.
# - "Urban" is the administrative (municipality) definition, not population density.
# - Inference is cluster-robust; the 14 sampling strata are not modelled.
# - The asset index is an imperfect measure of economic resources.
# - Migration is measured at district level; there is no household remittance variable in the DHS.

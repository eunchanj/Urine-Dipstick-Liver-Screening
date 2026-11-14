# Urine dipstick results added little to age, sex and body mass index in models predicting elevated aminotransferases in adults

This is the analysis code of the paper named in the title. No individual-level records are in this repository, and it contains no fitted models and no results.

## Authors

- Eun Chan Jang, https://orcid.org/0009-0002-4745-0167
- Pado Song, https://orcid.org/0000-0002-8302-6833
- Joung Eun Kim, https://orcid.org/0009-0002-1893-1156
- Myung Hwa Han, https://orcid.org/0000-0002-2335-9492
- Kiheon Lee (corresponding author), https://orcid.org/0000-0002-7139-2342

## The study in brief

The models rank adults by their risk of a raised aminotransferase on the day of a urinalysis, from urine dipstick results, age, sex and body mass index (BMI).

- **Outcomes.** Alanine aminotransferase (ALT) above 40 U/L, and aspartate aminotransferase (AST) above 40 U/L. A stored value counts as positive only when it is above 40 U/L, so exactly 40 U/L is negative. The same rule holds in both data sources and for women and men.
- **Two separate models.** ALT above 40 U/L and AST above 40 U/L each have their own models, thresholds and alerts. The two outcomes are never joined into one.
- **Development data.** Hospital records, divided at random by person into a training set and a development test set, within the four groups that ALT and AST above or not above 40 U/L form. Every threshold is chosen in the development test set.
- **External evaluation.** The Korea National Health and Nutrition Examination Survey (KNHANES) 2016-2024. The survey data serve only for evaluation: no model is fitted, calibrated or given a threshold with them.
- **Ensemble.** LightGBM, XGBoost and CatBoost learners, each tuned with Optuna, averaged by soft voting and calibrated by isotonic regression.
- **Comparisons.** Logistic regression with age, sex and BMI (`LR_ASB`), logistic regression with all 12 inputs (`LR_ALL`), and the rule BMI ≥25 kg/m² (`BMI25`).

## Inputs

The models have 12 inputs: nine urine dipstick results (pH, specific gravity, nitrite, and the graded glucose, protein, ketone, bilirubin, blood and urobilinogen results), age, sex and BMI.

The dipstick's leukocyte result is not an input, because the survey does not report it.

## Scripts

The scripts run in the order of their numbers. Scripts s00-s07 read individual-level data, so whoever holds the data runs them. Scripts s08-s11 read only the aggregate tables that s00-s07 write and turn them into tables, figures and a file of numbers.

The aggregate tables are the CSV and JSON files that s00-s07 write at the top level of `working/`, together with the text file `environment.txt`. In them, a count of 1 to 9 appears as `<10`. Files with a row for each person stay in `working/cache/`, and the fitted models stay in the model folder.

| Script | Reads | Writes |
|---|---|---|
| `s00_env_check.py` | File details and single columns of the hospital and survey files | `environment.txt`, `knhanes_variables.csv`, `knhanes_values.csv`, `knhanes_numeric.csv`, `dev_schema.csv`, `dev_values.csv`, `dev_numeric.csv`, `data_checks.csv` |
| `s01_build_cohort.py` | The hospital and survey files, `notes/knhanes_codes.csv` | Person-level `cache/dev.parquet` and `cache/knhanes.parquet`; `cohort_flow.csv`, `cohort_checks.csv`, `baseline_characteristics.csv`, `input_levels.csv`, `knhanes_coding.csv`, `enzyme_by_year.csv` |
| `s02_train.py` | The training set in `cache/dev.parquet`, `cohort_flow.csv` | In the model folder: `ens_ALT.joblib`, `ens_AST.joblib`, six `optuna_<outcome>_<learner>.csv`, `ens_config.json` and a `.gitignore`; `hyperparameters.csv`, `train_summary.csv` |
| `s03_predict_evaluate.py` | The two ensembles, the two caches, `train_summary.csv`, `cohort_flow.csv` | Four person-level `cache/preds_ENS_<outcome>_<dataset>.parquet`; `performance.csv`, `calibration.csv`, `calibration_bins.csv`, `prediction_histogram.csv`, `roc_points.csv` |
| `s04_comparators.py` | The two caches, the predictions of s03, `train_summary.csv`, `performance.csv`, `cohort_flow.csv` | Eight person-level `cache/preds_<regression>_<outcome>_<dataset>.parquet`; four `lr_<regression>_<outcome>.joblib` in the model folder; `model_comparison.csv`, `roc_points_lr.csv`, `lr_coefficients.csv` |
| `s05_thresholds.py` | The two caches, the predictions of s03 and s04, `performance.csv`, `model_comparison.csv`, `cohort_flow.csv` | `thresholds.json`, `operating_points.csv`, `comparators_alert.csv`, `decision_curve.csv` |
| `s06_subgroups_sensitivity.py` | The two caches, the predictions of s03, `performance.csv`, `calibration.csv`, `cohort_flow.csv` | `subgroups.csv`, `sensitivity.csv` |
| `s07_shap.py` | The two ensembles, the two caches (of which only the development test set is used), the predictions of s03, `train_summary.csv`, `performance.csv`, `cohort_flow.csv` | Person-level `cache/shap_ALT.parquet` and `cache/shap_AST.parquet`; `shap_summary.csv`, `shap_by_level.csv` |
| `s08_tables.py` | The aggregate tables | `manuscript/tables/`, `manuscript/tables.md`, `supplement/tables/` |
| `s09_figures.py` | The aggregate tables | `figures/`, `supplement/figures/` |
| `s10_supplement.py` | What s08 and s09 wrote under `supplement/` | `supplement/Supplementary_Information.md`, `supplement/supplement_index.csv` |
| `s11_numbers.py` | The aggregate tables | `manuscript/numbers.csv` |

In the file names, `<outcome>` is `ALT` or `AST`, `<dataset>` is `DEV_TEST` (the development test set) or `KN` (the eligible KNHANES adults), `<learner>` is `lgb`, `xgb` or `cat`, and `<regression>` is `LR_ASB` or `LR_ALL`. The output folders of s08-s11 are made next to `working/`.

s05 chooses every threshold from the predictions of the development test set, saves the thresholds in `thresholds.json` and applies them from that file, unchanged, to the development test set and to KNHANES. No alert threshold of a model is written in the code.

The other files in `working/` hold shared code:

- `cohort.py`: the cohorts, the person-level files and the check of the settings
- `recipe.py`: tuning and fitting of the ensemble
- `common.py`: the statistics
- `runlog.py`: status lines and the masking of small counts
- `display.py` and `figstyle.py`: the formats of the tables and figures
- `aggregate_guard.py`: keeps s08-s11 from opening individual-level files

In `cohort.py` and `s00_env_check.py`, the word "atlas" refers to the three derived hospital tables `LAB_DAY`, `DIPSTICK_EPISODE` and `ANTHRO_DAY`.

The scripts that build the manuscript files are not part of this repository, so `s10_supplement.py` puts `[[TEXT: ...]]` placeholders where the methods text of the supplement would go, and the reporting checklists that its contents line places in a Supplementary Note are not made here.

This copy of the code was edited for readers after the analysis had been run. Comments and messages were edited, one internal name became `ROOT`, the model folder must now be named `models`, `s00_env_check.py` lists the versions of the packages of `requirements.txt` and writes one row fewer to `data_checks.csv`, and `aggregate_guard.py` knows only the output folders of these scripts. Everything else is the code that was run.

## Environment

The analysis ran on Python 3.13.5 with these package versions, which `requirements.txt` pins:

| Package | Version |
|---|---|
| catboost | 1.2.8 |
| cycler | 0.11.0 |
| joblib | 1.4.2 |
| lightgbm | 4.6.0 |
| matplotlib | 3.10.0 |
| numpy | 2.1.3 |
| optuna | 4.8.0 |
| pandas | 2.2.3 |
| pillow | 11.1.0 |
| pyarrow | 19.0.0 |
| pyreadstat | 1.3.5 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.3 |
| xgboost | 3.1.2 |

Install them with:

```sh
pip install -r requirements.txt
```

The commands below are written for a macOS or Linux shell. On macOS, LightGBM and XGBoost need the OpenMP library (`brew install libomp`).

The main figures are drawn in Arial, and `s09_figures.py` stops when that font is missing.

The analysis used a NumPy built on OpenBLAS. A NumPy built on another linear algebra library, such as the NumPy that pip installs on macOS, can change results in the last digits, and predicted risks that were tied can then come apart.

## Data

The Korea Disease Control and Prevention Agency provides the KNHANES files after registration at https://knhanes.kdca.go.kr.

The hospital records cannot be made public: Korea's Personal Information Protection Act and the hospital's own rules prevent it. Requests for access go to the corresponding author of the paper, and any release needs institutional approval and a data use agreement with the hospital.

The corresponding author shares the fitted ALT and AST model files on reasonable request. A model file loads with joblib under the package versions above. An ensemble (`ens_ALT.joblib`, `ens_AST.joblib`) takes a table with the columns of `INPUTS` in `cohort.py`, in that order and with that coding; a regression (`lr_<regression>_<outcome>.joblib`) takes the columns of `LR_INPUTS` for its model.

The evaluation in KNHANES needs a model file and the survey files, not the hospital files. With `paths_local.py` set up, run Python in the `working` folder: after `import cohort as C`, `kn = C.kn_cohort(C.load_code_table())[0]` gives the eligible survey adults with the inputs coded as in the paper (it reads only the survey files and the code table), `model.predict_proba(kn[C.INPUTS])[:, 1]` gives the predicted risks of an ensemble `model` loaded with joblib, and `kn.alt_value > 40` and `kn.ast_value > 40` give the two outcomes. Scripts s03-s07 themselves need the person-level files that s01 makes from the hospital files, and s03 and s07 load an ensemble only when its SHA-256 is the one recorded in the `train_summary.csv` that s02 wrote with it.

## Hospital files

The code reads five files made from the hospital's records; the scripts that made them are not included. To use other records, prepare files with the same columns. Both CSV files are in UTF-8.

- `DATA_PATH` (CSV): the first urinalysis of each person with a blood test within 365 days of it (the first row of a person is used). Columns: `일련번호` (person number, a whole number), `소변검사일자` (urinalysis date), `age_at_urine_test`, `성별` (`M` or `F`), `days_difference` (days to the nearest blood test, used in checks only), `수진일자` (health check-up date, empty if none), `ua_glucose`, `ua_protein`, `ua_ketone`, `ua_bilirubin`, `ua_blood`, `ua_urobilinogen`, `ua_nitrite`, `ua_pH`, `ua_sg`; optionally `ALT` and `AST` (values of the nearest day, used in one agreement count only).
- `RAW_FIRST` (CSV, a row for each person): `일련번호`, `소변검사일시` (date and time) and `ua_nitrite` as the recorded text (such as POSITIVE, +, NEGATIVE, -, 0). s00 also counts the texts of `ua_glucose`.
- `LAB_DAY` (Parquet, a row for each person and day): `id`, `date`, `alt` and `ast` (U/L), each with `_cmp` (its comparison sign: empty, =, <, <=, >, >=), `_nrep` (the number of draws that day) and `_last` (the value of the last draw), then `age`, `adult` and `is_checkup`.
- `DIPSTICK_EPISODE` (Parquet): `id`, `dt` and `nitrite_pos`, used only to cross-check nitrite.
- `ANTHRO_DAY` (Parquet): `id`, `date`, `height` (cm), `weight` (kg), `bmi` (kg/m²) and `source`. s01 takes the BMI of the urinalysis day, else the latest one of the 30 days before it; among the records of one day, a record whose source text contains `check` or `검진` (a health check-up) is used first. When the table also has the columns `bmi_derived` and `nursing_conflict`, s00 counts them, and s01 counts the BMI records of the cohort that carry `nursing_conflict`.

Graded dipstick results are stored as the numbers 0, 0.5, 1, 2, 3 (urine glucose also 4) and nitrite as 0 or 1. Text in these columns is read as missing, and a missing nitrite is then taken from `RAW_FIRST`. s01 stops when the median ALT or AST of a cohort lies outside 10-40 U/L or its median BMI outside 20-28 kg/m², which catches a wrong unit. `STUDY_START` and `STUDY_END` in `cohort.py` set the study period; the flow labels in `dev_cohort()` of the same file state the period and need the same change. No made-up data are included.

## How to run

All commands run in a terminal from the `working` folder, in the order below. First, once:

```sh
cd working
cp paths_local.example.py paths_local.py
mkdir -p logs cache
```

Then edit `paths_local.py` and replace each placeholder path with the real one. Scripts s01-s07 check two settings when they start: the last folder of `MODEL_DIR` must be named `models` (`MODEL_FOLDER` in `cohort.py`), and `KN_CODE_TABLE` must point into the `notes` folder of this repository, as in the example. `.gitignore` keeps `paths_local.py`, `cache/`, `logs/` and all outputs out of the repository.

Scripts s00-s07 are each run in the same way. Standard output goes to `logs/<script>.out` and contains only counts and run times; standard error goes to `cache/<script>.err`. An error message there can contain values from the data, so keep `cache/` on your own computer. Each script also writes a single line to `logs/<script>.status`: `OK`, or `FAIL <exception type> <file>:<line>`. If `grep` prints no status line, the script stopped before its main part began, for example at an import.

The longer steps need a computer that does not go to sleep and stays on mains power; `s02_train.py` takes the longest.

### Step 0: environment and data layout (s00)

```sh
rm -f logs/s00_env_check.status
python s00_env_check.py > logs/s00_env_check.out 2> cache/s00_env_check.err
grep -H . logs/s00_env_check.status
```

`notes/knhanes_codes.csv` gives, for each survey year, the meaning of the codes of the pregnancy and alcohol-drinking variables. Check it against `knhanes_variables.csv`, `knhanes_values.csv` and the KNHANES codebook. s01 stops if the file is missing or if the survey files hold a code that it does not list.

If you edit the table in Excel, save it as "CSV UTF-8". s01 also reads CP949, the plain CSV of Korean Excel, but no other encoding: a file saved as "Unicode text" stops it with `UnicodeDecodeError`. The row `code table encoding` of `cohort_checks.csv` records which one was read.

### Step 1: cohorts and split (s01)

```sh
rm -f logs/s01_build_cohort.status
python s01_build_cohort.py > logs/s01_build_cohort.out 2> cache/s01_build_cohort.err
grep -H . logs/s01_build_cohort.status
```

If a check stops the run, `cohort_checks.csv` (and `knhanes_coding.csv`, once the survey files have been read) is still written up to that point and ends with a `RUN` row, and the status line names the code line of the check.

Two settings in `cohort.py` decide rare hospital values: `GLU4_RULE` (urine glucose 4+ is merged with 3+) and `SEX_MISSING_RULE` (a person without a recorded sex is left out). The `RULE` rows of `cohort_checks.csv` count the people that each one affects. After changing either, go on as described under "Running a step again".

### Step 2: fitting the ensembles (s02)

```sh
rm -f logs/s02_train.status
python s02_train.py > logs/s02_train.out 2> cache/s02_train.err
grep -H . logs/s02_train.status
```

s02 makes the model folder when it is missing, and it replaces `ens_config.json` and the `optuna_*.csv` files in it, so give each analysis its own model folder: a folder named `models` in a parent folder of its own. CatBoost also writes a `catboost_info/` folder in `working/`. s02 does not overwrite fitted ensembles: it stops when `ens_*.joblib` files are already in the model folder. To fit them again, first move them into a subfolder, from inside the model folder:

```sh
mkdir -p old && mv ens_*.joblib old/
```

### Step 3: evaluation, comparisons, thresholds, subgroups and SHAP (s03-s07)

```sh
rm -f logs/s03_predict_evaluate.status logs/s04_comparators.status logs/s05_thresholds.status \
      logs/s06_subgroups_sensitivity.status logs/s07_shap.status
for s in s03_predict_evaluate s04_comparators s05_thresholds s06_subgroups_sensitivity s07_shap; do
  python $s.py > logs/$s.out 2> cache/$s.err || break
done
grep -H . logs/s0[3-7]_*.status
```

The loop stops at the first failure; the scripts after it then have no status file.

### Step 4: tables, figures and numbers (s08-s11)

```sh
PYTHONDONTWRITEBYTECODE=1 python s08_tables.py
PYTHONDONTWRITEBYTECODE=1 python s09_figures.py
PYTHONDONTWRITEBYTECODE=1 python s10_supplement.py
PYTHONDONTWRITEBYTECODE=1 python s11_numbers.py
```

Each one prints a summary line at the end, exits with status 1 when it finds a problem, and writes the list of the files it opened to `logs/opened_<script>.txt`. Some of them also report that `text_patterns.py` is missing and that their word checks were skipped. That is expected: the file holds local writing checks and is not part of this repository.

These four scripts were written for the tables of this study. On other data, s08 and s09 may stop where a count of 1 to 9 would be shown or could be worked out by subtraction, for example from a flow step that removes 1 to 9 people; the aggregate tables of s00-s07 are complete by then. s08-s11 also refuse any path through a folder whose name starts with `00_` or `03_`, so keep the repository out of such folders.

### Running a step again

Scripts s01-s11 delete their own earlier outputs when they start, and s00 overwrites its outputs. Later scripts check the SHA-256 of the outputs they read, so when the output of a step changes, run that step and every later one again.

- When the cohort changes, move `ens_*.joblib` to `old/` as in step 2 and go on from s02. Otherwise s03 and s04 stop, because the training set no longer matches `train_summary.csv`.
- After s02, run s03-s07 again.
- After any of s01-s07, run s08-s11 again. No later script reads the outputs of s00.

To see every status line at once: `grep -H . logs/*.status`

## Licence

The code is released under the MIT licence (`LICENSE`); see also the patent notice in `NOTICE`.

## Funding

This research was supported by the Korea Health Technology R&D Project through the Korea Health Industry Development Institute, funded by the Ministry of Health and Welfare, Republic of Korea (grant number RS-2025-02307233). This work was supported by the Laboratory Specialized Startup Leading University Program funded by the Ministry of Education (National Research Foundation of Korea) and the Ministry of Science and ICT (Commercialization Promotion Agency for R&D Outcomes), Republic of Korea (NTIS: 2710101030).

## Generative artificial intelligence

How generative artificial intelligence was used is described in the Methods of the paper.

## How to cite

Please cite the paper:

```text
Eun Chan Jang, Pado Song, Joung Eun Kim, Myung Hwa Han and Kiheon Lee. Urine dipstick results added little to age, sex and body mass index in models predicting elevated aminotransferases in adults. 2026.
```

Release v1.0 of this code, the version cited in the paper, is archived on Zenodo: https://doi.org/10.5281/zenodo.24213318.

`CITATION.cff` gives the citation details of this code; GitHub shows them under "Cite this repository".

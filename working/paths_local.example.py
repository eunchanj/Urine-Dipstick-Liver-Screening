"""Local settings of the scripts that read individual-level data (s00-s07).

Copy this file to paths_local.py in the same folder, then replace every placeholder path with the real one.
paths_local.py stays on your computer: .gitignore keeps it out of the repository. s08-s11 do not read it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# False for a real run. True only for a trial on made-up data: the survey files may then be transport (.xpt) files,
# cohort.check_paths skips its two checks, and the variables LIV_SYNTH_TRIALS and LIV_SYNTH_BOOT may lower the number
# of Optuna trials and of bootstrap resamples.
SYNTHETIC = False

# hospital files (their columns: PROC_COLS and LAB_COLS at the top of cohort.py, and the readers in cohort.py)
DATA_PATH = "/path/to/hospital/processed_first_visit.csv"       # first urinalysis of each person (first row used)
RAW_FIRST = "/path/to/hospital/raw_first_visit.csv"             # the same urinalyses with nitrite as recorded text
LAB_DAY = "/path/to/hospital/derived/lab_day_wide.parquet"      # ALT and AST, a row for each person and day
DIPSTICK_EPISODE = "/path/to/hospital/derived/dipstick_episode_wide.parquet"    # a row for each urinalysis
ANTHRO_DAY = "/path/to/hospital/derived/anthro_day.parquet"     # height, weight and BMI records

# KNHANES: the SAS file of each survey year, from https://knhanes.kdca.go.kr
KN_YEARS = list(range(2016, 2025))
KN_FILES = {y: f"/path/to/knhanes/hn{str(y)[-2:]}_all.sas7bdat" for y in KN_YEARS}
KN_CODE_TABLE = os.path.join(HERE, os.pardir, "notes", "knhanes_codes.csv")     # in this repository

# fitted models: a folder named models (MODEL_FOLDER of cohort.py) outside the repository; s02 makes it
MODEL_DIR = "/path/to/models"

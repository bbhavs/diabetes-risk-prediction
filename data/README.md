# Data Access

Raw and analytical data are **deliberately not included** in this repository.

> ⚠️ **Important:** I had access to `updated_dataset.csv` and `LASI_with_residence.csv` (the 62,957-row analytical extracts used for modelling) during development, but did **not** copy them into this public repo. LASI Wave 1 is distributed by IIPS under a data-use agreement that typically requires individual request/approval — redistributing even a de-identified derived extract in a public GitHub repo may violate those terms. Please check your own LASI data-use agreement before deciding whether to commit `data/processed/*.csv` anywhere public. If you're sure it's permitted, you can add it under `data/processed/` yourself (it's excluded by `.gitignore`).

## LASI Wave 1
- Request access via the International Institute for Population Sciences (IIPS): https://iipsindia.ac.in/content/LASI-data
- Once obtained, place the linked individual + biomarker + DBS files here as:
  - `data/raw/lasi_individual.sav`
  - `data/raw/lasi_biomarker.sav`
  - `data/raw/lasi_dbs.sav`

## NHANES
- Publicly downloadable from the CDC: https://www.cdc.gov/nchs/nhanes
- Place harmonised extract as `data/raw/nhanes_subset.csv`

## Processed data
After running `src/preprocessing.py`, a cleaned analytical dataset will be written to `data/processed/`. This is also excluded from version control (see `.gitignore`) — regenerate it locally from the raw files above.

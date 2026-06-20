# Models

The trained model artifact (`xgboost_pipeline.joblib`, referenced as `xgboost_diabetes_model.pkl` in some scripts) is **not included** in this repo — it wasn't available as a file to copy in, and trained model binaries shouldn't go into plain git history anyway.

## To regenerate it
Run:
```bash
python src/models/xgb.py
```
This trains the final XGBoost pipeline on the full dataset and saves it via `joblib.dump(...)` (see the bottom of `xgb.py`).

## To add it to this repo properly
Model files are typically 1–50MB+ — too large/binary for normal git diffs. Use **Git LFS**:
```bash
git lfs install
git lfs track "*.joblib" "*.pkl"
git add .gitattributes
git add models/xgboost_pipeline.joblib
git commit -m "Add trained XGBoost pipeline via LFS"
```
Alternatively, upload it as a **GitHub Release asset** instead of committing it to the repo, and link to it from the main README.

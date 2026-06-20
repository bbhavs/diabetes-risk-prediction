# ==================================================
# XGBOOST – DIABETES RISK PREDICTION (NO LEAKAGE)
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("D:/updated_dataset.csv")
print("Dataset shape:", df.shape)


# -----------------------------
# Define Target
# -----------------------------
df["diabetes"] = (df["hba1c"] >= 6.5).astype(int)

print("\nTarget distribution:")
print(df["diabetes"].value_counts(normalize=True))


# -----------------------------
# Feature / Target Split
# -----------------------------
X = df.drop(columns=["diabetes", "hba1c"])
y = df["diabetes"]

X = X.drop(columns=["prim_key"], errors="ignore")


# -----------------------------
# Remove Leakage Variables
# -----------------------------
X = df.drop(columns=[

    "prim_key",
    "hba1c",
    "hb003_a",
    "dm008",
    "diabetes",
    "idrs_group",
    "dm001",
    "hypertension"
], errors="ignore")



print("\nFinal features:")
print(X.columns.tolist())


# -----------------------------
# Identify Feature Types
# -----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns


# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -----------------------------
# XGBoost Model
# -----------------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42
)


pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("xgb", xgb)
])


# -----------------------------
# Hyperparameter Grid
# -----------------------------
param_grid = {
    "xgb__n_estimators": [200, 300],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__subsample": [0.8],
    "xgb__colsample_bytree": [0.8]
}


# -----------------------------
# Cross-Validation
# -----------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# -----------------------------
# Grid Search (Optimize ROC-AUC)
# -----------------------------
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)


# -----------------------------
# Evaluation
# -----------------------------
best_model = grid.best_estimator_

y_prob = best_model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.50   # screening-friendly
y_pred = (y_prob >= THRESHOLD).astype(int)

print(f"\nClassification Report (Threshold = {THRESHOLD}):")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)


# -----------------------------
# ROC Curve
# -----------------------------
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve – XGBoost (Diabetes Risk Prediction)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

# ---------------------------------
# Define XGBoost model (best params)
# ---------------------------------
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("xgb", xgb_model)
])

# ---------------------------------
# Cross-validation
# ---------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(8, 6))

for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {i+1} (AUC = {roc_auc:.3f})")

# Plot mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr,
         color="blue",
         label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
         linewidth=2)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("5-Fold Cross-Validated ROC Curve – XGBoost")
plt.legend(loc="lower right")
plt.show()
import joblib

joblib.dump(pipeline, "xgboost_diabetes_model.pkl")
print("Model saved successfully.")

print(f"Mean ROC-AUC: {mean_auc:.3f}")

import shap
import pandas as pd

pipeline.fit(X, y)
# ✅ Use best params from GridSearch
best_params = {k.replace("xgb__", ""): v for k, v in grid.best_params_.items()}
print("Best params:", best_params)

xgb_final = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42,
    **best_params
)

final_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("xgb", xgb_final)
])

# Fit on full dataset
final_pipeline.fit(X, y)

# Save the CORRECT model
import joblib
joblib.dump(final_pipeline, "xgboost_diabetes_model.pkl")
print("Correct model saved!")

# Verify it works
import pandas as pd
test = pd.DataFrame([{
    "dm005": 72, "fm304s1": 1, "fm304s2": 1,
    "hb211": 0, "hb213": 0, "bm076": 115,
    "IDRS": 90, "crp": 20, "hb": 8,
    "bm017": 105, "bm018": 170, "hb101": 0,
}])
prob = final_pipeline.predict_proba(test)[0][1]
print(f"High risk test: {prob:.1%}")  # Should be well above 30%

model = pipeline.named_steps["xgb"]

X_transformed = pipeline.named_steps["preprocess"].transform(X)
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_transformed_df)

shap.summary_plot(shap_values, X_transformed_df)

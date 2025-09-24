# flake8: noqa: E501
import gzip
import pickle
import json
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix


# ------------------- Paso 1. Limpieza ------------------- #
train_df = pd.read_csv("files/input/train.csv")

# Renombrar y limpiar
train_df = train_df.rename(columns={"default payment next month": "default"})
train_df = train_df.drop(columns=["ID"])
train_df = train_df[train_df["EDUCATION"] != 0]
train_df = train_df[train_df["MARRIAGE"] != 0]
train_df["EDUCATION"] = train_df["EDUCATION"].replace({5: 4, 6: 4})

# ------------------- Paso 2. X / y ------------------- #
X = train_df.drop(columns=["default"])
y = train_df["default"]

# ------------------- Paso 3. Train / Test ------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ------------------- Paso 4. Preprocesamiento ------------------- #
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", MinMaxScaler(), numeric_cols)
    ]
)

# ------------------- Paso 5. Pipeline ------------------- #
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(f_classif)),
    ("classifier", LogisticRegression(random_state=42, max_iter=5000))
])

# ------------------- Paso 6. GridSearchCV ------------------- #
param_grid = {
    "selector__k": [1, 2, 3, 4, 5],
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["lbfgs", "liblinear"],
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ------------------- Paso 7. Guardar modelo ------------------- #
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)

# ------------------- Métricas ------------------- #
y_pred = grid_search.predict(X_test)

metrics = {
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

with open("files/output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

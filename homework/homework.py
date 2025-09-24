# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
#---------------------- Paso 1.---------------------------------------
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import json


train_df = pd.read_csv("files/input/train_data.csv.zip")
test_df = pd.read_csv("files/input/test_data.csv.zip")

#limpieza de datos
def clean_data(df):
    """Limpieza estructurada del dataset para predicción de default"""
    df_clean = df.copy()

    # Renombrar columna objetivo si existe
    if "default payment next month" in df_clean.columns:
        df_clean.rename(columns={"default payment next month": "default"}, inplace=True)

    # Eliminar columna ID si está presente
    if "ID" in df_clean.columns:
        df_clean.drop(columns=["ID"], inplace=True)

    # Reemplazar valores no disponibles por NaN
    df_clean["EDUCATION"] = df_clean["EDUCATION"].replace(0, np.nan)
    df_clean["MARRIAGE"] = df_clean["MARRIAGE"].replace(0, np.nan)

    # Agrupar niveles superiores de educación como "others"
    df_clean.loc[df_clean["EDUCATION"] > 4, "EDUCATION"] = 4
    df_clean["EDUCATION"] = df_clean["EDUCATION"].map({1: "1", 2: "2", 3: "3", 4: "others"})

    # Eliminar duplicados y registros incompletos
    df_clean.dropna(inplace=True)

    return df_clean

train_df = clean_data(train_df)
test_df = clean_data(test_df)

# --------------------Paso 2.-----------------------------------------
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]
x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]


#-------------------- Paso 3.-----------------------------------------
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numeric_features = [c for c in x_train.columns if c not in categorical_features]

# Definición de los pipelines para las variables categóricas y numéricas
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

# Definición del pipeline para las variables numéricas
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Combinación de ambos pipelines en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_pipeline, categorical_features),
        ("num", numeric_pipeline, numeric_features)
    ]
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(f_classif)),
    ("classifier", LogisticRegression(random_state=42, max_iter=5000))
])



# -------------------------Paso 4.----------------------------------------
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

param_grid = [
    {
        "selector__k": [1],
        "classifier__C": [100.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["liblinear"],
        "classifier__class_weight": [None]
    }
]

# Configurar validación cruzada estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipe, param_grid=param_grid,cv=cv, scoring="balanced_accuracy",n_jobs=-1,verbose=1,return_train_score=True)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

# Aplicar el mejor umbral encontrado
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)


# ------------------------------Paso 5.----------------------------------------
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

import pickle, gzip
with gzip.open("files/models/model.pkl.gz", "wb") as f:
   pickle.dump(grid_search, f)

#--------------------- Paso 6.------------------------------------
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
metrics = []

train_metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred, zero_division=0),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred, zero_division=0),
    'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
}
metrics.append(train_metrics)

test_metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred, zero_division=0),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
}
metrics.append(test_metrics)

# ------------------------------Paso 7.----------------------------------------
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Matriz de confusión para entrenamiento
cm_train = confusion_matrix(y_train, y_train_pred)
cm_train_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
    'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
}
metrics.append(cm_train_dict)

# Matriz de confusión para prueba
cm_test = confusion_matrix(y_test, y_test_pred)
cm_test_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
    'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
}
metrics.append(cm_test_dict)

# Guardar todo en una sola escritura
with open('files/output/metrics.json', 'w') as f:
    for m in metrics:
        f.write(json.dumps(m) + '\n')
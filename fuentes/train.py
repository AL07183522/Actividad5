"""
Script de Entrenamiento, GridSearch CV y Tracking con MLflow (train.py)
Materia: Gestión de Proyectos de Inteligencia Artificial
Actividad: 5
Alumno: Manuel de Jesus Ramiro Velázquez Tapia
Matrícula: AL07183522
Docente: Luis Ariel Vazquez Piña
"""

import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)

# Control para entorno local con o sin MLflow
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[WARN] MLflow no está instalado en este entorno. Las métricas se imprimirán localmente.")

# Importación del pipeline de preprocesamiento para autogestión de datos
from datos_prep import ejecutar_pipeline_preprocesamiento

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_LIMPIO = os.path.join(BASE_DIR, "datos", "datos_limp", "breast_cancer_clean.csv")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
)

def cargar_o_preparar_dataset() -> pd.DataFrame:
    """Garantiza la disponibilidad del dataset limpio antes del entrenamiento."""
    if not os.path.exists(PATH_LIMPIO):
        print(f"[WARN] No se encontró '{PATH_LIMPIO}'. Ejecutando pipeline de limpieza e importación...")
        ejecutar_pipeline_preprocesamiento()
    else:
        print(f"[INFO] Cargando dataset limpio existente desde: '{PATH_LIMPIO}'")
        
    df = pd.read_csv(PATH_LIMPIO)
    return df

def entrenar_y_registrar():
    # 1. Cargar o generar automáticamente el dataset procesado
    df = cargar_o_preparar_dataset()
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Partición estratificada 75% Entrenamiento / 25% Prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Breast_Cancer_Classification_Act5")
        print(f"[INFO] MLflow activo. Tracking URI: {MLFLOW_TRACKING_URI}")

    # ----------------------------------------------------
    # PIPELINE 1: REGRESIÓN LOGÍSTICA
    # ----------------------------------------------------
    param_grid_lr = {'C': [0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs']}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=cv, scoring='f1')
    grid_lr.fit(X_train, y_train)
    
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)[:, 1]
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    
    if HAS_MLFLOW:
        with mlflow.start_run(run_name="Logistic_Regression_GridSearch"):
            mlflow.set_tags({"model": "logistic_regression", "dataset": "breast_cancer"})
            mlflow.log_param("cv_folds", cv.n_splits)
            mlflow.log_params(grid_lr.best_params_)
            mlflow.log_metrics({
                "accuracy": acc_lr, 
                "precision": prec_lr, 
                "recall": rec_lr, 
                "f1_score": f1_lr, 
                "roc_auc": auc_lr
            })
            mlflow.sklearn.log_model(
                best_lr,
                name="model_logistic_regression",
                serialization_format="pickle"
            )
            
    print(f"[LR] Regresión Logística -> Accuracy: {acc_lr:.4f}, Precision: {prec_lr:.4f}, Recall: {rec_lr:.4f}, F1: {f1_lr:.4f}, ROC-AUC: {auc_lr:.4f}")

    # ----------------------------------------------------
    # PIPELINE 2: RANDOM FOREST CLASSIFIER
    # ----------------------------------------------------
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'random_state': [42]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=cv, scoring='f1')
    grid_rf.fit(X_train, y_train)
    
    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    
    if HAS_MLFLOW:
        with mlflow.start_run(run_name="Random_Forest_GridSearch"):
            mlflow.set_tags({"model": "random_forest", "dataset": "breast_cancer"})
            mlflow.log_param("cv_folds", cv.n_splits)
            mlflow.log_params(grid_rf.best_params_)
            mlflow.log_metrics({
                "accuracy": acc_rf, 
                "precision": prec_rf, 
                "recall": rec_rf, 
                "f1_score": f1_rf, 
                "roc_auc": auc_rf
            })
            mlflow.sklearn.log_model(
                best_rf,
                name="model_random_forest",
                serialization_format="pickle"
            )
            
    print(f"[RF] Random Forest Classifier -> Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}, ROC-AUC: {auc_rf:.4f}")

if __name__ == "__main__":
    entrenar_y_registrar()
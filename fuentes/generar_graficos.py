"""
Script para Generación de Gráficas y Evidencias Visuales (generar_graficas.py)
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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Detectar ruta raíz del repositorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "fuentes" else os.getcwd()

PATH_LIMPIO = os.path.join(REPO_ROOT, "datos", "datos_limp", "breast_cancer_clean.csv")
PATH_IMG = os.path.join(REPO_ROOT, "imagenes")

os.makedirs(PATH_IMG, exist_ok=True)

def generar_todas_las_graficas():
    print("=== GENERANDO GRÁFICAS Y EVIDENCIAS VISUALES EN 'imagenes/' ===")
    
    # Garantizar la disponibilidad de los datos limpios
    if not os.path.exists(PATH_LIMPIO):
        from datos_prep import ejecutar_pipeline_preprocesamiento
        ejecutar_pipeline_preprocesamiento()

    df = pd.read_csv(PATH_LIMPIO)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Gráfica de Distribución de Clases
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=y, hue=y, palette=['#e74c3c', '#2ecc71'], legend=False)
    plt.title('Distribución de Clases (0: Maligno, 1: Benigno)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Clase Diagnosticada')
    plt.ylabel('Cantidad de Muestras')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Maligno (0)', 'Benigno (1)'])
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_IMG, 'distribucion_clases.png'), dpi=300)
    plt.close()

    # 2. Mapa de Calor de Correlación (Top 10 Variables)
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    top_corr_cols = corr['target'].abs().sort_values(ascending=False).head(11).index
    sns.heatmap(df[top_corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Mapa de Calor - Top 10 Variables Correlacionadas con Target', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_IMG, 'matriz_correlacion.png'), dpi=300)
    plt.close()

    # Entrenar modelos para generar métricas visuales
    lr = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    # 3. Matriz de Confusión - Regresión Logística
    plt.figure(figsize=(5, 4))
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Maligno', 'Benigno'], yticklabels=['Maligno', 'Benigno'])
    plt.title('Matriz de Confusión - Regresión Logística', fontsize=11, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_IMG, 'matriz_confusion_lr.png'), dpi=300)
    plt.close()

    # 4. Matriz de Confusión - Random Forest
    plt.figure(figsize=(5, 4))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Maligno', 'Benigno'], yticklabels=['Maligno', 'Benigno'])
    plt.title('Matriz de Confusión - Random Forest', fontsize=11, fontweight='bold')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_IMG, 'matriz_confusion_rf.png'), dpi=300)
    plt.close()

    # 5. Curvas ROC Comparativas
    plt.figure(figsize=(7, 5))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.plot(fpr_lr, tpr_lr, color='#1f77b4', lw=2, label=f'Regresión Logística (AUC = {auc_lr:.4f})')
    plt.plot(fpr_rf, tpr_rf, color='#2ca02c', lw=2, linestyle='--', label=f'Random Forest (AUC = {auc_rf:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle=':', lw=1.5)
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)')
    plt.title('Curvas ROC Comparativas', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_IMG, 'curva_roc_comparativa.png'), dpi=300)
    plt.close()

    # 6. Importancia de Características - Random Forest
    plt.figure(figsize=(8, 5))
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='crest', legend=False)
    plt.title('Top 10 Importancia de Características (Random Forest)', fontsize=11, fontweight='bold')
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_IMG, 'feature_importance.png'), dpi=300)
    plt.close()

    print("=== TODAS LAS GRÁFICAS SE HAN GENERADO Y GUARDADO EN 'imagenes/' ===")

if __name__ == "__main__":
    generar_todas_las_graficas()
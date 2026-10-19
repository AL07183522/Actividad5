# Actividad 5: Gestión de Datos, Entrenamiento y Tracking MLOps con MLflow Local

Materia: Gestión de Proyectos de Inteligencia Artificial  
Actividad: 5  
Alumno: Manuel de Jesus Ramiro Velázquez Tapia  
Matrícula: AL07183522  
Docente: Luis Ariel Vazquez Piña  

---

## 📌 1. Descripción General del Proyecto

Este repositorio contiene la solución técnica integral para la **Actividad 5**, enfocado en el desarrollo de un pipeline MLOps completo de aprendizaje automático supervisado en entorno local. El proyecto abarca desde la ingesta automática y preprocesamiento de datos hasta el entrenamiento de modelos clasificadores, optimización de hiperparámetros mediante `GridSearchCV` con validación cruzada estratificada (`5-Fold Stratified K-Fold`) y el seguimiento sistemático de experimentos con un servidor local de **MLflow** respaldado por **SQLite** y almacenamiento local de artefactos.

---

## 📊 2. Información del Dataset (Kaggle)

El conjunto de datos utilizado es el **Breast Cancer Wisconsin (Diagnostic) Dataset**, alojado en Kaggle (`abdelazizsami/breast-cancer-wisconsin-diagnostic`) y derivado del repositorio UCI Machine Learning Repository:

* **Dataset en Kaggle:** Breast Cancer Wisconsin Diagnostic (`abdelazizsami/breast-cancer-wisconsin-diagnostic`)
* **Tipo de Problema:** Clasificación Binaria Supervisada (`target`: `0` = Maligno, `1` = Benigno).
* **Volumen de Datos:** 569 observaciones clínicas únicas derivadas de biopsias por aspiración con aguja fina (PAF) de masas mamarias.
* **Atributos:** 30 características numéricas continuas basadas en la morfología nuclear.
* **Distribución de Clases:** 357 casos Benignos (62.7%) y 212 casos Malignos (37.3%).

---

## 📁 3. Estructura Completa del Repositorio (`/Users/manu/Documents/Actividad5`)
```
Actividad5/
├── .venv/                   # Entorno virtual de Python
├── datos/
│   ├── datos_ini/           # Dataset crudo original con duplicados (breast_cancer_raw.csv)
│   └── datos_limp/          # Dataset limpio y estandarizado Z-Score (breast_cancer_clean.csv)
├── fuentes/
│   ├── datos_prep.py        # Ingesta, limpieza, imputación por mediana y estandarización Z-Score
│   ├── generar_graficas.py  # Generación automática de visualizaciones EDA y métricas
│   ├── train.py             # Entrenamiento, GridSearchCV, 5-Fold Stratified CV y MLflow Tracking
│   └── entrena.ipynb        # Notebook interactivo para pruebas y Google Colab
├── imagenes/
│   ├── distribucion_clases.png
│   ├── matriz_correlacion.png
│   ├── matriz_confusion_lr.png
│   ├── matriz_confusion_rf.png
│   ├── curva_roc_comparativa.png
│   ├── feature_importance.png
│   └── evidenciaMLFLOW/     # Evidencias visuales de la interfaz y ejecuciones en MLflow
│       ├── dashboard_runs.png
│       ├── metric_comparison.png
│       └── artifacts_logged.png
├── mlartifacts/             # Almacenamiento local de artefactos de MLflow (modelos y gráficos)
├── mlflow.db                # Base de datos SQLite para el backend de MLflow
├── README.md                # Guía de reproducción local y arquitectura
└── CHANGELOG.md             # Reporte técnico final y registro de versiones (T11/T12)
```
---

## 🚀 4. Guía de Ejecución Local Paso a Paso

### Paso 1: Configurar el Entorno Virtual e Instalar Dependencias
cd /Users/manu/Documents/Actividad5

python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn mlflow

---

### Paso 2: Iniciar el Servidor Local de MLflow (Terminal 1)
.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /Users/manu/Documents/Actividad5/mlartifacts --host 127.0.0.1 --port 5000

---

### Paso 3: Ejecutar los Pipelines y Registrar Experimentos (Terminal 2)
cd /Users/manu/Documents/Actividad5
source .venv/bin/activate

python fuentes/datos_prep.py
python fuentes/generar_graficas.py
python fuentes/train.py

---

## 📈 5. Decisiones de Diseño y Resumen de Resultados

* **Estandarización Z-Score:** Aplicación de `StandardScaler` para evitar sesgos de escala.
* **Validación Cruzada Estratificada:** `StratifiedKFold(n_splits=5)` para preservar la proporción de clases.
* **Separación de Responsabilidades:** Scripts modularizados para orquestación automatizada.

| Modelo | Hiperparámetros Optimizados | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | `C=10.0`, `penalty='l2'`, `solver='lbfgs'` | **98.60%** | **98.89%** | **98.89%** | **0.9889** | **0.9977** |
| **Random Forest** | `n_estimators=100`, `max_depth=5` | **95.80%** | **95.65%** | **97.78%** | **0.9670** | **0.9950** |

---

## 📷 6. Evidencias de MLflow en el Repositorio

Las capturas de pantalla que respaldan la ejecución del servidor local y el registro de experimentos se encuentran disponibles en la carpeta:  
👉 `imagenes/evidenciaMLFLOW/` dentro del repositorio en GitHub (`https://github.com/AL07183522/Actividad5/tree/main/imagenes/evidenciaMLFLOW`).

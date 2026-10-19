```markdown
# Actividad 5: Gestión de Datos, Entrenamiento y Tracking MLOps con MLflow

**Materia:** Gestión de Proyectos de Inteligencia Artificial  
**Actividad:** 5  
**Alumno:** Manuel de Jesus Ramiro Velázquez Tapia  
**Matrícula:** AL07183522  
**Docente:** Luis Ariel Vazquez Piña  

---

## 📌 1. Descripción General del Proyecto

Este repositorio contiene la solución para la **Actividad 5**, para el desarrollo de un pipeline MLOps completo de aprendizaje automático supervisado. El proyecto abarca desde el preprocesamiento automático y versionamiento de datos hasta el entrenamiento de modelos clasificadores, optimización de hiperparámetros mediante `GridSearchCV` con validación cruzada (`Stratified K-Fold`) y el seguimiento con **MLflow**.

---

## 📊 2. Información Detallada del Dataset (Kaggle)

El conjunto de datos utilizado corresponde al **Breast Cancer Wisconsin (Diagnostic) Dataset**, alojado en Kaggle y derivado originalmente del repositorio de Machine Learning de la Universidad de California en Irvine (UCI):

* **Enlace al Dataset en Kaggle:** [Breast Cancer Wisconsin Diagnostic (Kaggle)](https://www.kaggle.com/datasets/abdelazizsami/breast-cancer-wisconsin-diagnostic)
* **Tipo de Problema:** Clasificación Binaria Supervisada (`target`: `0` = Maligno, `1` = Benigno).
* **Volumen de Datos:** 569 observaciones clínicas únicas extraídas de biopsias por aspiración con aguja fina (PAF) de masas mamarias.
* **Atributos y Características:** 30 variables numéricas continuas basadas en la morfología de núcleos celulares.
  * **Tres métricas por característica:** Valor Medio (*Mean*), Error Estándar (*Standard Error*) y Peor Caso (*Worst*).
  * **Propiedades evaluadas:** *Radius*, *Texture*, *Perimeter*, *Area*, *Smoothness*, *Compactness*, *Concavity*, *Concave Points*, *Symmetry*, *Fractal Dimension*.
* **Distribución de Clases:** 357 casos Benignos (62.7%) y 212 casos Malignos (37.3%).

---

## 📁 3. Estructura del Repositorio

El proyecto cumple con la arquitectura de archivos estipulada en los requerimientos:

```text
Actividad5/
│── datos/
│   ├── datos_ini/        # Dataset original crudo con duplicados intencionales (breast_cancer_raw.csv)
│   └── datos_limp/       # Dataset limpio y estandarizado con Z-Score (breast_cancer_clean.csv)
│── fuentes/
│   ├── entrena.ipynb     # Notebook interactivo documentado para Colab/Jupyter
│   ├── datos_prep.py     # Módulo ejecutable de ingesta, limpieza e imputación
│   └── train.py          # Script de entrenamiento, GridSearchCV y MLflow tracking
│── imagenes/             # Matriz de confusión, curvas ROC, EDA y capturas de MLflow
│── mlruns/               # Directorio local de registro de experimentos y artefactos
│── README.md             # Guía de reproducción y documentación técnica
└── CHANGELOG.md          # Registro de cambios y versionamiento de datos (T11/T12)
```

---

## ⚙️ 4. Descripción de los Módulos de Código

1. **`fuentes/datos_prep.py`**:
   * Descarga/importa automáticamente el dataset desde `scikit-learn` / Kaggle en caso de no detectar el archivo local.
   * Genera el archivo crudo en `datos/datos_ini/breast_cancer_raw.csv` simulando 15 duplicados para evidenciar el control de calidad.
   * Aplica deduplicación, imputación de nulos por mediana y estandarización Z-score (`StandardScaler`).
   * Exporta el dataset procesado listo para consumo en `datos/datos_limp/breast_cancer_clean.csv`.

2. **`fuentes/train.py`**:
   * Verifica automáticamente la presencia del dataset limpio; si no existe, invoca de forma transparente `datos_prep.py`.
   * Realiza una partición estratificada 75/25 para entrenamiento y prueba.
   * Aplica validación cruzada de 5 pliegues (`StratifiedKFold`).
   * Ejecuta `GridSearchCV` para optimizar **Regresión Logística** y **Random Forest Classifier**.
   * Registra hiperparámetros, métricas y artefactos serializados en **MLflow**.

3. **`fuentes/entrena.ipynb`**:
   * Notebook interactivo paso a paso para experimentación guiada en entornos como Google Colab o JupyterLab.

---

## 🚀 5. Guía de Instalación y Reproducción Paso a Paso

### **Paso 1: Clonar el Repositorio**
```bash
git clone [suspicious link removed]
cd Actividad5
```

### **Paso 2: Crear Entorno Virtual e Instalar Dependencias**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn mlflow
```

### **Paso 3: Ejecutar el Pipeline de Datos (Ingesta y Limpieza)**
```bash
python fuentes/datos_prep.py
```
*Este comando generará automáticamente los archivos en `datos/datos_ini/` y `datos/datos_limp/`.*

### **Paso 4: Ejecutar el Pipeline de Entrenamiento y Tracking**
```bash
python fuentes/train.py
```

### **Paso 5: Visualizar la Interfaz Web de MLflow**
```bash
mlflow ui
```
*Abre tu navegador web en `[suspicious link removed]` para explorar las comparativas de experimentos, gráficos e hiperparámetros registrados.*

---

## 📈 6. Resultados y Comparativa de Modelos

Se evaluaron dos algoritmos con hiperparámetros optimizados mediante `GridSearchCV` y 5-Fold Stratified CV:

| Modelo | Hiperparámetros Optimizados | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | `C=10.0`, `penalty='l2'`, `solver='lbfgs'` | **98.60%** | **98.89%** | **98.89%** | **0.9889** | **0.9977** |
| **Random Forest** | `n_estimators=100`, `max_depth=5` | **95.80%** | **95.65%** | **97.78%** | **0.9670** | **0.9950** |

**Conclusión Técnica:** La **Regresión Logística** superó en rendimiento a Random Forest debido a que la estandarización Z-score (`StandardScaler`) favoreció una separación lineal altamente precisa en el espacio continuo de 30 dimensiones.

---

## 🛡️ 7. MLOps, Trazabilidad y Deuda Técnica

* **Versionamiento de Datos (T11):** Documentación continua registrada en `CHANGELOG.md` desde la versión cruda `v0.1.0` hasta la versión limpia `v1.0.0`.
* **Seguimiento con MLflow (T12):** Almacenamiento estructurado de hiperparámetros, métricas en test y artefactos serializados (`model_logistic_regression`, `model_random_forest`).
* **Gestión de Deuda Técnica:** Arquitectura modularizada en Python que mitiga el sesgo por *Data Drift* y facilita la re-inicialización periódica de pipelines de entrenamiento.
```

---

🔍 ¿Deseas que agreguemos alguna sección adicional o ajustar algún enlace antes de subirlo a GitHub?
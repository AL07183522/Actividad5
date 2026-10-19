# Reporte Técnico Final y Registro de Cambios (CHANGELOG.md)

Materia: Gestión de Proyectos de Inteligencia Artificial  
Actividad: 5  
Alumno(a): Manuel de Jesus Ramiro Velázquez Tapia  
Matrícula: AL07183522  
Docente: Luis Ariel Vazquez Piña  

---

## 📋 1. Descripción del Dataset y Fuente de Datos

### 1.1 Fuente y Origen
El proyecto utiliza el **Breast Cancer Wisconsin (Diagnostic) Dataset**, alojado en Kaggle (`abdelazizsami/breast-cancer-wisconsin-diagnostic`) y proveniente del repositorio UCI Machine Learning Repository.

* **Tipo de Problema:** Clasificación Binaria Supervisada (`target`: `0` = Maligno, `1` = Benigno).
* **Volumen:** 569 registros clínicos únicos extraídos de biopsias por aspiración con aguja fina (PAF) de masas mamarias.
* **Distribución de Clases:** 357 observaciones de tumores Benignos (62.7%) y 212 observaciones de tumores Malignos (37.3%).

### 1.2 Diccionario de Datos Principales
El dataset cuenta con 30 características numéricas continuas correspondientes al valor Medio (*mean*), Error Estándar (*error*) y Peor Caso (*worst*) de 10 propiedades morfológicas celulares (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal).

---

## 🧹 2. Proceso de Limpieza, Estandarización y Reglas de Negocio

1. **Detección y Eliminación de Duplicados:** En la versión cruda (`datos/datos_ini/breast_cancer_raw.csv`), se simularon 15 filas duplicadas. El módulo `datos_prep.py` eliminó estas 15 filas, dejando 569 observaciones únicas.
2. **Imputación de Valores Faltantes:** Imputación por mediana en variables numéricas continuas.
3. **Estandarización (Z-Score):** Se aplicó `StandardScaler` (\\(z = \frac{x - \mu}{\sigma}\\)) para nivelar las escalas heterogéneas.

---

## 🗂️ 3. Estrategia de Versionamiento de Datos y Código (T11)

### [v0.1.0] - Versión Cruda (`datos/datos_ini/`)
* **Archivo:** `breast_cancer_raw.csv` (584 registros con duplicados intencionales para prueba de filtros).

### [v1.0.0] - Versión Procesada (`datos/datos_limp/`)
* **Archivo:** `breast_cancer_clean.csv` (569 registros únicos, 30 variables estandarizadas Z-Score).

---

## 📊 4. Visualizaciones Relevantes y Análisis Exploratorio (EDA)

1. **Distribución de Clases (`imagenes/distribucion_clases.png`):** Muestra el balance leve de clases (62.7% vs 37.3%), justificando el uso de `StratifiedKFold` y **F1-Score / ROC-AUC**.
2. **Mapa de Calor de Correlación (`imagenes/matriz_correlacion.png`):** Muestra alta correlación negativa con la variable objetivo en `concave points_worst` (-0.79) y `perimeter_worst` (-0.78).

---

## 🤖 5. Descripción de Modelos, Hiperparámetros y Métricas (T12)

### 5.1 Estrategia de Validación
Partición estratificada 75/25 y **5-Fold Stratified Cross-Validation** en `GridSearchCV`.

### 5.2 Hiperparámetros Optimizados
* **Regresión Logística:** \\(C = 10.0\\), penalización \\(L2\\), solver `lbfgs`.
* **Random Forest Classifier:** `n_estimators = 100`, `max_depth = 5`.

### 5.3 Cuadro Comparativo de Métricas

| Modelo | Hiperparámetros Óptimos | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | `C=10.0`, `penalty='l2'` | **98.60%** | **98.89%** | **98.89%** | **0.9889** | **0.9977** |
| **Random Forest** | `n_estimators=100`, `max_depth=5` | **95.80%** | **95.65%** | **97.78%** | **0.9670** | **0.9950** |

---

## 🛠️ 6. MLOps Local con MLflow: Ventajas, Facilidad de Configuración y Evidencias

### 6.1 Ventajas de Usar la Librería MLflow en Local
1. **Facilidad e Inmediatez de Configuración:** Basta con ejecutar `pip install mlflow` e iniciar el servidor local mediante `.venv/bin/mlflow server`. No requiere cuentas en la nube ni configuraciones complejas de red.
2. **Persistencia Local Transparente:** El backend store en **SQLite** (`sqlite:///mlflow.db`) y la carpeta de artefactos (`mlartifacts/`) aseguran la conservación privativa y local de todas las corridas.
3. **Registro Automático y Desacoplado:** Mediante `mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)` en `train.py`, se registran hiperparámetros, métricas en test y modelos `.pkl` de forma transparente.
4. **Interfaz Web Interactiva:** Permite explorar visualmente el historial de ejecuciones, comparar métricas e inspeccionar artefactos en el puerto 5000 (`[suspicious link removed]`).

### 6.2 Evidencias Visuales de MLflow en el Repositorio
Las capturas de pantalla de la interfaz web y el servidor local de MLflow están alojadas en la carpeta dedicada del repositorio:  
👉 **Ruta de Evidencias en GitHub:** `https://github.com/AL07183522/Actividad5/tree/main/imagenes/evidenciaMLFLOW`

* **`dashboard_runs.png`:** Muestra la vista general del servidor MLflow con las corridas del experimento `"Breast_Cancer_Classification_Act5"`.
* **`metric_comparison.png`:** Muestra la comparación gráfica de F1-Score, ROC-AUC y Accuracy entre Regresión Logística y Random Forest.
* **`artifacts_logged.png`:** Muestra los artefactos guardados en `mlartifacts/`, incluyendo los binarios `.pkl` y las matrices de confusión.

---

## 💡 7. Reflexión sobre Deuda Técnica y Filosofía MLOps

1. **Gestión de Data Drift & Concept Drift:** Implementación de pruebas Kolmogorov-Smirnov sobre datos entrantes.
2. **Automatización de CI/CD:** Desacoplamiento modular entre `datos_prep.py` y `train.py` listo para integración continua con GitHub Actions.
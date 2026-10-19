# Reporte Técnico Final y Registro de Cambios (CHANGELOG.md)

**Materia:** Gestión de Proyectos de Inteligencia Artificial  
**Actividad:** 5  
**Alumno:** Manuel de Jesus Ramiro Velázquez Tapia  
**Matrícula:** AL07183522  
**Docente:** Luis Ariel Vazquez Piña  

---

## 📋 1. Descripción del Dataset y Fuente de Datos

### 1.1 Fuente y Origen
El proyecto utiliza el **Breast Cancer Wisconsin (Diagnostic) Dataset**, alojado en Kaggle y proveniente del repositorio UCI Machine Learning Repository.

* **Tipo de Problema:** Clasificación Binaria Supervisada (`target`: `0` = Maligno, `1` = Benigno).
* **Volumen:** 569 registros clínicos únicos extraídos de biopsias por aspiración con aguja fina (PAF) de masas mamarias.
* **Distribución de Clases:** 357 observaciones de tumores Benignos (62.7%) y 212 observaciones de tumores Malignos (37.3%).

### 1.2 Diccionario de Datos Principales
El dataset cuenta con 30 características numéricas continuas correspondientes al valor Medio (*mean*), Error Estándar (*error*) y Peor Caso (*worst*) de las siguientes 10 propiedades morfológicas de los núcleos celulares:
1. `radius`: Media de las distancias desde el centro a los puntos del contorno.
2. `texture`: Desviación estándar de los valores de la escala de grises.
3. `perimeter`: Tamaño del perímetro del núcleo tumoral.
4. `area`: Área total del núcleo celular.
5. `smoothness`: Variación local en las longitudes de los radios.
6. `compactness`: Calculado como \\((\text{perímetro}^2 / \text{área} - 1.0)\\).
7. `concavity`: Severidad de las porciones cóncavas del contorno.
8. `concave points`: Número de porciones cóncavas del contorno.
9. `symmetry`: Simetría de la estructura nuclear.
10. `fractal dimension`: Aproximación de línea de costa (aproximación fractal \\(- 1\\)).

---

## 🧹 2. Proceso de Limpieza, Estandarización y Reglas de Negocio

### 2.1 Ingesta y Calidad de Datos
1. **Detección y Eliminación de Duplicados:** En la versión cruda (`datos/datos_ini/breast_cancer_raw.csv`), se simularon 15 filas duplicadas para validar los controles de calidad. El módulo `datos_prep.py` eliminó exitosamente estas 15 filas, dejando 569 observaciones únicas.
2. **Imputación de Valores Faltantes:** Se verificó la presencia de valores nulos e infinitos, aplicando imputación por mediana en variables numéricas continuas para preservar la solidez estadística.
3. **Estandarización (\\(Z\\)-Score):** Dado que variables como `area_worst` manejan valores superiores a 2000 mientras que `smoothness` maneja valores < 0.1, se aplicó `StandardScaler`:
   \\[z = \frac{x - \mu}{\sigma}\\]
   Esto garantiza que los modelos basados en gradientes y distancias no sufran sesgos por escala.

---

## 🗂️ 3. Estrategia de Versionamiento de Datos y Código (T11)

### [v0.1.0] - Versión Cruda (`datos/datos_ini/`)
* **Archivo:** `breast_cancer_raw.csv`
* **Estado:** 584 registros (569 reales + 15 duplicados intencionales para prueba de filtros).
* **Propósito:** Registrar el estado inicial no filtrado para auditoría de calidad de datos.

### [v1.0.0] - Versión Procesada (`datos/datos_limp/`)
* **Archivo:** `breast_cancer_clean.csv`
* **Estado:** 569 registros únicos, 30 variables estandarizadas \\(Z\\)-score y variable objetivo `target`.
* **Propósito:** Servir como única fuente de verdad para los pipelines de entrenamiento y validación cruzada.

---

## 📊 4. Visualizaciones Relevantes y Análisis Exploratorio (EDA)

1. **Distribución de Clases (`imagenes/distribucion_clases.png`):** Confirms un desbalance leve (62.7% benigno vs 37.3% maligno), lo que justifica el uso de `StratifiedKFold` y métricas ponderadas como **F1-Score** y **ROC-AUC** en lugar de depender únicamente del Accuracy.
2. **Mapa de Calor de Correlación (`imagenes/matriz_correlacion.png`):** Muestra alta correlación lineal negativa entre la variable objetivo `target` y variables como `concave points_worst` (\\(-0.79\\)), `perimeter_worst` (\\(-0.78\\)) y `concave points_mean` (\\(-0.78\\)), señalando que tumores malignos presentan mayores irregularidades de contorno.

---

## 🤖 5. Descripción de Modelos, Hiperparámetros y Métricas (T12)

### 5.1 Estrategia de Validación
Se empleó una partición estratificada 75% Entrenamiento / 25% Prueba (143 muestras de test) junto con **5-Fold Stratified Cross-Validation** en `GridSearchCV`.

### 5.2 Hiperparámetros Optimizados
* **Regresión Logística:** Espacio de búsqueda en \\(C \in \{0.1, 1.0, 10.0\}\\), penalización \\(L2\\), solver `lbfgs`. Parámetro óptimo seleccionado: \\(C = 10.0\\).
* **Random Forest Classifier:** Espacio de búsqueda en `n_estimators` \\(\in \{50, 100, 200\}\\), `max_depth` \\(\in \{3, 5, 10\}\\). Parámetros óptimos seleccionados: `n_estimators = 100`, `max_depth = 5`.

### 5.3 Cuadro Comparativo de Métricas

| Modelo | Hiperparámetros Óptimos | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | `C=10.0`, `penalty='l2'` | **98.60%** | **98.89%** | **98.89%** | **0.9889** | **0.9977** |
| **Random Forest** | `n_estimators=100`, `max_depth=5` | **95.80%** | **95.65%** | **97.78%** | **0.9670** | **0.9950** |

---

## 🔬 6. Comparación Estadística e Interpretación Técnica

1. **Interpretación de Matrices de Confusión:**
   * **Regresión Logística (`imagenes/matriz_confusion_lr.png`):** De 143 casos de prueba, solo cometió 2 errores (1 Falso Positivo y 1 Falso Negativo), alcanzando un **Recall de 98.89%**, crucial en diagnóstico médico para evitar falsos negativos.
   * **Random Forest (`imagenes/matriz_confusion_rf.png`):** Registró 6 errores (4 Falsos Positivos y 2 Falsos Negativos), mostrando menor sensibilidad en fronteras de decisión densas.

2. **Curva ROC y AUC (`imagenes/curva_roc_comparativa.png`):** Ambos modelos demostraron excelente capacidad de discriminación, con un AUC de **0.9977** para Regresión Logística frente a **0.9950** para Random Forest.

---

## 🛠️ 7. Evidencia del Uso de MLflow y Trazabilidad

Se configuró el seguimiento automático en MLflow bajo el experimento `"Breast_Cancer_Classification_Act5"`.
* **Parámetros Registrados:** `C`, `penalty`, `solver`, `n_estimators`, `max_depth`.
* **Métricas Registradas:** `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`.
* **Artefactos Guardados:** Serialización del modelo ajustado (`model_logistic_regression/` y `model_random_forest/`).

---

## 💡 8. Reflexión sobre Deuda Técnica y Filosofía MLOps

1. **Gestión de Data Drift & Concept Drift:** En producción médica, la evolución de equipos de mamografía puede cambiar las distribuciones de entrada. Se propone integrar monitoreos periódicos de la prueba Kolmogorov-Smirnov sobre datos entrantes.
2. **Automatización de CI/CD para ML:** La separación modular realizada entre `datos_prep.py` y `train.py` permite orquestar ejecuciones automáticas en GitHub Actions o Apache Airflow ante cada actualización del dataset en `datos/datos_ini/`.

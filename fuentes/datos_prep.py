"""
Módulo de Preparación y Limpieza de Datos (datos_prep.py)
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
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Configuración de rutas relativas dinámicas para ejecución local fluida
# Funciona ya sea que ejecutes 'python fuentes/datos_prep.py' desde la raíz o dentro de 'fuentes/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "fuentes" else os.getcwd()

PATH_RAW = os.path.join(REPO_ROOT, "datos", "datos_ini", "breast_cancer_raw.csv")
PATH_LIMPIO = os.path.join(REPO_ROOT, "datos", "datos_limp", "breast_cancer_clean.csv")

def descargar_o_cargar_datos_raw(ruta_raw: str = PATH_RAW) -> pd.DataFrame:
    """
    Importa/descarga el dataset Breast Cancer desde scikit-learn y lo guarda
    en la ruta inicial (datos/datos_ini/breast_cancer_raw.csv) si no existe.
    Incluye filas duplicadas intencionales para validar el filtro de limpieza (T11).
    """
    os.makedirs(os.path.dirname(ruta_raw), exist_ok=True)
    
    if not os.path.exists(ruta_raw):
        print(f"[INFO] No se encontró el dataset en local. Importando desde scikit-learn...")
        cancer = load_breast_cancer()
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target
        
        # Simulación de 15 filas duplicadas intencionales para evidencias de limpieza (T11)
        df_duplicados = df.iloc[:15].copy()
        df = pd.concat([df, df_duplicados], axis=0).reset_index(drop=True)
        
        df.to_csv(ruta_raw, index=False)
        print(f"[INFO] Dataset crudo generado localmente en: {ruta_raw} (Shape: {df.shape})")
    else:
        print(f"[INFO] Cargando dataset crudo existente desde: {ruta_raw}")
        df = pd.read_csv(ruta_raw)
        
    return df

def limpiar_y_preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline de limpieza:
    1. Eliminación de filas duplicadas.
    2. Imputación de valores faltantes (mediana).
    3. Normalización / Escalado Z-Score con StandardScaler.
    """
    df_clean = df.copy()

    # 1. Eliminación de duplicados
    filas_iniciales = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    duplicados_removidos = filas_iniciales - len(df_clean)
    print(f"[INFO] Se eliminaron {duplicados_removidos} filas duplicadas (Filas restantes: {len(df_clean)}).")

    # 2. Manejo de valores nulos (imputación por mediana en variables numéricas)
    columnas_num = [col for col in df_clean.columns if col != 'target']
    for col in columnas_num:
        if df_clean[col].isnull().sum() > 0:
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)
            print(f"[INFO] Imputados nulos en '{col}' con mediana = {mediana:.4f}")

    # 3. Estandarización Z-Score de características
    scaler = StandardScaler()
    df_clean[columnas_num] = scaler.fit_transform(df_clean[columnas_num])
    print("[INFO] Características estandarizadas exitosamente con StandardScaler (Z-Score).")

    return df_clean

def guardar_datos_limpios(df: pd.DataFrame, ruta_limpios: str = PATH_LIMPIO):
    """Guarda el DataFrame procesado en la ubicación de destino para consumo de los pipelines."""
    os.makedirs(os.path.dirname(ruta_limpios), exist_ok=True)
    df.to_csv(ruta_limpios, index=False)
    print(f"[INFO] Dataset limpio disponible para pipelines en: {ruta_limpios} (Shape: {df.shape})")

def ejecutar_pipeline_preprocesamiento(ruta_raw: str = PATH_RAW, ruta_limpios: str = PATH_LIMPIO) -> pd.DataFrame:
    """Ejecuta el flujo completo de importación, limpieza y guardado."""
    print("=== INICIANDO PIPELINE DE PREPROCESAMIENTO DE DATOS ===")
    df_raw = descargar_o_cargar_datos_raw(ruta_raw)
    df_clean = limpiar_y_preprocesar(df_raw)
    guardar_datos_limpios(df_clean, ruta_limpios)
    print("=== PIPELINE DE PREPROCESAMIENTO FINALIZADO ===")
    return df_clean

if __name__ == "__main__":
    ejecutar_pipeline_preprocesamiento()
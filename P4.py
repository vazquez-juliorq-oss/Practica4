"""
Pipeline de Generación de Datos Sintéticos y División Estratificada (Train-Test).

Este script utiliza la librería Faker para simular un dataset de clientes 
con localización en México ('es_MX'). Posteriormente, aplica codificación 
One-Hot a variables categóricas y realiza una partición estratificada del 
dataset (80% entrenamiento / 20% prueba) conservando la proporción de la 
clase objetivo (Churn).
"""

from faker import Faker
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ejecutar_pipeline_faker():
    """Genera datos sintéticos, procesa variables categóricas y ejecuta división estratificada.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, tabla_proporciones)
    """
    # -------------------------------------------------------------------------
    # 1. CONFIGURACIÓN Y GENERACIÓN DE DATOS SINTÉTICOS
    # -------------------------------------------------------------------------
    fake = Faker("es_MX")
    Faker.seed(42)
    np.random.seed(42)

    num_registros = 1000
    registros = []

    for i in range(num_registros):
        registros.append({
            "id_cliente": i + 1,
            "nombre": fake.name(),
            "email": fake.email(),
            "telefono": fake.phone_number(),
            "fecha_nacimiento": fake.date_of_birth(
                minimum_age=18, maximum_age=75
            ),
            "estado_civil": np.random.choice(
                ["Soltero", "Casado", "Divorciado"], p=[0.5, 0.4, 0.1]
            ),
            "ingreso_anual": round(np.random.uniform(100000, 1000000), 2),
            "churn": np.random.choice([0, 1], p=[0.8, 0.2]),
        })

    df_sintetico = pd.DataFrame(registros)

    print("=== 1. MUESTRA DEL DATASET SINTÉTICO GENERADO (Faker es_MX) ===")
    print(df_sintetico.head(5))
    print(f"\nTotal Registros Generados: {len(df_sintetico)}")
    print("\n")

    # -------------------------------------------------------------------------
    # 2. PREPROCESAMIENTO Y CODIFICACIÓN ONE-HOT
    # -------------------------------------------------------------------------
    df_encoded = pd.get_dummies(
        df_sintetico, columns=["estado_civil"], drop_first=False
    )

    # Separación de características (X) y variable objetivo (y)
    X = df_encoded.drop(
        columns=[
            "id_cliente",
            "nombre",
            "email",
            "telefono",
            "fecha_nacimiento",
            "churn",
        ]
    )
    y = df_encoded["churn"]

    # -------------------------------------------------------------------------
    # 3. DIVISIÓN ESTRATIFICADA TRAIN / TEST
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("=== 2. DIMENSIONES DE LOS CONJUNTOS DE DATOS ===")
    print(f"Dataset Completo X: {X.shape} | y: {y.shape}")
    print(
        f"Conjunto de Entrenamiento (Train) X_train: {X_train.shape} | y_train:"
        f" {y_train.shape}"
    )
    print(
        f"Conjunto de Prueba (Test) X_test: {X_test.shape} | y_test:"
        f" {y_test.shape}"
    )
    print("\n")

    # -------------------------------------------------------------------------
    # 4. VALIDACIÓN DE ESTRATIFICACIÓN
    # -------------------------------------------------------------------------
    tabla_proporciones = pd.DataFrame({
        "Conjunto": [
            "Dataset Completo",
            "Entrenamiento (Train 80%)",
            "Prueba (Test 20%)",
        ],
        "Total Registros": [len(y), len(y_train), len(y_test)],
        "Casos Churn = 1": [y.sum(), y_train.sum(), y_test.sum()],
        "Casos No Churn = 0": [
            len(y) - y.sum(),
            len(y_train) - y_train.sum(),
            len(y_test) - y_test.sum(),
        ],
        "Tasa de Churn (Proporción)": [
            f"{y.mean():.4f} ({y.mean()*100:.2f}%)",
            f"{y_train.mean():.4f} ({y_train.mean()*100:.2f}%)",
            f"{y_test.mean():.4f} ({y_test.mean()*100:.2f}%)",
        ],
    })

    print("=== 3. TABLA COMPARATIVA DE ESTRATIFICACIÓN DE LA CLASE CHURN ===")
    print(tabla_proporciones.to_string(index=False))

    return X_train, X_test, y_train, y_test, tabla_proporciones


if __name__ == "__main__":
    ejecutar_pipeline_faker()

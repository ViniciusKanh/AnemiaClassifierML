import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(csv_path):
    """
    Lê os dados, remove duplicatas e valores ausentes, codifica o target e retorna X, y e o codificador.
    """
    df = pd.read_csv(csv_path)

    # Limpeza básica
    df = df.drop_duplicates().dropna()

    # Separação de features e target
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    # Codificação do target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le

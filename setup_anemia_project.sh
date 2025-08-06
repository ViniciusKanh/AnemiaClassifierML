#!/bin/bash

# Nome do projeto
PROJECT_NAME="AnemiaClassifierML"

echo "🔧 Criando estrutura de pastas para o projeto $PROJECT_NAME..."

# Criar pastas
mkdir -p $PROJECT_NAME/data
mkdir -p $PROJECT_NAME/notebooks
mkdir -p $PROJECT_NAME/models
mkdir -p $PROJECT_NAME/scripts
mkdir -p $PROJECT_NAME/utils

echo "📁 Estrutura criada."

# ===============================
# Arquivo: scripts/preprocessing.py
# ===============================
cat <<EOF > $PROJECT_NAME/scripts/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Remover duplicatas e valores nulos
    df = df.drop_duplicates().dropna()

    # Separar variáveis
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    # Codificar variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Balancear com SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y_encoded)

    return X_res, y_res, le
EOF

# ===============================
# Arquivo: scripts/train_model.py
# ===============================
cat <<EOF > $PROJECT_NAME/scripts/train_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scripts.preprocessing import preprocess_data

# Caminho do dataset
csv_path = "data/AnemiaTypesClassification_data.csv"

# Pré-processamento
X, y, label_encoder = preprocess_data(csv_path)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Salvar modelo
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("✅ Modelo treinado e salvo em models/")
EOF

# ===============================
# Arquivo: scripts/evaluate_model.py
# ===============================
cat <<EOF > $PROJECT_NAME/scripts/evaluate_model.py
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scripts.preprocessing import preprocess_data

# Caminho do dataset
csv_path = "data/AnemiaTypesClassification_data.csv"

# Pré-processamento
X, y, le = preprocess_data(csv_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Carregar modelo
model = joblib.load("models/random_forest_model.pkl")

# Previsão
y_pred = model.predict(X_test)

# Avaliação
print("📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("🧱 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
EOF

# ===============================
# Arquivo: utils/helpers.py
# ===============================
cat <<EOF > $PROJECT_NAME/utils/helpers.py
import joblib

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
EOF

# ===============================
# Arquivo: utils/__init__.py
# ===============================
touch $PROJECT_NAME/utils/__init__.py

echo "✅ Arquivos criados com sucesso."
echo "🚀 Coloque seu dataset como 'AnemiaTypesClassification_data.csv' na pasta data/ e execute:"
echo "   python scripts/train_model.py"
echo "   python scripts/evaluate_model.py"

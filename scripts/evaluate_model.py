
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

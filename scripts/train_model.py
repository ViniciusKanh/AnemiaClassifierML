import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

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

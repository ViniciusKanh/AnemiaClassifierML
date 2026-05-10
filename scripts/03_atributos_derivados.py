# -*- coding: utf-8 -*-
"""
Script 03 - Comparação entre CBC original e CBC com atributos derivados.

Objetivo:
    - Ler a base de anemia.
    - Remover duplicatas exatas.
    - Comparar dois cenários de entrada:
        1. CBC original.
        2. CBC original + atributos hematológicos derivados.
    - Criar os atributos derivados dentro do pipeline, evitando vazamento.
    - Aplicar limpeza clínica ampla, imputação, winsorização e padronização quando necessário.
    - Aplicar SMOTE apenas no subconjunto de treino de cada fold.
    - Selecionar hiperparâmetros por validação cruzada estratificada no treino.
    - Avaliar o melhor modelo de cada cenário no conjunto de teste isolado.
    - Gerar tabelas, figuras, relatórios por classe, matrizes de confusão e modelos treinados.

Atributos derivados avaliados:
    - Mentzer_Index = MCV / RBC
    - HGB_RBC_ratio = HGB / RBC
    - HCT_RBC_ratio = HCT / RBC
    - PLT_RBC_ratio = PLT / RBC
    - NLR = NEUTn / LYMn
    - PLR = PLT / LYMn
    - WBC_RBC_ratio = WBC / RBC
    - PLT_WBC_ratio = PLT / WBC

Observação metodológica:
    - RDW e RDWI não são utilizados, pois RDW não existe no dataset.
    - Divisões por zero são convertidas para NaN e tratadas por imputação dentro do pipeline.
    - O conjunto de teste não participa da seleção de hiperparâmetros nem do ajuste das transformações.

Autor:
    Vinicius de Souza Santos
"""

from pathlib import Path
import warnings
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except ImportError as erro:
    raise ImportError(
        "\n[ERRO] O pacote imbalanced-learn não está instalado.\n"
        "Instale com:\n\n"
        "    pip install imbalanced-learn\n"
    ) from erro

import matplotlib.pyplot as plt


# ============================================================
# 1. Configurações gerais
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "AnemiaTypesClassification_data.csv"

RESULTS_TABLES = BASE_DIR / "results" / "tables"
RESULTS_FIGURES = BASE_DIR / "results" / "figures"
RESULTS_MODELS = BASE_DIR / "results" / "models"

RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS_MODELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Tentativa de importar XGBoost
# ============================================================

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print(
        "\n[AVISO] Pacote xgboost não encontrado. "
        "O modelo XGBoost será ignorado.\n"
        "Para instalar, use:\n\n"
        "    pip install xgboost\n"
    )


# ============================================================
# 3. Transformadores customizados
# ============================================================

class ClinicalRangeCleaner(BaseEstimator, TransformerMixin):
    """
    Substitui por NaN valores fora de limites clínicos amplos.

    Importante:
        - Os limites são fixos e definidos antes do treinamento.
        - Portanto, essa etapa não aprende informação da distribuição dos dados.
        - A imputação posterior é ajustada apenas no treino dentro do pipeline.
    """

    def __init__(self, bounds=None):
        self.bounds = bounds

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        if self.bounds is None:
            return X.copy() if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()

            for col, limits in self.bounds.items():
                if col in X_clean.columns:
                    min_value = limits["min"]
                    max_value = limits["max"]

                    mask_invalid = (
                        (X_clean[col] < min_value) |
                        (X_clean[col] > max_value)
                    )

                    X_clean.loc[mask_invalid, col] = np.nan

            return X_clean

        X_array = np.array(X, dtype=float, copy=True)

        if self.feature_names_in_ is None:
            return X_array

        X_clean = pd.DataFrame(X_array, columns=self.feature_names_in_)

        for col, limits in self.bounds.items():
            if col in X_clean.columns:
                min_value = limits["min"]
                max_value = limits["max"]

                mask_invalid = (
                    (X_clean[col] < min_value) |
                    (X_clean[col] > max_value)
                )

                X_clean.loc[mask_invalid, col] = np.nan

        return X_clean


class DerivedHematologicalFeatures(BaseEstimator, TransformerMixin):
    """
    Adiciona atributos hematológicos derivados a partir das variáveis originais do CBC.

    Os atributos são calculados dentro do pipeline para manter o protocolo reprodutível.
    Divisões por zero ou valores infinitos são convertidos para NaN.
    A imputação posterior é ajustada apenas no treino.
    """

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None
        return self

    @staticmethod
    def _safe_divide(numerator, denominator):
        denominator_safe = denominator.replace(0, np.nan)
        result = numerator / denominator_safe
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def transform(self, X):
        if not self.add_features:
            return X.copy() if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
        else:
            if self.feature_names_in_ is None:
                raise ValueError(
                    "DerivedHematologicalFeatures recebeu array sem nomes de colunas."
                )
            X_new = pd.DataFrame(np.asarray(X, dtype=float), columns=self.feature_names_in_)

        required_cols = [
            "MCV", "RBC", "HGB", "HCT", "PLT",
            "NEUTn", "LYMn", "WBC"
        ]

        missing_cols = [col for col in required_cols if col not in X_new.columns]

        if missing_cols:
            raise ValueError(
                "Colunas necessárias para atributos derivados não encontradas: "
                + ", ".join(missing_cols)
            )

        X_new["Mentzer_Index"] = self._safe_divide(X_new["MCV"], X_new["RBC"])
        X_new["HGB_RBC_ratio"] = self._safe_divide(X_new["HGB"], X_new["RBC"])
        X_new["HCT_RBC_ratio"] = self._safe_divide(X_new["HCT"], X_new["RBC"])
        X_new["PLT_RBC_ratio"] = self._safe_divide(X_new["PLT"], X_new["RBC"])
        X_new["NLR"] = self._safe_divide(X_new["NEUTn"], X_new["LYMn"])
        X_new["PLR"] = self._safe_divide(X_new["PLT"], X_new["LYMn"])
        X_new["WBC_RBC_ratio"] = self._safe_divide(X_new["WBC"], X_new["RBC"])
        X_new["PLT_WBC_ratio"] = self._safe_divide(X_new["PLT"], X_new["WBC"])

        X_new = X_new.replace([np.inf, -np.inf], np.nan)

        return X_new


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """
    Aplica winsorização por IQR.

    Importante:
        - Os limites são calculados apenas no conjunto de treino de cada fold.
        - Isso evita vazamento de informação para validação/teste.
        - Quando IQR = 0, usa mínimo e máximo observados no treino.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)

        q1 = np.nanpercentile(X_array, 25, axis=0)
        q3 = np.nanpercentile(X_array, 75, axis=0)
        iqr = q3 - q1

        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr

        degenerated = iqr == 0

        if np.any(degenerated):
            self.lower_bounds_[degenerated] = np.nanmin(
                X_array[:, degenerated],
                axis=0
            )
            self.upper_bounds_[degenerated] = np.nanmax(
                X_array[:, degenerated],
                axis=0
            )

        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)


# ============================================================
# 4. Funções auxiliares de métricas
# ============================================================

def multiclass_brier_score(y_true, y_proba, classes):
    """
    Calcula Brier score multiclasses pela média do erro quadrático
    entre matriz one-hot real e probabilidades preditas.
    """

    y_true_bin = label_binarize(y_true, classes=classes)

    if y_true_bin.shape[1] != y_proba.shape[1]:
        raise ValueError(
            "Dimensão incompatível entre y_true binarizado e y_proba."
        )

    return np.mean(np.sum((y_true_bin - y_proba) ** 2, axis=1))


def macro_average_precision(y_true, y_proba, classes):
    """
    Calcula PR-AUC macro one-vs-rest.
    """

    y_true_bin = label_binarize(y_true, classes=classes)
    scores = []

    for idx in range(len(classes)):
        if len(np.unique(y_true_bin[:, idx])) < 2:
            continue

        score = average_precision_score(y_true_bin[:, idx], y_proba[:, idx])
        scores.append(score)

    return float(np.mean(scores)) if scores else np.nan


def align_proba_columns(model, y_proba, expected_classes):
    """
    Garante que as colunas de probabilidade estejam alinhadas
    com a ordem global das classes codificadas.
    """

    model_classes = model.named_steps["model"].classes_
    aligned = np.zeros((y_proba.shape[0], len(expected_classes)))

    for idx_model, cls in enumerate(model_classes):
        idx_expected = np.where(expected_classes == cls)[0][0]
        aligned[:, idx_expected] = y_proba[:, idx_model]

    return aligned


def evaluate_model(model, X_test, y_test, classes):
    """
    Avalia modelo no conjunto de teste.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)
        y_proba = align_proba_columns(model, y_proba, classes)

        try:
            metrics["roc_auc_macro_ovr"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=classes
            )
        except Exception:
            metrics["roc_auc_macro_ovr"] = np.nan

        try:
            metrics["pr_auc_macro_ovr"] = macro_average_precision(
                y_test,
                y_proba,
                classes
            )
        except Exception:
            metrics["pr_auc_macro_ovr"] = np.nan

        try:
            metrics["brier_multiclass"] = multiclass_brier_score(
                y_test,
                y_proba,
                classes
            )
        except Exception:
            metrics["brier_multiclass"] = np.nan

    else:
        metrics["roc_auc_macro_ovr"] = np.nan
        metrics["pr_auc_macro_ovr"] = np.nan
        metrics["brier_multiclass"] = np.nan

    return metrics, y_pred


def safe_filename(text):
    """
    Converte texto em nome seguro para arquivo.
    """

    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
    )


# ============================================================
# 5. Leitura e preparação da base
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\n[ERRO] Arquivo não encontrado:\n{DATA_PATH}\n"
        "Verifique se o CSV está dentro da pasta data/."
    )

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.drop_duplicates().copy()

target_col = "Diagnosis"

if target_col not in df.columns:
    raise ValueError(
        f"\n[ERRO] A coluna-alvo '{target_col}' não foi encontrada na base."
    )

feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].copy()
y_text = df[target_col].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

classes_encoded = np.arange(len(label_encoder.classes_))

class_mapping = pd.DataFrame({
    "Classe_codificada": classes_encoded,
    "Classe_original": label_encoder.classes_
})

class_mapping.to_csv(
    RESULTS_TABLES / "03_label_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)

class_distribution = (
    pd.Series(y_text)
    .value_counts()
    .rename_axis("classe")
    .reset_index(name="quantidade")
)

class_distribution["percentual"] = (
    class_distribution["quantidade"] / class_distribution["quantidade"].sum()
)

class_distribution.to_csv(
    RESULTS_TABLES / "03_class_distribution_dedup.csv",
    index=False,
    encoding="utf-8-sig"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 6. Limites clínicos amplos usados para auditoria/limpeza
# ============================================================

clinical_bounds = {
    "WBC": {"min": 0.1, "max": 100},
    "LYMp": {"min": 0, "max": 100},
    "NEUTp": {"min": 0, "max": 100},
    "LYMn": {"min": 0, "max": 100},
    "NEUTn": {"min": 0, "max": 100},
    "RBC": {"min": 0.1, "max": 10},
    "HGB": {"min": 0, "max": 25},
    "HCT": {"min": 0, "max": 75},
    "MCV": {"min": 40, "max": 150},
    "MCH": {"min": 5, "max": 60},
    "MCHC": {"min": 10, "max": 60},
    "PLT": {"min": 1, "max": 1000},
    "PDW": {"min": 0, "max": 50},
    "PCT": {"min": 0, "max": 2}
}


# ============================================================
# 7. Definição dos cenários de atributos
# ============================================================

feature_scenarios = {
    "CBC_original": {
        "descricao": "Variáveis originais do hemograma completo",
        "add_derived": False
    },
    "CBC_com_derivados": {
        "descricao": "Variáveis originais do CBC acrescidas de atributos derivados",
        "add_derived": True
    }
}

feature_sets_info = pd.DataFrame([
    {
        "cenario": "CBC_original",
        "descricao": "Variáveis originais do hemograma completo",
        "n_atributos_entrada": len(feature_cols),
        "atributos_derivados": ""
    },
    {
        "cenario": "CBC_com_derivados",
        "descricao": "Variáveis originais do CBC acrescidas de atributos derivados",
        "n_atributos_entrada": len(feature_cols) + 8,
        "atributos_derivados": (
            "Mentzer_Index; HGB_RBC_ratio; HCT_RBC_ratio; "
            "PLT_RBC_ratio; NLR; PLR; WBC_RBC_ratio; PLT_WBC_ratio"
        )
    }
])

feature_sets_info.to_csv(
    RESULTS_TABLES / "03_feature_sets.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. Função para montar pipeline
# ============================================================

def build_pipeline(model, use_scaler=False, use_smote=True, add_derived=False):
    """
    Monta pipeline de treino.

    Ordem:
        1. Limpeza por limites clínicos amplos.
        2. Criação opcional de atributos derivados.
        3. Imputação por mediana.
        4. Winsorização por IQR ajustada no treino.
        5. Padronização, se necessário.
        6. SMOTE apenas no treino.
        7. Modelo final.
    """

    steps = [
        ("clinical_cleaner", ClinicalRangeCleaner(bounds=clinical_bounds)),
        ("derived_features", DerivedHematologicalFeatures(add_features=add_derived)),
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", IQRWinsorizer(factor=1.5))
    ]

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=1
                )
            )
        )

    steps.append(("model", model))

    return ImbPipeline(steps=steps)


# ============================================================
# 9. Função para definir modelos e grades
# ============================================================

def get_models(add_derived=False):
    """
    Retorna os modelos e grades de hiperparâmetros para um cenário de atributos.
    """

    models = {
        "Logistic Regression": {
            "pipeline": build_pipeline(
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE
                ),
                use_scaler=True,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["lbfgs"],
                "model__class_weight": [None, "balanced"]
            }
        },

        "SVM": {
            "pipeline": build_pipeline(
                SVC(
                    probability=True,
                    random_state=RANDOM_STATE
                ),
                use_scaler=True,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"]
            }
        },

        "KNN": {
            "pipeline": build_pipeline(
                KNeighborsClassifier(),
                use_scaler=True,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]
            }
        },

        "GaussianNB": {
            "pipeline": build_pipeline(
                GaussianNB(),
                use_scaler=True,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },

        "Random Forest": {
            "pipeline": build_pipeline(
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ),
                use_scaler=False,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", "log2"]
            }
        },

        "MLP": {
            "pipeline": build_pipeline(
                MLPClassifier(
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                    max_iter=500
                ),
                use_scaler=True,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [1e-3, 3e-3]
            }
        }
    }

    if HAS_XGBOOST:
        models["XGBoost"] = {
            "pipeline": build_pipeline(
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist"
                ),
                use_scaler=False,
                use_smote=True,
                add_derived=add_derived
            ),
            "params": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }

    return models


# ============================================================
# 10. Treinamento e avaliação
# ============================================================

cv_inner = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

all_results = []
best_models = {}
best_predictions = {}

for scenario_name, scenario_config in feature_scenarios.items():
    print("\n" + "=" * 80)
    print(f"CENÁRIO DE ATRIBUTOS: {scenario_name}")
    print(scenario_config["descricao"])
    print("=" * 80)

    add_derived = scenario_config["add_derived"]
    models = get_models(add_derived=add_derived)

    for model_name, config in models.items():
        print(f"\n================ TREINANDO: {scenario_name} | {model_name} ================\n")

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["params"],
            scoring="f1_macro",
            cv=cv_inner,
            n_jobs=-1,
            refit=True,
            verbose=1,
            error_score="raise"
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        metrics, y_pred = evaluate_model(
            best_model,
            X_test,
            y_test,
            classes_encoded
        )

        key = (scenario_name, model_name)
        best_models[key] = best_model
        best_predictions[key] = y_pred

        row = {
            "cenario": scenario_name,
            "modelo": model_name,
            "usa_atributos_derivados": add_derived,
            "best_cv_f1_macro": grid.best_score_,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
            "pr_auc_macro_ovr": metrics["pr_auc_macro_ovr"],
            "brier_multiclass": metrics["brier_multiclass"],
            "best_params": json.dumps(grid.best_params_, ensure_ascii=False)
        }

        all_results.append(row)

        model_file = (
            RESULTS_MODELS /
            f"03_best_model_{safe_filename(scenario_name)}_{safe_filename(model_name)}.joblib"
        )

        joblib.dump(best_model, model_file)

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=classes_encoded,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(
            RESULTS_TABLES /
            f"03_classification_report_{safe_filename(scenario_name)}_{safe_filename(model_name)}.csv",
            encoding="utf-8-sig"
        )

        print(f"Melhor F1 macro na CV interna: {grid.best_score_:.4f}")
        print(f"F1 macro no teste: {metrics['f1_macro']:.4f}")
        print(f"F1 ponderado no teste: {metrics['f1_weighted']:.4f}")
        print(f"Recall macro no teste: {metrics['recall_macro']:.4f}")


# ============================================================
# 11. Salvamento dos resultados globais
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["f1_macro", "f1_weighted"],
    ascending=[False, False]
).reset_index(drop=True)

results_df.to_csv(
    RESULTS_TABLES / "03_atributos_derivados_resultados.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n================ RESULTADOS GERAIS - ATRIBUTOS DERIVADOS ================\n")
print(
    results_df[
        [
            "cenario",
            "modelo",
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "roc_auc_macro_ovr",
            "pr_auc_macro_ovr",
            "brier_multiclass"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. Melhor modelo por cenário
# ============================================================

best_by_scenario = (
    results_df
    .sort_values(
        by=["cenario", "f1_macro", "f1_weighted"],
        ascending=[True, False, False]
    )
    .groupby("cenario", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_by_scenario.to_csv(
    RESULTS_TABLES / "03_melhor_modelo_por_cenario.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. Comparação direta CBC original vs CBC com derivados por modelo
# ============================================================

metric_cols = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc_macro_ovr",
    "pr_auc_macro_ovr",
    "brier_multiclass"
]

original_df = (
    results_df[results_df["cenario"] == "CBC_original"]
    .set_index("modelo")[metric_cols]
)

derived_df = (
    results_df[results_df["cenario"] == "CBC_com_derivados"]
    .set_index("modelo")[metric_cols]
)

common_models = sorted(set(original_df.index).intersection(set(derived_df.index)))

delta_rows = []

for model_name in common_models:
    row = {"modelo": model_name}

    for metric in metric_cols:
        original_value = original_df.loc[model_name, metric]
        derived_value = derived_df.loc[model_name, metric]
        row[f"{metric}_original"] = original_value
        row[f"{metric}_derivado"] = derived_value
        row[f"delta_{metric}"] = derived_value - original_value

    delta_rows.append(row)

delta_df = pd.DataFrame(delta_rows)

delta_df = delta_df.sort_values(
    by="delta_f1_macro",
    ascending=False
).reset_index(drop=True)

delta_df.to_csv(
    RESULTS_TABLES / "03_delta_atributos_derivados_por_modelo.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n================ IMPACTO DOS ATRIBUTOS DERIVADOS POR MODELO ================\n")

if not delta_df.empty:
    print(
        delta_df[
            [
                "modelo",
                "f1_macro_original",
                "f1_macro_derivado",
                "delta_f1_macro",
                "recall_macro_original",
                "recall_macro_derivado",
                "delta_recall_macro",
                "brier_multiclass_original",
                "brier_multiclass_derivado",
                "delta_brier_multiclass"
            ]
        ].to_string(index=False)
    )
else:
    print("Não foi possível calcular deltas entre os cenários.")


# ============================================================
# 14. Figura comparativa: F1 macro por modelo e cenário
# ============================================================

f1_pivot = results_df.pivot(
    index="modelo",
    columns="cenario",
    values="f1_macro"
)

f1_pivot = f1_pivot.sort_values(
    by="CBC_com_derivados" if "CBC_com_derivados" in f1_pivot.columns else f1_pivot.columns[0],
    ascending=False
)

ax = f1_pivot.plot(kind="bar", figsize=(11, 6))
ax.set_title("Comparação de F1 macro: CBC original vs CBC com atributos derivados")
ax.set_ylabel("F1 macro")
ax.set_xlabel("Modelo")
ax.set_ylim(0, 1.05)
ax.legend(title="Cenário")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "03_comparacao_f1_macro_atributos_derivados.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. Figura comparativa: deltas de F1 macro
# ============================================================

if not delta_df.empty:
    delta_plot = delta_df.set_index("modelo")["delta_f1_macro"].sort_values(ascending=False)

    ax = delta_plot.plot(kind="bar", figsize=(10, 5))
    ax.axhline(0, linewidth=1)
    ax.set_title("Impacto dos atributos derivados sobre o F1 macro")
    ax.set_ylabel("Delta F1 macro")
    ax.set_xlabel("Modelo")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES / "03_delta_f1_macro_atributos_derivados.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 16. Matrizes de confusão dos melhores modelos por cenário
# ============================================================

for _, row in best_by_scenario.iterrows():
    scenario_name = row["cenario"]
    model_name = row["modelo"]

    key = (scenario_name, model_name)

    best_model = best_models[key]
    y_pred_best = best_predictions[key]

    cm = confusion_matrix(y_test, y_pred_best, labels=classes_encoded)

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_file = (
        RESULTS_TABLES /
        f"03_confusion_matrix_{safe_filename(scenario_name)}_{safe_filename(model_name)}.csv"
    )

    cm_df.to_csv(cm_file, encoding="utf-8-sig")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, xticks_rotation=90, values_format="d")
    ax.set_title(f"Matriz de confusão - {scenario_name} - {model_name}")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES /
        f"03_confusion_matrix_{safe_filename(scenario_name)}_{safe_filename(model_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    cm_normalized = confusion_matrix(
        y_test,
        y_pred_best,
        labels=classes_encoded,
        normalize="true"
    )

    cm_norm_df = pd.DataFrame(
        cm_normalized,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_norm_file = (
        RESULTS_TABLES /
        f"03_confusion_matrix_normalized_{safe_filename(scenario_name)}_{safe_filename(model_name)}.csv"
    )

    cm_norm_df.to_csv(cm_norm_file, encoding="utf-8-sig")

    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp_norm.plot(ax=ax, xticks_rotation=90, values_format=".2f")
    ax.set_title(f"Matriz de confusão normalizada - {scenario_name} - {model_name}")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES /
        f"03_confusion_matrix_normalized_{safe_filename(scenario_name)}_{safe_filename(model_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 17. Salvamento do LabelEncoder e informações da execução
# ============================================================

joblib.dump(label_encoder, RESULTS_MODELS / "03_label_encoder.joblib")

split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_total_original": len(df_raw),
    "n_total_deduplicado": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "criterio_selecao": "f1_macro",
    "cenarios_avaliados": "; ".join(feature_scenarios.keys()),
    "xgboost_disponivel": HAS_XGBOOST
}

pd.DataFrame([split_info]).to_csv(
    RESULTS_TABLES / "03_split_info.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 18. Encerramento
# ============================================================

print("\nArquivos gerados:")
print(f"- {RESULTS_TABLES / '03_atributos_derivados_resultados.csv'}")
print(f"- {RESULTS_TABLES / '03_melhor_modelo_por_cenario.csv'}")
print(f"- {RESULTS_TABLES / '03_delta_atributos_derivados_por_modelo.csv'}")
print(f"- {RESULTS_TABLES / '03_feature_sets.csv'}")
print(f"- {RESULTS_TABLES / '03_class_distribution_dedup.csv'}")
print(f"- {RESULTS_TABLES / '03_label_mapping.csv'}")
print(f"- {RESULTS_TABLES / '03_split_info.csv'}")
print(f"- {RESULTS_FIGURES / '03_comparacao_f1_macro_atributos_derivados.png'}")
print(f"- {RESULTS_FIGURES / '03_delta_f1_macro_atributos_derivados.png'}")
print(f"- {RESULTS_MODELS}")

print("\nComparação CBC original vs CBC com atributos derivados finalizada com sucesso.")
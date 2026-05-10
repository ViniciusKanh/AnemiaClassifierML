# -*- coding: utf-8 -*-
"""
Script 02 - Benchmark base dos classificadores.

Objetivo:
    - Ler a base de anemia.
    - Remover duplicatas exatas.
    - Aplicar limpeza clínica ampla dentro do pipeline.
    - Aplicar imputação, winsorização e, quando necessário, padronização.
    - Aplicar SMOTE apenas no conjunto de treino de cada fold.
    - Treinar modelos com GridSearchCV no conjunto de treino.
    - Selecionar modelos usando F1 macro como critério primário.
    - Avaliar o melhor modelo de cada família no conjunto de teste.
    - Gerar tabelas, figuras, matriz de confusão e artefatos reprodutíveis.

Observação metodológica:
    Este script gera o benchmark inicial com atributos originais do CBC.
    Os experimentos com atributos derivados, balanceamento, calibração,
    limiares por classe, abstenção e interpretabilidade serão tratados
    em scripts posteriores.

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
        - Os limites são fixos, definidos antes do treinamento.
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
            return np.asarray(X, dtype=float)

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

            return X_clean.values

        X_clean = np.array(X, dtype=float, copy=True)

        if self.feature_names_in_ is None:
            return X_clean

        for idx, col in enumerate(self.feature_names_in_):
            if col in self.bounds:
                min_value = self.bounds[col]["min"]
                max_value = self.bounds[col]["max"]

                mask_invalid = (
                    (X_clean[:, idx] < min_value) |
                    (X_clean[:, idx] > max_value)
                )

                X_clean[mask_invalid, idx] = np.nan

        return X_clean


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
    Converte nome de modelo em nome seguro para arquivo.
    """

    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
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
    RESULTS_TABLES / "02_label_mapping.csv",
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
    RESULTS_TABLES / "02_class_distribution_dedup.csv",
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
# 7. Função para montar pipeline
# ============================================================

def build_pipeline(model, use_scaler=False, use_smote=True):
    """
    Monta pipeline de treino.

    Ordem:
        1. Limpeza por limites clínicos amplos.
        2. Imputação por mediana.
        3. Winsorização por IQR ajustada no treino.
        4. Padronização, se necessário.
        5. SMOTE apenas no treino.
        6. Modelo final.
    """

    steps = [
        ("clinical_cleaner", ClinicalRangeCleaner(bounds=clinical_bounds)),
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
# 8. Definição dos modelos e grades compactas
# ============================================================

models = {
    "Logistic Regression": {
        "pipeline": build_pipeline(
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            ),
            use_scaler=True,
            use_smote=True
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
            use_smote=True
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
            use_smote=True
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
            use_smote=True
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
            use_smote=True
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
            use_smote=True
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
            use_smote=True
        ),
        "params": {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    }


# ============================================================
# 9. Treinamento com GridSearchCV
# ============================================================

cv_inner = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
best_models = {}

for model_name, config in models.items():
    print(f"\n================ TREINANDO: {model_name} ================\n")

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
    best_models[model_name] = best_model

    metrics, y_pred = evaluate_model(
        best_model,
        X_test,
        y_test,
        classes_encoded
    )

    row = {
        "modelo": model_name,
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

    results.append(row)

    model_file = RESULTS_MODELS / f"02_best_model_{safe_filename(model_name)}.joblib"
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
        RESULTS_TABLES / f"02_classification_report_{safe_filename(model_name)}.csv",
        encoding="utf-8-sig"
    )

    print(f"Melhor F1 macro na CV interna: {grid.best_score_:.4f}")
    print(f"F1 macro no teste: {metrics['f1_macro']:.4f}")
    print(f"F1 ponderado no teste: {metrics['f1_weighted']:.4f}")
    print(f"Recall macro no teste: {metrics['recall_macro']:.4f}")


# ============================================================
# 10. Salvamento dos resultados globais
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    RESULTS_TABLES / "02_benchmark_base.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n================ RESULTADOS DO BENCHMARK BASE ================\n")
print(
    results_df[
        [
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
# 11. Figura comparativa das principais métricas
# ============================================================

plot_df = results_df.set_index("modelo")[
    [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "recall_macro"
    ]
]

ax = plot_df.plot(kind="bar", figsize=(12, 6))
ax.set_title("Desempenho dos classificadores no conjunto de teste")
ax.set_ylabel("Valor da métrica")
ax.set_xlabel("Modelo")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "02_benchmark_metrics_bars.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. Matriz de confusão do melhor modelo por F1 macro
# ============================================================

best_model_name = results_df.iloc[0]["modelo"]
best_model = best_models[best_model_name]

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best, labels=classes_encoded)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_df.to_csv(
    RESULTS_TABLES / "02_confusion_matrix_best_model.csv",
    encoding="utf-8-sig"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(11, 9))
disp.plot(ax=ax, xticks_rotation=90, values_format="d")
ax.set_title(f"Matriz de confusão - {best_model_name}")
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "02_confusion_matrix_best_model.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. Matriz de confusão normalizada por classe real
# ============================================================

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

cm_norm_df.to_csv(
    RESULTS_TABLES / "02_confusion_matrix_best_model_normalized.csv",
    encoding="utf-8-sig"
)

disp_norm = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(11, 9))
disp_norm.plot(ax=ax, xticks_rotation=90, values_format=".2f")
ax.set_title(f"Matriz de confusão normalizada - {best_model_name}")
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "02_confusion_matrix_best_model_normalized.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. Salvamento do LabelEncoder e informações da partição
# ============================================================

joblib.dump(label_encoder, RESULTS_MODELS / "02_label_encoder.joblib")

split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_total_original": len(df_raw),
    "n_total_deduplicado": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "criterio_selecao": "f1_macro",
    "melhor_modelo_f1_macro": best_model_name,
    "xgboost_disponivel": HAS_XGBOOST
}

pd.DataFrame([split_info]).to_csv(
    RESULTS_TABLES / "02_split_info.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Encerramento
# ============================================================

print("\nArquivos gerados:")
print(f"- {RESULTS_TABLES / '02_benchmark_base.csv'}")
print(f"- {RESULTS_TABLES / '02_label_mapping.csv'}")
print(f"- {RESULTS_TABLES / '02_class_distribution_dedup.csv'}")
print(f"- {RESULTS_TABLES / '02_confusion_matrix_best_model.csv'}")
print(f"- {RESULTS_TABLES / '02_confusion_matrix_best_model_normalized.csv'}")
print(f"- {RESULTS_TABLES / '02_split_info.csv'}")
print(f"- {RESULTS_FIGURES / '02_benchmark_metrics_bars.png'}")
print(f"- {RESULTS_FIGURES / '02_confusion_matrix_best_model.png'}")
print(f"- {RESULTS_FIGURES / '02_confusion_matrix_best_model_normalized.png'}")
print(f"- {RESULTS_MODELS}")

print("\nBenchmark base finalizado com sucesso.")
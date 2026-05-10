# -*- coding: utf-8 -*-
"""
Script 04 - Avaliação de estratégias de balanceamento.

Objetivo:
    - Avaliar o impacto de diferentes estratégias de tratamento do
      desbalanceamento de classes.
    - Utilizar o melhor cenário da etapa anterior:
        CBC original + atributos hematológicos derivados.
    - Comparar as seguintes estratégias:
        1. Sem balanceamento explícito.
        2. Ponderação por classe / pesos amostrais balanceados.
        3. SMOTE.
        4. Borderline-SMOTE.
        5. SMOTE-ENN.
    - Selecionar hiperparâmetros por validação cruzada estratificada
      no conjunto de treinamento.
    - Usar F1 macro como critério primário.
    - Avaliar o melhor modelo de cada estratégia no conjunto de teste isolado.
    - Gerar tabelas, relatórios por classe, matrizes de confusão e figuras.

Observação metodológica:
    - O conjunto de teste permanece isolado.
    - SMOTE, Borderline-SMOTE e SMOTE-ENN são aplicados apenas no treino
      de cada fold, dentro do pipeline.
    - Para XGBoost, a estratégia "class_weight" é implementada por pesos
      amostrais balanceados calculados exclusivamente no conjunto usado
      em cada chamada de fit.
    - Os atributos derivados são calculados dentro do pipeline.

Autor:
    Vinicius de Souza Santos
"""

from pathlib import Path
import warnings
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.combine import SMOTEENN
    from imblearn.under_sampling import EditedNearestNeighbours
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
# 3. Modelos avaliados nesta etapa
# ============================================================
#
# Por padrão, a etapa 04 avalia o modelo líder da etapa 03.
# Se XGBoost estiver instalado, ele será avaliado.
# Caso contrário, usa Random Forest como alternativa.
#
# Para avaliar os dois líderes, altere manualmente para:
# MODELOS_AVALIADOS = ["XGBoost", "Random Forest"]
# ============================================================

if HAS_XGBOOST:
    MODELOS_AVALIADOS = ["XGBoost"]
else:
    MODELOS_AVALIADOS = ["Random Forest"]


# ============================================================
# 4. Transformadores customizados
# ============================================================

class ClinicalRangeCleaner(BaseEstimator, TransformerMixin):
    """
    Substitui por NaN valores fora de limites clínicos amplos.

    Os limites são fixos e definidos antes do treinamento.
    Essa etapa não aprende informação da distribuição dos dados.
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
            if isinstance(X, pd.DataFrame):
                return X.copy()
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
    Adiciona atributos hematológicos derivados ao CBC original.

    Atributos:
        - Mentzer_Index = MCV / RBC
        - HGB_RBC_ratio = HGB / RBC
        - HCT_RBC_ratio = HCT / RBC
        - PLT_RBC_ratio = PLT / RBC
        - NLR = NEUTn / LYMn
        - PLR = PLT / LYMn
        - WBC_RBC_ratio = WBC / RBC
        - PLT_WBC_ratio = PLT / WBC

    Divisões por zero e valores infinitos são convertidos para NaN.
    A imputação posterior ocorre dentro do pipeline.
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
            if isinstance(X, pd.DataFrame):
                return X.copy()
            return np.asarray(X, dtype=float)

        if isinstance(X, pd.DataFrame):
            X_new = X.copy()
        else:
            if self.feature_names_in_ is None:
                raise ValueError(
                    "DerivedHematologicalFeatures recebeu array sem nomes de colunas."
                )
            X_new = pd.DataFrame(
                np.asarray(X, dtype=float),
                columns=self.feature_names_in_
            )

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

    Os limites são calculados apenas no conjunto de treino de cada fold.
    Quando IQR = 0, usa mínimo e máximo observados no treino.
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
# 5. Wrapper seguro para XGBoost com pesos balanceados
# ============================================================

class SafeXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper para XGBClassifier.

    Finalidade:
        - Permitir uso de pesos amostrais balanceados.
        - Evitar erro quando alguma estratégia de reamostragem remove
          uma classe e os rótulos deixam de ser contíguos.
        - Mapear rótulos originais para 0..K-1 durante o fit e restaurar
          os rótulos originais no predict.

    A estratégia class_weight para XGBoost é implementada por:
        sample_weight = compute_sample_weight("balanced", y)
    """

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        balanced_sample_weight=False
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.balanced_sample_weight = balanced_sample_weight

    def fit(self, X, y):
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost não está instalado. Execute: pip install xgboost"
            )

        y_array = np.asarray(y)

        self.classes_ = np.unique(y_array)
        self.class_to_index_ = {
            original_class: idx
            for idx, original_class in enumerate(self.classes_)
        }
        self.index_to_class_ = {
            idx: original_class
            for original_class, idx in self.class_to_index_.items()
        }

        y_mapped = np.array([self.class_to_index_[value] for value in y_array])

        sample_weight = None

        if self.balanced_sample_weight:
            sample_weight = compute_sample_weight(
                class_weight="balanced",
                y=y_mapped
            )

        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            num_class=len(self.classes_)
        )

        self.estimator_.fit(
            X,
            y_mapped,
            sample_weight=sample_weight
        )

        return self

    def predict(self, X):
        y_pred_mapped = self.estimator_.predict(X)
        y_pred_mapped = np.asarray(y_pred_mapped).astype(int)
        return np.array([self.index_to_class_[idx] for idx in y_pred_mapped])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ============================================================
# 6. Funções auxiliares de métricas
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
    Alinha as colunas de probabilidade com a ordem global das classes.
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
        .replace("-", "_")
    )


# ============================================================
# 7. Leitura e preparação da base
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
    RESULTS_TABLES / "04_label_mapping.csv",
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
    class_distribution["quantidade"] /
    class_distribution["quantidade"].sum()
)

class_distribution.to_csv(
    RESULTS_TABLES / "04_class_distribution_dedup.csv",
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
# 8. Limites clínicos amplos
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
# 9. Estratégias de balanceamento
# ============================================================

BALANCING_STRATEGIES = {
    "sem_balanceamento": {
        "descricao": "Sem balanceamento explícito",
        "resampler": None,
        "class_weight": False
    },
    "class_weight": {
        "descricao": "Ponderação por classe ou pesos amostrais balanceados",
        "resampler": None,
        "class_weight": True
    },
    "SMOTE": {
        "descricao": "SMOTE com k_neighbors=1 aplicado apenas no treino",
        "resampler": SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=1
        ),
        "class_weight": False
    },
    "Borderline_SMOTE": {
        "descricao": "Borderline-SMOTE aplicado apenas no treino",
        "resampler": BorderlineSMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=1,
            m_neighbors=3,
            kind="borderline-1"
        ),
        "class_weight": False
    },
    "SMOTE_ENN": {
        "descricao": "SMOTE-ENN aplicado apenas no treino",
        "resampler": SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=1
            ),
            enn=EditedNearestNeighbours(
                n_neighbors=3
            )
        ),
        "class_weight": False
    }
}

strategies_info = pd.DataFrame([
    {
        "estrategia": strategy_name,
        "descricao": config["descricao"],
        "usa_reamostragem": config["resampler"] is not None,
        "usa_ponderacao": config["class_weight"]
    }
    for strategy_name, config in BALANCING_STRATEGIES.items()
])

strategies_info.to_csv(
    RESULTS_TABLES / "04_balanceamento_estrategias.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Função para criar modelos
# ============================================================

def create_model(model_name, use_class_weight):
    """
    Cria o estimador final de acordo com a estratégia de balanceamento.
    """

    if model_name == "XGBoost":
        return SafeXGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            balanced_sample_weight=use_class_weight
        )

    if model_name == "Random Forest":
        class_weight_value = "balanced" if use_class_weight else None

        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight_value
        )

    raise ValueError(f"Modelo não suportado nesta etapa: {model_name}")


def get_param_grid(model_name):
    """
    Retorna a grade compacta de hiperparâmetros.
    """

    if model_name == "XGBoost":
        return {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }

    if model_name == "Random Forest":
        return {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"]
        }

    raise ValueError(f"Modelo não suportado nesta etapa: {model_name}")


# ============================================================
# 11. Função para montar pipeline
# ============================================================

def build_pipeline(model, resampler=None):
    """
    Monta o pipeline experimental.

    Ordem:
        1. Limpeza clínica ampla.
        2. Atributos derivados.
        3. Imputação por mediana.
        4. Winsorização por IQR.
        5. Reamostragem, quando aplicável.
        6. Modelo final.

    Observação:
        Nesta etapa, o cenário fixo é CBC + atributos derivados.
    """

    steps = [
        ("clinical_cleaner", ClinicalRangeCleaner(bounds=clinical_bounds)),
        ("derived_features", DerivedHematologicalFeatures(add_features=True)),
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", IQRWinsorizer(factor=1.5))
    ]

    if resampler is not None:
        steps.append(("resampler", resampler))

    steps.append(("model", model))

    return ImbPipeline(steps=steps)


# ============================================================
# 12. Treinamento e avaliação
# ============================================================

cv_inner = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

all_results = []
best_models = {}
best_predictions = {}

for model_name in MODELOS_AVALIADOS:
    print("\n" + "=" * 80)
    print(f"MODELO AVALIADO: {model_name}")
    print("Cenário fixo: CBC original + atributos derivados")
    print("=" * 80)

    for strategy_name, strategy_config in BALANCING_STRATEGIES.items():
        print(
            f"\n================ TREINANDO: {model_name} | "
            f"{strategy_name} ================\n"
        )

        model = create_model(
            model_name=model_name,
            use_class_weight=strategy_config["class_weight"]
        )

        pipeline = build_pipeline(
            model=model,
            resampler=strategy_config["resampler"]
        )

        param_grid = get_param_grid(model_name)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv_inner,
            n_jobs=-1,
            refit=True,
            verbose=1,
            error_score=np.nan
        )

        try:
            grid.fit(X_train, y_train)

            if np.isnan(grid.best_score_):
                raise RuntimeError(
                    "Todos os candidatos retornaram score NaN."
                )

            best_model = grid.best_estimator_

            metrics, y_pred = evaluate_model(
                best_model,
                X_test,
                y_test,
                classes_encoded
            )

            status = "ok"
            error_message = ""

            key = (model_name, strategy_name)
            best_models[key] = best_model
            best_predictions[key] = y_pred

            model_file = (
                RESULTS_MODELS /
                f"04_best_model_{safe_filename(model_name)}_{safe_filename(strategy_name)}.joblib"
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
                f"04_classification_report_{safe_filename(model_name)}_{safe_filename(strategy_name)}.csv",
                encoding="utf-8-sig"
            )

            print(f"Melhor F1 macro na CV interna: {grid.best_score_:.4f}")
            print(f"F1 macro no teste: {metrics['f1_macro']:.4f}")
            print(f"F1 ponderado no teste: {metrics['f1_weighted']:.4f}")
            print(f"Recall macro no teste: {metrics['recall_macro']:.4f}")

            row = {
                "modelo": model_name,
                "estrategia_balanceamento": strategy_name,
                "descricao_estrategia": strategy_config["descricao"],
                "status": status,
                "erro": error_message,
                "usa_reamostragem": strategy_config["resampler"] is not None,
                "usa_ponderacao": strategy_config["class_weight"],
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

        except Exception as erro:
            print(f"[ERRO] Falha na estratégia {strategy_name}: {erro}")

            row = {
                "modelo": model_name,
                "estrategia_balanceamento": strategy_name,
                "descricao_estrategia": strategy_config["descricao"],
                "status": "erro",
                "erro": str(erro),
                "usa_reamostragem": strategy_config["resampler"] is not None,
                "usa_ponderacao": strategy_config["class_weight"],
                "best_cv_f1_macro": np.nan,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "f1_macro": np.nan,
                "f1_weighted": np.nan,
                "roc_auc_macro_ovr": np.nan,
                "pr_auc_macro_ovr": np.nan,
                "brier_multiclass": np.nan,
                "best_params": ""
            }

        all_results.append(row)


# ============================================================
# 13. Salvamento dos resultados globais
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["f1_macro", "f1_weighted"],
    ascending=[False, False],
    na_position="last"
).reset_index(drop=True)

results_df.to_csv(
    RESULTS_TABLES / "04_balanceamento_resultados.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n================ RESULTADOS GERAIS - BALANCEAMENTO ================\n")

print(
    results_df[
        [
            "modelo",
            "estrategia_balanceamento",
            "status",
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
# 14. Melhor estratégia por modelo
# ============================================================

valid_results_df = results_df[results_df["status"] == "ok"].copy()

if valid_results_df.empty:
    raise RuntimeError(
        "Nenhuma estratégia de balanceamento foi executada com sucesso."
    )

best_by_model = (
    valid_results_df
    .sort_values(
        by=["modelo", "f1_macro", "f1_weighted"],
        ascending=[True, False, False]
    )
    .groupby("modelo", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_by_model.to_csv(
    RESULTS_TABLES / "04_melhor_estrategia_por_modelo.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Delta em relação ao cenário sem balanceamento
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

delta_rows = []

for model_name in valid_results_df["modelo"].unique():
    model_df = valid_results_df[valid_results_df["modelo"] == model_name].copy()

    baseline_df = model_df[
        model_df["estrategia_balanceamento"] == "sem_balanceamento"
    ]

    if baseline_df.empty:
        continue

    baseline = baseline_df.iloc[0]

    for _, row in model_df.iterrows():
        delta_row = {
            "modelo": model_name,
            "estrategia_balanceamento": row["estrategia_balanceamento"]
        }

        for metric in metric_cols:
            delta_row[f"{metric}_baseline"] = baseline[metric]
            delta_row[f"{metric}_estrategia"] = row[metric]
            delta_row[f"delta_{metric}"] = row[metric] - baseline[metric]

        delta_rows.append(delta_row)

delta_df = pd.DataFrame(delta_rows)

if not delta_df.empty:
    delta_df = delta_df.sort_values(
        by=["modelo", "delta_f1_macro"],
        ascending=[True, False]
    ).reset_index(drop=True)

delta_df.to_csv(
    RESULTS_TABLES / "04_delta_balanceamento_vs_sem_balanceamento.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. Figuras comparativas
# ============================================================

for model_name in valid_results_df["modelo"].unique():
    model_df = valid_results_df[
        valid_results_df["modelo"] == model_name
    ].copy()

    model_df = model_df.sort_values(
        by="f1_macro",
        ascending=False
    )

    plot_df = model_df.set_index("estrategia_balanceamento")[
        [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "recall_macro"
        ]
    ]

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(f"Impacto do balanceamento - {model_name}")
    ax.set_ylabel("Valor da métrica")
    ax.set_xlabel("Estratégia de balanceamento")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES /
        f"04_balanceamento_metricas_{safe_filename(model_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    f1_series = model_df.set_index("estrategia_balanceamento")["f1_macro"]

    ax = f1_series.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"F1 macro por estratégia de balanceamento - {model_name}")
    ax.set_ylabel("F1 macro")
    ax.set_xlabel("Estratégia de balanceamento")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES /
        f"04_balanceamento_f1_macro_{safe_filename(model_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if not delta_df.empty:
    for model_name in delta_df["modelo"].unique():
        model_delta_df = delta_df[
            delta_df["modelo"] == model_name
        ].copy()

        delta_plot = model_delta_df.set_index(
            "estrategia_balanceamento"
        )["delta_f1_macro"].sort_values(ascending=False)

        ax = delta_plot.plot(kind="bar", figsize=(10, 5))
        ax.axhline(0, linewidth=1)
        ax.set_title(
            f"Delta de F1 macro em relação ao cenário sem balanceamento - {model_name}"
        )
        ax.set_ylabel("Delta F1 macro")
        ax.set_xlabel("Estratégia de balanceamento")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        plt.savefig(
            RESULTS_FIGURES /
            f"04_delta_f1_macro_balanceamento_{safe_filename(model_name)}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 17. Matriz de confusão da melhor estratégia global
# ============================================================

best_global_row = valid_results_df.sort_values(
    by=["f1_macro", "f1_weighted"],
    ascending=[False, False]
).iloc[0]

best_model_name = best_global_row["modelo"]
best_strategy_name = best_global_row["estrategia_balanceamento"]

best_key = (best_model_name, best_strategy_name)

best_model = best_models[best_key]
y_pred_best = best_predictions[best_key]

cm = confusion_matrix(
    y_test,
    y_pred_best,
    labels=classes_encoded
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_df.to_csv(
    RESULTS_TABLES / "04_confusion_matrix_best_balanceamento.csv",
    encoding="utf-8-sig"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(11, 9))
disp.plot(ax=ax, xticks_rotation=90, values_format="d")
ax.set_title(
    f"Matriz de confusão - {best_model_name} - {best_strategy_name}"
)
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "04_confusion_matrix_best_balanceamento.png",
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

cm_norm_df.to_csv(
    RESULTS_TABLES / "04_confusion_matrix_best_balanceamento_normalized.csv",
    encoding="utf-8-sig"
)

disp_norm = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(11, 9))
disp_norm.plot(ax=ax, xticks_rotation=90, values_format=".2f")
ax.set_title(
    f"Matriz de confusão normalizada - {best_model_name} - {best_strategy_name}"
)
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "04_confusion_matrix_best_balanceamento_normalized.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. Relatório por classe da melhor estratégia global
# ============================================================

best_report_dict = classification_report(
    y_test,
    y_pred_best,
    labels=classes_encoded,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

best_report_df = pd.DataFrame(best_report_dict).transpose()

best_report_df.to_csv(
    RESULTS_TABLES / "04_classification_report_best_balanceamento.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 19. Salvamento do LabelEncoder e informações da execução
# ============================================================

joblib.dump(label_encoder, RESULTS_MODELS / "04_label_encoder.joblib")

split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_total_original": len(df_raw),
    "n_total_deduplicado": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "criterio_selecao": "f1_macro",
    "cenario_atributos": "CBC_original_plus_atributos_derivados",
    "modelos_avaliados": "; ".join(MODELOS_AVALIADOS),
    "estrategias_avaliadas": "; ".join(BALANCING_STRATEGIES.keys()),
    "melhor_modelo": best_model_name,
    "melhor_estrategia": best_strategy_name,
    "xgboost_disponivel": HAS_XGBOOST
}

pd.DataFrame([split_info]).to_csv(
    RESULTS_TABLES / "04_split_info.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. Encerramento
# ============================================================

print("\n================ MELHOR RESULTADO GLOBAL ================\n")
print(f"Modelo: {best_model_name}")
print(f"Estratégia: {best_strategy_name}")
print(f"F1 macro: {best_global_row['f1_macro']:.4f}")
print(f"F1 ponderado: {best_global_row['f1_weighted']:.4f}")
print(f"Recall macro: {best_global_row['recall_macro']:.4f}")
print(f"Acurácia: {best_global_row['accuracy']:.4f}")
print(f"Brier multiclasses: {best_global_row['brier_multiclass']:.4f}")

print("\nArquivos gerados:")
print(f"- {RESULTS_TABLES / '04_balanceamento_resultados.csv'}")
print(f"- {RESULTS_TABLES / '04_melhor_estrategia_por_modelo.csv'}")
print(f"- {RESULTS_TABLES / '04_delta_balanceamento_vs_sem_balanceamento.csv'}")
print(f"- {RESULTS_TABLES / '04_balanceamento_estrategias.csv'}")
print(f"- {RESULTS_TABLES / '04_class_distribution_dedup.csv'}")
print(f"- {RESULTS_TABLES / '04_label_mapping.csv'}")
print(f"- {RESULTS_TABLES / '04_confusion_matrix_best_balanceamento.csv'}")
print(f"- {RESULTS_TABLES / '04_confusion_matrix_best_balanceamento_normalized.csv'}")
print(f"- {RESULTS_TABLES / '04_classification_report_best_balanceamento.csv'}")
print(f"- {RESULTS_TABLES / '04_split_info.csv'}")
print(f"- {RESULTS_FIGURES / '04_confusion_matrix_best_balanceamento.png'}")
print(f"- {RESULTS_FIGURES / '04_confusion_matrix_best_balanceamento_normalized.png'}")
print(f"- {RESULTS_MODELS}")

print("\nAvaliação de estratégias de balanceamento finalizada com sucesso.")
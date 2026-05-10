# -*- coding: utf-8 -*-
"""
Script 05 - Calibração probabilística, limiares por classe e abstenção assistida.

Objetivo:
    - Utilizar a configuração líder obtida até a etapa 04:
        Modelo: XGBoost
        Atributos: CBC original + atributos hematológicos derivados
        Balanceamento: Borderline-SMOTE
    - Comparar:
        1. Sem calibração probabilística.
        2. Calibração sigmoid/Platt.
        3. Calibração isotônica.
    - Avaliar qualidade probabilística:
        - Brier score multiclasses.
        - Expected Calibration Error (ECE).
        - ROC-AUC macro one-vs-rest.
        - PR-AUC macro one-vs-rest.
    - Ajustar limiares específicos por classe usando apenas validação interna.
    - Avaliar abstenção assistida por:
        - Baixa probabilidade máxima.
        - Baixa margem entre top-1 e top-2.
    - Manter o conjunto de teste isolado da seleção de hiperparâmetros,
      calibração, limiares e critérios de abstenção.

Observação metodológica:
    - O conjunto de teste é usado apenas para avaliação final.
    - Os limiares por classe e critérios de abstenção são ajustados
      exclusivamente em validação interna extraída do conjunto de treinamento.
    - Borderline-SMOTE é aplicado apenas no subconjunto de treino de cada fold,
      dentro do pipeline.

Autor:
    Vinicius de Souza Santos
"""

from pathlib import Path
import warnings
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, label_binarize
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

try:
    from sklearn.calibration import CalibratedClassifierCV
except ImportError as erro:
    raise ImportError(
        "\n[ERRO] Não foi possível importar CalibratedClassifierCV.\n"
        "Verifique a instalação do scikit-learn.\n"
    ) from erro

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import BorderlineSMOTE
except ImportError as erro:
    raise ImportError(
        "\n[ERRO] O pacote imbalanced-learn não está instalado.\n"
        "Instale com:\n\n"
        "    pip install imbalanced-learn\n"
    ) from erro

try:
    from xgboost import XGBClassifier
except ImportError as erro:
    raise ImportError(
        "\n[ERRO] O pacote xgboost não está instalado.\n"
        "Instale com:\n\n"
        "    pip install xgboost\n"
    ) from erro

import matplotlib.pyplot as plt


# ============================================================
# 1. Configurações gerais
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
INTERNAL_VALIDATION_SIZE = 0.25

MIN_COVERAGE_ABSTENTION = 0.80
N_BINS_ECE = 10

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "AnemiaTypesClassification_data.csv"

RESULTS_TABLES = BASE_DIR / "results" / "tables"
RESULTS_FIGURES = BASE_DIR / "results" / "figures"
RESULTS_MODELS = BASE_DIR / "results" / "models"

RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS_MODELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Transformadores customizados
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
# 3. Wrapper seguro para XGBoost
# ============================================================

class SafeXGBClassifier(ClassifierMixin, BaseEstimator):
    """
    Wrapper para XGBClassifier.

    Finalidade:
        - Permitir uso dentro de pipelines, calibração e validação cruzada.
        - Mapear rótulos originais para 0..K-1 durante o fit e restaurar
          os rótulos originais no predict.
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
        tree_method="hist"
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

    def fit(self, X, y):
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

        self.estimator_.fit(X, y_mapped)

        return self

    def predict(self, X):
        y_pred_mapped = self.estimator_.predict(X)
        y_pred_mapped = np.asarray(y_pred_mapped).astype(int)
        return np.array([self.index_to_class_[idx] for idx in y_pred_mapped])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# ============================================================
# 4. Funções auxiliares de métricas e calibração
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


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Calcula ECE multiclasses com base na confiança máxima.

    Para cada amostra:
        confiança = maior probabilidade prevista;
        acerto = 1 se argmax coincide com y_true, 0 caso contrário.
    """

    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    rows = []

    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]

        if bin_idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        n_bin = int(mask.sum())

        if n_bin == 0:
            rows.append({
                "bin": bin_idx + 1,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "accuracy": np.nan,
                "confidence": np.nan,
                "gap": np.nan
            })
            continue

        acc_bin = float(np.mean(correctness[mask]))
        conf_bin = float(np.mean(confidences[mask]))
        gap = abs(acc_bin - conf_bin)

        ece += (n_bin / len(y_true)) * gap

        rows.append({
            "bin": bin_idx + 1,
            "lower": lower,
            "upper": upper,
            "n": n_bin,
            "accuracy": acc_bin,
            "confidence": conf_bin,
            "gap": gap
        })

    return float(ece), pd.DataFrame(rows)


def get_model_classes(model):
    """
    Recupera classes de um estimador, pipeline ou calibrador.
    """

    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)

    if hasattr(model, "named_steps"):
        final_model = model.named_steps.get("model")

        if final_model is not None and hasattr(final_model, "classes_"):
            return np.asarray(final_model.classes_)

    raise AttributeError("Não foi possível recuperar classes do modelo.")


def predict_proba_aligned(model, X, expected_classes):
    """
    Prediz probabilidades e alinha colunas com a ordem global de classes.
    """

    y_proba = model.predict_proba(X)
    model_classes = get_model_classes(model)

    aligned = np.zeros((y_proba.shape[0], len(expected_classes)))

    for idx_model, cls in enumerate(model_classes):
        idx_expected = np.where(expected_classes == cls)[0][0]
        aligned[:, idx_expected] = y_proba[:, idx_model]

    return aligned


def evaluate_predictions(y_true, y_pred, y_proba, classes):
    """
    Avalia predições discretas e probabilidades.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    try:
        metrics["roc_auc_macro_ovr"] = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
            labels=classes
        )
    except Exception:
        metrics["roc_auc_macro_ovr"] = np.nan

    try:
        metrics["pr_auc_macro_ovr"] = macro_average_precision(
            y_true,
            y_proba,
            classes
        )
    except Exception:
        metrics["pr_auc_macro_ovr"] = np.nan

    try:
        metrics["brier_multiclass"] = multiclass_brier_score(
            y_true,
            y_proba,
            classes
        )
    except Exception:
        metrics["brier_multiclass"] = np.nan

    try:
        metrics["ece"] = expected_calibration_error(
            y_true,
            y_proba,
            n_bins=N_BINS_ECE
        )[0]
    except Exception:
        metrics["ece"] = np.nan

    return metrics


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
# 5. Funções para limiares por classe e abstenção
# ============================================================

def tune_class_thresholds(y_true, y_proba, classes):
    """
    Ajusta limiares one-vs-rest por classe usando validação interna.

    Para cada classe, escolhe o limiar que maximiza F1 binário one-vs-rest.
    """

    thresholds = {}
    rows = []

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    for class_idx in classes:
        y_binary = (y_true == class_idx).astype(int)

        best_threshold = 0.50
        best_f1 = -1.0
        best_precision = np.nan
        best_recall = np.nan

        for threshold in grid:
            pred_binary = (y_proba[:, class_idx] >= threshold).astype(int)

            f1 = f1_score(y_binary, pred_binary, zero_division=0)
            precision = precision_score(y_binary, pred_binary, zero_division=0)
            recall = recall_score(y_binary, pred_binary, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        thresholds[class_idx] = best_threshold

        rows.append({
            "classe_codificada": class_idx,
            "threshold": best_threshold,
            "f1_ovr_validacao": best_f1,
            "precision_ovr_validacao": best_precision,
            "recall_ovr_validacao": best_recall
        })

    return thresholds, pd.DataFrame(rows)


def predict_with_class_thresholds(y_proba, thresholds):
    """
    Prediz usando limiares por classe.

    Regra:
        - Considera candidatas as classes cuja probabilidade >= limiar da classe.
        - Se houver candidatas, escolhe a candidata de maior probabilidade.
        - Se nenhuma classe atingir o limiar, usa argmax como fallback.
    """

    predictions = []

    threshold_array = np.array([
        thresholds[idx]
        for idx in range(y_proba.shape[1])
    ])

    for row in y_proba:
        candidates = row >= threshold_array

        if np.any(candidates):
            candidate_indices = np.where(candidates)[0]
            best_candidate = candidate_indices[np.argmax(row[candidate_indices])]
            predictions.append(best_candidate)
        else:
            predictions.append(int(np.argmax(row)))

    return np.asarray(predictions, dtype=int)


def evaluate_abstention(y_true, y_pred, accepted_mask, classes):
    """
    Avalia desempenho com abstenção assistida.
    """

    n_total = len(y_true)
    n_accepted = int(np.sum(accepted_mask))
    n_abstained = n_total - n_accepted

    coverage = n_accepted / n_total
    abstention_rate = n_abstained / n_total

    if n_accepted == 0:
        return {
            "n_total": n_total,
            "n_accepted": 0,
            "n_abstained": n_abstained,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "accuracy_accepted": np.nan,
            "error_accepted": np.nan,
            "f1_macro_accepted": np.nan,
            "recall_macro_accepted": np.nan,
            "precision_macro_accepted": np.nan
        }

    y_true_accepted = y_true[accepted_mask]
    y_pred_accepted = y_pred[accepted_mask]

    accuracy = accuracy_score(y_true_accepted, y_pred_accepted)

    return {
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_abstained": n_abstained,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "accuracy_accepted": accuracy,
        "error_accepted": 1.0 - accuracy,
        "f1_macro_accepted": f1_score(
            y_true_accepted,
            y_pred_accepted,
            labels=classes,
            average="macro",
            zero_division=0
        ),
        "recall_macro_accepted": recall_score(
            y_true_accepted,
            y_pred_accepted,
            labels=classes,
            average="macro",
            zero_division=0
        ),
        "precision_macro_accepted": precision_score(
            y_true_accepted,
            y_pred_accepted,
            labels=classes,
            average="macro",
            zero_division=0
        )
    }


def tune_abstention_max_probability(y_true, y_pred, y_proba, classes):
    """
    Ajusta limiar de abstenção por baixa probabilidade máxima.

    Aceita caso se:
        max_prob >= tau
    """

    rows = []
    max_prob = np.max(y_proba, axis=1)

    grid = np.round(np.arange(0.50, 0.96, 0.05), 2)

    for tau in grid:
        accepted_mask = max_prob >= tau
        metrics = evaluate_abstention(y_true, y_pred, accepted_mask, classes)

        rows.append({
            "politica": "probabilidade_maxima",
            "parametro": "tau",
            "valor": tau,
            **metrics
        })

    curve_df = pd.DataFrame(rows)

    feasible = curve_df[curve_df["coverage"] >= MIN_COVERAGE_ABSTENTION].copy()

    if feasible.empty:
        selected = curve_df.sort_values(
            by=["coverage", "f1_macro_accepted"],
            ascending=[False, False]
        ).iloc[0]
    else:
        selected = feasible.sort_values(
            by=["f1_macro_accepted", "error_accepted", "coverage"],
            ascending=[False, True, False]
        ).iloc[0]

    return float(selected["valor"]), curve_df


def tune_abstention_margin(y_true, y_pred, y_proba, classes):
    """
    Ajusta limiar de abstenção por margem top-1/top-2.

    Aceita caso se:
        p_top1 - p_top2 >= delta
    """

    rows = []

    sorted_proba = np.sort(y_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]

    grid = np.round(np.arange(0.00, 0.51, 0.05), 2)

    for delta in grid:
        accepted_mask = margin >= delta
        metrics = evaluate_abstention(y_true, y_pred, accepted_mask, classes)

        rows.append({
            "politica": "margem_top1_top2",
            "parametro": "delta",
            "valor": delta,
            **metrics
        })

    curve_df = pd.DataFrame(rows)

    feasible = curve_df[curve_df["coverage"] >= MIN_COVERAGE_ABSTENTION].copy()

    if feasible.empty:
        selected = curve_df.sort_values(
            by=["coverage", "f1_macro_accepted"],
            ascending=[False, False]
        ).iloc[0]
    else:
        selected = feasible.sort_values(
            by=["f1_macro_accepted", "error_accepted", "coverage"],
            ascending=[False, True, False]
        ).iloc[0]

    return float(selected["valor"]), curve_df


# ============================================================
# 6. Funções de pipeline e calibração
# ============================================================

def make_calibrated_classifier(estimator, method, cv):
    """
    Cria CalibratedClassifierCV de forma compatível com versões recentes
    e antigas do scikit-learn.
    """

    try:
        return CalibratedClassifierCV(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=-1
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=-1
        )


def build_pipeline():
    """
    Monta pipeline com:
        - limpeza clínica;
        - atributos derivados;
        - imputação;
        - winsorização;
        - Borderline-SMOTE;
        - XGBoost.
    """

    model = SafeXGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    steps = [
        ("clinical_cleaner", ClinicalRangeCleaner(bounds=clinical_bounds)),
        ("derived_features", DerivedHematologicalFeatures(add_features=True)),
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", IQRWinsorizer(factor=1.5)),
        (
            "resampler",
            BorderlineSMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=1,
                m_neighbors=3,
                kind="borderline-1"
            )
        ),
        ("model", model)
    ]

    return ImbPipeline(steps=steps)


def fit_scenario(estimator_template, scenario_name, X_fit, y_fit, cv_calibration):
    """
    Ajusta modelo conforme cenário de calibração.
    """

    if scenario_name == "sem_calibracao":
        model = clone(estimator_template)
        model.fit(X_fit, y_fit)
        return model

    if scenario_name == "sigmoid":
        base = clone(estimator_template)
        model = make_calibrated_classifier(
            estimator=base,
            method="sigmoid",
            cv=cv_calibration
        )
        model.fit(X_fit, y_fit)
        return model

    if scenario_name == "isotonica":
        base = clone(estimator_template)
        model = make_calibrated_classifier(
            estimator=base,
            method="isotonic",
            cv=cv_calibration
        )
        model.fit(X_fit, y_fit)
        return model

    raise ValueError(f"Cenário de calibração desconhecido: {scenario_name}")


def plot_reliability_diagram(bin_df, title, output_path):
    """
    Plota diagrama de confiabilidade multiclasses baseado na confiança máxima.
    """

    valid = bin_df.dropna(subset=["accuracy", "confidence"]).copy()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Calibração ideal")

    if not valid.empty:
        ax.plot(
            valid["confidence"],
            valid["accuracy"],
            marker="o",
            label="Modelo"
        )

    ax.set_title(title)
    ax.set_xlabel("Confiança média")
    ax.set_ylabel("Acurácia média")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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
    RESULTS_TABLES / "05_label_mapping.csv",
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
    RESULTS_TABLES / "05_class_distribution_dedup.csv",
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

X_train_core, X_valid_internal, y_train_core, y_valid_internal = train_test_split(
    X_train,
    y_train,
    test_size=INTERNAL_VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train
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
# 9. Seleção de hiperparâmetros no conjunto de treinamento
# ============================================================

print("\n================ SELEÇÃO DE HIPERPARÂMETROS ================\n")

base_pipeline = build_pipeline()

param_grid = {
    "model__n_estimators": [200, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

cv_search = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

grid = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_search,
    n_jobs=-1,
    refit=True,
    verbose=1,
    error_score="raise"
)

grid.fit(X_train, y_train)

best_params = grid.best_params_
best_cv_score = grid.best_score_

search_info = {
    "modelo": "XGBoost",
    "atributos": "CBC_original_plus_derivados",
    "balanceamento": "Borderline_SMOTE",
    "best_cv_f1_macro": best_cv_score,
    "best_params": json.dumps(best_params, ensure_ascii=False)
}

pd.DataFrame([search_info]).to_csv(
    RESULTS_TABLES / "05_hyperparam_search.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"Melhor F1 macro na CV interna: {best_cv_score:.4f}")
print(f"Melhores hiperparâmetros: {best_params}")

best_pipeline_template = clone(base_pipeline)
best_pipeline_template.set_params(**best_params)


# ============================================================
# 10. Avaliação de calibração, limiares e abstenção
# ============================================================

calibration_scenarios = {
    "sem_calibracao": "Sem calibração probabilística",
    "sigmoid": "Calibração sigmoid/Platt",
    "isotonica": "Calibração isotônica"
}

cv_calibration = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

calibration_results = []
threshold_results = []
threshold_rows_all = []
abstention_results = []
abstention_curve_rows = []

best_models = {}
test_probabilities = {}

for scenario_name, scenario_description in calibration_scenarios.items():
    print("\n" + "=" * 80)
    print(f"CENÁRIO DE CALIBRAÇÃO: {scenario_name}")
    print(scenario_description)
    print("=" * 80)

    # --------------------------------------------------------
    # 10.1 Modelo ajustado somente no treino-core para escolher
    #      limiares e critérios de abstenção na validação interna
    # --------------------------------------------------------

    validation_model = fit_scenario(
        estimator_template=best_pipeline_template,
        scenario_name=scenario_name,
        X_fit=X_train_core,
        y_fit=y_train_core,
        cv_calibration=cv_calibration
    )

    y_proba_valid = predict_proba_aligned(
        validation_model,
        X_valid_internal,
        classes_encoded
    )

    y_pred_valid_argmax = np.argmax(y_proba_valid, axis=1)

    class_thresholds, thresholds_df = tune_class_thresholds(
        y_true=y_valid_internal,
        y_proba=y_proba_valid,
        classes=classes_encoded
    )

    thresholds_df["cenario_calibracao"] = scenario_name
    thresholds_df["classe_original"] = thresholds_df["classe_codificada"].map(
        lambda idx: label_encoder.classes_[int(idx)]
    )

    threshold_rows_all.append(thresholds_df)

    y_pred_valid_threshold = predict_with_class_thresholds(
        y_proba_valid,
        class_thresholds
    )

    tau_max, max_prob_curve = tune_abstention_max_probability(
        y_true=y_valid_internal,
        y_pred=y_pred_valid_argmax,
        y_proba=y_proba_valid,
        classes=classes_encoded
    )

    delta_margin, margin_curve = tune_abstention_margin(
        y_true=y_valid_internal,
        y_pred=y_pred_valid_argmax,
        y_proba=y_proba_valid,
        classes=classes_encoded
    )

    max_prob_curve["cenario_calibracao"] = scenario_name
    margin_curve["cenario_calibracao"] = scenario_name

    abstention_curve_rows.append(max_prob_curve)
    abstention_curve_rows.append(margin_curve)

    # --------------------------------------------------------
    # 10.2 Modelo final ajustado em todo o conjunto de treino
    #      para avaliação única no teste isolado
    # --------------------------------------------------------

    final_model = fit_scenario(
        estimator_template=best_pipeline_template,
        scenario_name=scenario_name,
        X_fit=X_train,
        y_fit=y_train,
        cv_calibration=cv_calibration
    )

    best_models[scenario_name] = final_model

    model_path = RESULTS_MODELS / f"05_model_{safe_filename(scenario_name)}.joblib"
    joblib.dump(final_model, model_path)

    y_proba_test = predict_proba_aligned(
        final_model,
        X_test,
        classes_encoded
    )

    test_probabilities[scenario_name] = y_proba_test

    y_pred_test_argmax = np.argmax(y_proba_test, axis=1)

    metrics_argmax = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred_test_argmax,
        y_proba=y_proba_test,
        classes=classes_encoded
    )

    ece_value, reliability_df = expected_calibration_error(
        y_true=y_test,
        y_proba=y_proba_test,
        n_bins=N_BINS_ECE
    )

    reliability_df["cenario_calibracao"] = scenario_name
    reliability_df.to_csv(
        RESULTS_TABLES / f"05_reliability_bins_{safe_filename(scenario_name)}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_reliability_diagram(
        bin_df=reliability_df,
        title=f"Diagrama de calibração - {scenario_name}",
        output_path=RESULTS_FIGURES / f"05_reliability_{safe_filename(scenario_name)}.png"
    )

    calibration_results.append({
        "cenario_calibracao": scenario_name,
        "descricao": scenario_description,
        "accuracy": metrics_argmax["accuracy"],
        "balanced_accuracy": metrics_argmax["balanced_accuracy"],
        "precision_macro": metrics_argmax["precision_macro"],
        "recall_macro": metrics_argmax["recall_macro"],
        "f1_macro": metrics_argmax["f1_macro"],
        "f1_weighted": metrics_argmax["f1_weighted"],
        "roc_auc_macro_ovr": metrics_argmax["roc_auc_macro_ovr"],
        "pr_auc_macro_ovr": metrics_argmax["pr_auc_macro_ovr"],
        "brier_multiclass": metrics_argmax["brier_multiclass"],
        "ece": ece_value
    })

    # --------------------------------------------------------
    # 10.3 Avaliação com limiares por classe no teste
    # --------------------------------------------------------

    y_pred_test_threshold = predict_with_class_thresholds(
        y_proba_test,
        class_thresholds
    )

    metrics_threshold = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred_test_threshold,
        y_proba=y_proba_test,
        classes=classes_encoded
    )

    threshold_results.append({
        "cenario_calibracao": scenario_name,
        "accuracy": metrics_threshold["accuracy"],
        "balanced_accuracy": metrics_threshold["balanced_accuracy"],
        "precision_macro": metrics_threshold["precision_macro"],
        "recall_macro": metrics_threshold["recall_macro"],
        "f1_macro": metrics_threshold["f1_macro"],
        "f1_weighted": metrics_threshold["f1_weighted"],
        "roc_auc_macro_ovr": metrics_threshold["roc_auc_macro_ovr"],
        "pr_auc_macro_ovr": metrics_threshold["pr_auc_macro_ovr"],
        "brier_multiclass": metrics_threshold["brier_multiclass"],
        "ece": metrics_threshold["ece"],
        "thresholds": json.dumps(
            {
                label_encoder.classes_[idx]: class_thresholds[idx]
                for idx in class_thresholds
            },
            ensure_ascii=False
        )
    })

    report_threshold = classification_report(
        y_test,
        y_pred_test_threshold,
        labels=classes_encoded,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_threshold).transpose().to_csv(
        RESULTS_TABLES / f"05_classification_report_thresholds_{safe_filename(scenario_name)}.csv",
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 10.4 Avaliação da abstenção no teste
    # --------------------------------------------------------

    max_prob_test = np.max(y_proba_test, axis=1)

    sorted_proba_test = np.sort(y_proba_test, axis=1)
    margin_test = sorted_proba_test[:, -1] - sorted_proba_test[:, -2]

    accepted_max_prob = max_prob_test >= tau_max
    metrics_abst_max = evaluate_abstention(
        y_true=y_test,
        y_pred=y_pred_test_argmax,
        accepted_mask=accepted_max_prob,
        classes=classes_encoded
    )

    abstention_results.append({
        "cenario_calibracao": scenario_name,
        "politica": "probabilidade_maxima",
        "parametro": "tau",
        "valor_escolhido_validacao": tau_max,
        **metrics_abst_max
    })

    accepted_margin = margin_test >= delta_margin
    metrics_abst_margin = evaluate_abstention(
        y_true=y_test,
        y_pred=y_pred_test_argmax,
        accepted_mask=accepted_margin,
        classes=classes_encoded
    )

    abstention_results.append({
        "cenario_calibracao": scenario_name,
        "politica": "margem_top1_top2",
        "parametro": "delta",
        "valor_escolhido_validacao": delta_margin,
        **metrics_abst_margin
    })

    # --------------------------------------------------------
    # 10.5 Matrizes de confusão por cenário
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred_test_argmax,
        labels=classes_encoded
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_df.to_csv(
        RESULTS_TABLES / f"05_confusion_matrix_argmax_{safe_filename(scenario_name)}.csv",
        encoding="utf-8-sig"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, xticks_rotation=90, values_format="d")
    ax.set_title(f"Matriz de confusão - argmax - {scenario_name}")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES / f"05_confusion_matrix_argmax_{safe_filename(scenario_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    cm_norm = confusion_matrix(
        y_test,
        y_pred_test_argmax,
        labels=classes_encoded,
        normalize="true"
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_norm_df.to_csv(
        RESULTS_TABLES / f"05_confusion_matrix_argmax_normalized_{safe_filename(scenario_name)}.csv",
        encoding="utf-8-sig"
    )

    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp_norm.plot(ax=ax, xticks_rotation=90, values_format=".2f")
    ax.set_title(f"Matriz de confusão normalizada - argmax - {scenario_name}")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES / f"05_confusion_matrix_argmax_normalized_{safe_filename(scenario_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"F1 macro argmax no teste: {metrics_argmax['f1_macro']:.4f}")
    print(f"Brier no teste: {metrics_argmax['brier_multiclass']:.4f}")
    print(f"ECE no teste: {ece_value:.4f}")
    print(f"F1 macro com limiares no teste: {metrics_threshold['f1_macro']:.4f}")
    print(f"Abstenção por prob. máxima: tau={tau_max:.2f}, cobertura={metrics_abst_max['coverage']:.4f}")
    print(f"Abstenção por margem: delta={delta_margin:.2f}, cobertura={metrics_abst_margin['coverage']:.4f}")


# ============================================================
# 11. Salvamento das tabelas finais
# ============================================================

calibration_df = pd.DataFrame(calibration_results).sort_values(
    by=["f1_macro", "brier_multiclass"],
    ascending=[False, True]
).reset_index(drop=True)

calibration_df.to_csv(
    RESULTS_TABLES / "05_calibracao_resultados.csv",
    index=False,
    encoding="utf-8-sig"
)

thresholds_all_df = pd.concat(threshold_rows_all, ignore_index=True)

thresholds_all_df.to_csv(
    RESULTS_TABLES / "05_thresholds_por_classe.csv",
    index=False,
    encoding="utf-8-sig"
)

threshold_results_df = pd.DataFrame(threshold_results).sort_values(
    by=["f1_macro", "brier_multiclass"],
    ascending=[False, True]
).reset_index(drop=True)

threshold_results_df.to_csv(
    RESULTS_TABLES / "05_limiares_resultados.csv",
    index=False,
    encoding="utf-8-sig"
)

abstention_curves_df = pd.concat(abstention_curve_rows, ignore_index=True)

abstention_curves_df.to_csv(
    RESULTS_TABLES / "05_abstencao_curvas_validacao.csv",
    index=False,
    encoding="utf-8-sig"
)

abstention_results_df = pd.DataFrame(abstention_results).sort_values(
    by=["f1_macro_accepted", "coverage"],
    ascending=[False, False]
).reset_index(drop=True)

abstention_results_df.to_csv(
    RESULTS_TABLES / "05_abstencao_resultados.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. Figuras comparativas
# ============================================================

# Figura: calibração por métrica probabilística
plot_calib = calibration_df.set_index("cenario_calibracao")[
    [
        "f1_macro",
        "recall_macro",
        "brier_multiclass",
        "ece"
    ]
]

ax = plot_calib.plot(kind="bar", figsize=(10, 6))
ax.set_title("Comparação entre cenários de calibração")
ax.set_ylabel("Valor da métrica")
ax.set_xlabel("Cenário de calibração")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "05_comparacao_calibracao_metricas.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Figura: limiares por classe
for scenario_name in calibration_scenarios:
    scenario_thresholds = thresholds_all_df[
        thresholds_all_df["cenario_calibracao"] == scenario_name
    ].copy()

    scenario_thresholds = scenario_thresholds.sort_values(
        by="threshold",
        ascending=False
    )

    ax = scenario_thresholds.set_index("classe_original")["threshold"].plot(
        kind="bar",
        figsize=(11, 5)
    )

    ax.set_title(f"Limiares por classe - {scenario_name}")
    ax.set_ylabel("Limiar")
    ax.set_xlabel("Classe")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURES / f"05_thresholds_por_classe_{safe_filename(scenario_name)}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# Figura: abstenção
for scenario_name in calibration_scenarios:
    scenario_abst = abstention_curves_df[
        abstention_curves_df["cenario_calibracao"] == scenario_name
    ].copy()

    for politica in scenario_abst["politica"].unique():
        policy_df = scenario_abst[
            scenario_abst["politica"] == politica
        ].copy()

        fig, ax1 = plt.subplots(figsize=(9, 5))

        ax1.plot(
            policy_df["valor"],
            policy_df["coverage"],
            marker="o",
            label="Cobertura"
        )

        ax1.plot(
            policy_df["valor"],
            policy_df["f1_macro_accepted"],
            marker="s",
            label="F1 macro aceitos"
        )

        ax1.set_title(f"Curva de abstenção - {scenario_name} - {politica}")
        ax1.set_xlabel("Valor do critério")
        ax1.set_ylabel("Métrica")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS_FIGURES /
            f"05_abstencao_{safe_filename(scenario_name)}_{safe_filename(politica)}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 13. Seleção de melhor configuração operacional
# ============================================================

best_calibration_row = calibration_df.iloc[0]
best_threshold_row = threshold_results_df.iloc[0]
best_abstention_row = abstention_results_df.iloc[0]

summary_rows = [
    {
        "tipo": "argmax",
        "cenario_calibracao": best_calibration_row["cenario_calibracao"],
        "criterio": "maior_f1_macro_argmax",
        "f1_macro": best_calibration_row["f1_macro"],
        "recall_macro": best_calibration_row["recall_macro"],
        "brier_multiclass": best_calibration_row["brier_multiclass"],
        "ece": best_calibration_row["ece"],
        "coverage": 1.0,
        "abstention_rate": 0.0
    },
    {
        "tipo": "limiares_por_classe",
        "cenario_calibracao": best_threshold_row["cenario_calibracao"],
        "criterio": "maior_f1_macro_com_limiares",
        "f1_macro": best_threshold_row["f1_macro"],
        "recall_macro": best_threshold_row["recall_macro"],
        "brier_multiclass": best_threshold_row["brier_multiclass"],
        "ece": best_threshold_row["ece"],
        "coverage": 1.0,
        "abstention_rate": 0.0
    },
    {
        "tipo": "abstencao_assistida",
        "cenario_calibracao": best_abstention_row["cenario_calibracao"],
        "criterio": best_abstention_row["politica"],
        "f1_macro": best_abstention_row["f1_macro_accepted"],
        "recall_macro": best_abstention_row["recall_macro_accepted"],
        "brier_multiclass": np.nan,
        "ece": np.nan,
        "coverage": best_abstention_row["coverage"],
        "abstention_rate": best_abstention_row["abstention_rate"]
    }
]

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULTS_TABLES / "05_resumo_operacional.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. Salvamento do LabelEncoder e informações da execução
# ============================================================

joblib.dump(label_encoder, RESULTS_MODELS / "05_label_encoder.joblib")

split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "internal_validation_size_from_train": INTERNAL_VALIDATION_SIZE,
    "n_total_original": len(df_raw),
    "n_total_deduplicado": len(df),
    "n_train": len(X_train),
    "n_train_core": len(X_train_core),
    "n_valid_internal": len(X_valid_internal),
    "n_test": len(X_test),
    "modelo": "XGBoost",
    "cenario_atributos": "CBC_original_plus_atributos_derivados",
    "balanceamento": "Borderline_SMOTE",
    "criterio_hiperparametros": "f1_macro",
    "best_cv_f1_macro": best_cv_score,
    "best_params": json.dumps(best_params, ensure_ascii=False),
    "min_coverage_abstention": MIN_COVERAGE_ABSTENTION,
    "n_bins_ece": N_BINS_ECE
}

pd.DataFrame([split_info]).to_csv(
    RESULTS_TABLES / "05_split_info.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Encerramento
# ============================================================

print("\n================ RESULTADOS - CALIBRAÇÃO ================\n")
print(
    calibration_df[
        [
            "cenario_calibracao",
            "accuracy",
            "f1_macro",
            "recall_macro",
            "f1_weighted",
            "brier_multiclass",
            "ece",
            "roc_auc_macro_ovr",
            "pr_auc_macro_ovr"
        ]
    ].to_string(index=False)
)

print("\n================ RESULTADOS - LIMIARES POR CLASSE ================\n")
print(
    threshold_results_df[
        [
            "cenario_calibracao",
            "accuracy",
            "f1_macro",
            "recall_macro",
            "f1_weighted",
            "brier_multiclass",
            "ece"
        ]
    ].to_string(index=False)
)

print("\n================ RESULTADOS - ABSTENÇÃO ASSISTIDA ================\n")
print(
    abstention_results_df[
        [
            "cenario_calibracao",
            "politica",
            "valor_escolhido_validacao",
            "coverage",
            "abstention_rate",
            "accuracy_accepted",
            "error_accepted",
            "f1_macro_accepted",
            "recall_macro_accepted"
        ]
    ].to_string(index=False)
)

print("\n================ RESUMO OPERACIONAL ================\n")
print(summary_df.to_string(index=False))

print("\nArquivos gerados:")
print(f"- {RESULTS_TABLES / '05_hyperparam_search.csv'}")
print(f"- {RESULTS_TABLES / '05_calibracao_resultados.csv'}")
print(f"- {RESULTS_TABLES / '05_thresholds_por_classe.csv'}")
print(f"- {RESULTS_TABLES / '05_limiares_resultados.csv'}")
print(f"- {RESULTS_TABLES / '05_abstencao_curvas_validacao.csv'}")
print(f"- {RESULTS_TABLES / '05_abstencao_resultados.csv'}")
print(f"- {RESULTS_TABLES / '05_resumo_operacional.csv'}")
print(f"- {RESULTS_TABLES / '05_split_info.csv'}")
print(f"- {RESULTS_FIGURES / '05_comparacao_calibracao_metricas.png'}")
print(f"- {RESULTS_MODELS}")

print("\nCalibração, limiares e abstenção finalizados com sucesso.")
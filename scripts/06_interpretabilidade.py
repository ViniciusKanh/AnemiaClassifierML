# -*- coding: utf-8 -*-
"""
Script 06 - Interpretabilidade do modelo líder.

Objetivo:
    - Treinar a configuração líder obtida nas etapas anteriores:
        Modelo: XGBoost
        Atributos: CBC original + atributos hematológicos derivados
        Balanceamento: Borderline-SMOTE
    - Avaliar interpretabilidade por:
        1. Importância por permutação no conjunto de teste.
        2. Importância nativa do XGBoost.
        3. SHAP global, quando o pacote shap estiver instalado.
        4. SHAP por classe, quando aplicável.
    - Gerar tabelas e figuras para uso na seção de Resultados e Discussão.

Observação metodológica:
    - O modelo é ajustado apenas com o conjunto de treinamento.
    - A interpretabilidade é calculada no conjunto de teste isolado.
    - A reamostragem Borderline-SMOTE ocorre apenas no treinamento, dentro do pipeline.
    - Os atributos derivados são criados dentro do pipeline.
    - O objetivo é apoiar a análise de plausibilidade clínica, não produzir regra diagnóstica definitiva.

Autor:
    Vinicius de Souza Santos
"""

from pathlib import Path
import warnings
import json
import ast
import joblib

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

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

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print(
        "\n[AVISO] Pacote shap não encontrado. "
        "A análise SHAP será ignorada.\n"
        "Para instalar, use:\n\n"
        "    pip install shap\n"
    )

import matplotlib.pyplot as plt


# ============================================================
# 1. Configurações gerais
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

SHAP_MAX_TEST_SAMPLES = 247
SHAP_MAX_BACKGROUND_SAMPLES = 200
PERMUTATION_REPEATS = 30
TOP_N_FEATURES = 20

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

    Os limites são calculados apenas no conjunto de treino.
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
        - Permitir uso dentro de pipelines.
        - Mapear rótulos originais para 0..K-1 durante o fit.
        - Restaurar os rótulos originais no predict.
    """

    def __init__(
        self,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
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
# 4. Funções auxiliares
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


def evaluate_model(model, X_test, y_test, classes):
    """
    Avalia modelo no conjunto de teste.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

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

    return metrics, y_pred, y_proba


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


def load_best_params_from_stage_05():
    """
    Tenta carregar hiperparâmetros da etapa 05.

    Caso o arquivo não exista ou não possa ser lido, usa os parâmetros
    líderes observados na execução anterior.
    """

    default_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    search_path = RESULTS_TABLES / "05_hyperparam_search.csv"

    if not search_path.exists():
        print(
            "\n[AVISO] Arquivo 05_hyperparam_search.csv não encontrado. "
            "Usando hiperparâmetros padrão da configuração líder.\n"
        )
        return default_params

    try:
        df_search = pd.read_csv(search_path)
        raw_params = df_search.loc[0, "best_params"]

        try:
            parsed = json.loads(raw_params)
        except Exception:
            parsed = ast.literal_eval(raw_params)

        params = default_params.copy()

        for key, value in parsed.items():
            clean_key = key.replace("model__", "")
            if clean_key in params:
                params[clean_key] = value

        print("\nHiperparâmetros carregados da etapa 05:")
        print(params)

        return params

    except Exception as erro:
        print(
            "\n[AVISO] Não foi possível ler os hiperparâmetros da etapa 05. "
            f"Motivo: {erro}\n"
            "Usando hiperparâmetros padrão da configuração líder.\n"
        )
        return default_params


def get_preprocessed_features(fitted_pipeline, X):
    """
    Aplica as etapas de pré-processamento ajustadas no pipeline,
    sem aplicar o reamostrador.

    Retorna:
        - DataFrame com atributos após limpeza, derivados, imputação e winsorização.
        - Lista de nomes dos atributos.
    """

    clinical_cleaner = fitted_pipeline.named_steps["clinical_cleaner"]
    derived_features = fitted_pipeline.named_steps["derived_features"]
    imputer = fitted_pipeline.named_steps["imputer"]
    winsorizer = fitted_pipeline.named_steps["winsorizer"]

    X_clean = clinical_cleaner.transform(X)
    X_derived = derived_features.transform(X_clean)

    if not isinstance(X_derived, pd.DataFrame):
        raise ValueError("A etapa de atributos derivados deveria retornar DataFrame.")

    feature_names = list(X_derived.columns)

    X_imputed = imputer.transform(X_derived)
    X_winsorized = winsorizer.transform(X_imputed)

    X_preprocessed = pd.DataFrame(
        X_winsorized,
        columns=feature_names,
        index=X.index if isinstance(X, pd.DataFrame) else None
    )

    return X_preprocessed, feature_names


def plot_horizontal_importance(df, value_col, feature_col, title, output_path, top_n=20):
    """
    Gera gráfico horizontal de importância de atributos.
    """

    plot_df = df.sort_values(value_col, ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df[feature_col], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("Atributo")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def normalize_shap_values(shap_values, n_classes):
    """
    Normaliza diferentes formatos de saída do SHAP para matriz:
        n_amostras x n_atributos x n_classes
    """

    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=2)

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        # Possíveis formatos:
        # 1. n_amostras x n_atributos x n_classes
        # 2. n_classes x n_amostras x n_atributos
        if shap_array.shape[2] == n_classes:
            return shap_array
        if shap_array.shape[0] == n_classes:
            return np.transpose(shap_array, (1, 2, 0))

    if shap_array.ndim == 2:
        # Caso binário ou retorno agregado.
        return shap_array[:, :, np.newaxis]

    raise ValueError(
        f"Formato inesperado de SHAP values: shape={shap_array.shape}"
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
    RESULTS_TABLES / "06_label_mapping.csv",
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
# 6. Limites clínicos amplos
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
# 7. Construção e treinamento do modelo líder
# ============================================================

best_params = load_best_params_from_stage_05()

model = SafeXGBClassifier(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

pipeline = ImbPipeline(
    steps=[
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
)

print("\n================ TREINANDO MODELO LÍDER PARA INTERPRETABILIDADE ================\n")

pipeline.fit(X_train, y_train)

joblib.dump(
    pipeline,
    RESULTS_MODELS / "06_modelo_lider_xgboost_borderline_smote.joblib"
)


# ============================================================
# 8. Avaliação do modelo no teste
# ============================================================

metrics, y_pred, y_proba = evaluate_model(
    pipeline,
    X_test,
    y_test,
    classes_encoded
)

metrics_df = pd.DataFrame([{
    "modelo": "XGBoost",
    "atributos": "CBC_original_plus_derivados",
    "balanceamento": "Borderline_SMOTE",
    **metrics
}])

metrics_df.to_csv(
    RESULTS_TABLES / "06_modelo_lider_metricas.csv",
    index=False,
    encoding="utf-8-sig"
)

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
    RESULTS_TABLES / "06_classification_report_modelo_lider.csv",
    encoding="utf-8-sig"
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes_encoded
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_df.to_csv(
    RESULTS_TABLES / "06_confusion_matrix_modelo_lider.csv",
    encoding="utf-8-sig"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(11, 9))
disp.plot(ax=ax, xticks_rotation=90, values_format="d")
ax.set_title("Matriz de confusão - modelo líder")
plt.tight_layout()

plt.savefig(
    RESULTS_FIGURES / "06_confusion_matrix_modelo_lider.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Métricas do modelo líder no teste:")
for key, value in metrics.items():
    print(f"- {key}: {value:.4f}")


# ============================================================
# 9. Preparação da matriz transformada para interpretabilidade
# ============================================================

X_train_processed, feature_names = get_preprocessed_features(pipeline, X_train)
X_test_processed, _ = get_preprocessed_features(pipeline, X_test)

final_model = pipeline.named_steps["model"]

feature_info = pd.DataFrame({
    "feature": feature_names,
    "tipo": [
        "derivado" if feature in [
            "Mentzer_Index",
            "HGB_RBC_ratio",
            "HCT_RBC_ratio",
            "PLT_RBC_ratio",
            "NLR",
            "PLR",
            "WBC_RBC_ratio",
            "PLT_WBC_ratio"
        ] else "original"
        for feature in feature_names
    ]
})

feature_info.to_csv(
    RESULTS_TABLES / "06_feature_names_pos_preprocessamento.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Importância por permutação
# ============================================================

print("\n================ IMPORTÂNCIA POR PERMUTAÇÃO ================\n")

permutation_result = permutation_importance(
    estimator=final_model,
    X=X_test_processed,
    y=y_test,
    scoring="f1_macro",
    n_repeats=PERMUTATION_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std
})

permutation_df = permutation_df.merge(
    feature_info,
    on="feature",
    how="left"
)

permutation_df = permutation_df.sort_values(
    by="importance_mean",
    ascending=False
).reset_index(drop=True)

permutation_df.to_csv(
    RESULTS_TABLES / "06_permutation_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

plot_horizontal_importance(
    df=permutation_df,
    value_col="importance_mean",
    feature_col="feature",
    title="Importância por permutação - F1 macro",
    output_path=RESULTS_FIGURES / "06_permutation_importance_top20.png",
    top_n=TOP_N_FEATURES
)

print("Top 10 atributos por importância por permutação:")
print(
    permutation_df[
        ["feature", "tipo", "importance_mean", "importance_std"]
    ].head(10).to_string(index=False)
)


# ============================================================
# 11. Importância nativa do XGBoost
# ============================================================

print("\n================ IMPORTÂNCIA NATIVA DO XGBOOST ================\n")

native_importance = final_model.estimator_.feature_importances_

native_df = pd.DataFrame({
    "feature": feature_names,
    "xgb_feature_importance": native_importance
})

native_df = native_df.merge(
    feature_info,
    on="feature",
    how="left"
)

native_df = native_df.sort_values(
    by="xgb_feature_importance",
    ascending=False
).reset_index(drop=True)

native_df.to_csv(
    RESULTS_TABLES / "06_xgboost_native_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

plot_horizontal_importance(
    df=native_df,
    value_col="xgb_feature_importance",
    feature_col="feature",
    title="Importância nativa do XGBoost",
    output_path=RESULTS_FIGURES / "06_xgboost_native_importance_top20.png",
    top_n=TOP_N_FEATURES
)

print("Top 10 atributos por importância nativa do XGBoost:")
print(
    native_df[
        ["feature", "tipo", "xgb_feature_importance"]
    ].head(10).to_string(index=False)
)


# ============================================================
# 12. SHAP global e por classe
# ============================================================

if HAS_SHAP:
    print("\n================ ANÁLISE SHAP ================\n")

    rng = np.random.default_rng(RANDOM_STATE)

    n_test_sample = min(SHAP_MAX_TEST_SAMPLES, len(X_test_processed))
    n_background_sample = min(SHAP_MAX_BACKGROUND_SAMPLES, len(X_train_processed))

    test_indices = rng.choice(
        len(X_test_processed),
        size=n_test_sample,
        replace=False
    )

    background_indices = rng.choice(
        len(X_train_processed),
        size=n_background_sample,
        replace=False
    )

    X_shap = X_test_processed.iloc[test_indices].copy()
    X_background = X_train_processed.iloc[background_indices].copy()

    try:
        explainer = shap.TreeExplainer(final_model.estimator_)
        raw_shap_values = explainer.shap_values(X_shap)

        shap_values = normalize_shap_values(
            raw_shap_values,
            n_classes=len(label_encoder.classes_)
        )

        mean_abs_global = np.mean(np.abs(shap_values), axis=(0, 2))

        shap_global_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_global
        })

        shap_global_df = shap_global_df.merge(
            feature_info,
            on="feature",
            how="left"
        )

        shap_global_df = shap_global_df.sort_values(
            by="mean_abs_shap",
            ascending=False
        ).reset_index(drop=True)

        shap_global_df.to_csv(
            RESULTS_TABLES / "06_shap_global_importance.csv",
            index=False,
            encoding="utf-8-sig"
        )

        plot_horizontal_importance(
            df=shap_global_df,
            value_col="mean_abs_shap",
            feature_col="feature",
            title="Importância global por SHAP",
            output_path=RESULTS_FIGURES / "06_shap_global_importance_top20.png",
            top_n=TOP_N_FEATURES
        )

        print("Top 10 atributos por SHAP global:")
        print(
            shap_global_df[
                ["feature", "tipo", "mean_abs_shap"]
            ].head(10).to_string(index=False)
        )

        # ----------------------------------------------------
        # SHAP por classe
        # ----------------------------------------------------

        per_class_rows = []

        for class_idx, class_name in enumerate(label_encoder.classes_):
            shap_class = shap_values[:, :, class_idx]
            mean_abs_class = np.mean(np.abs(shap_class), axis=0)

            class_df = pd.DataFrame({
                "classe_codificada": class_idx,
                "classe_original": class_name,
                "feature": feature_names,
                "mean_abs_shap": mean_abs_class
            })

            class_df = class_df.merge(
                feature_info,
                on="feature",
                how="left"
            )

            class_df = class_df.sort_values(
                by="mean_abs_shap",
                ascending=False
            ).reset_index(drop=True)

            per_class_rows.append(class_df)

            plot_horizontal_importance(
                df=class_df,
                value_col="mean_abs_shap",
                feature_col="feature",
                title=f"SHAP por classe - {class_name}",
                output_path=(
                    RESULTS_FIGURES /
                    f"06_shap_por_classe_{safe_filename(class_name)}.png"
                ),
                top_n=10
            )

        shap_class_df = pd.concat(per_class_rows, ignore_index=True)

        shap_class_df.to_csv(
            RESULTS_TABLES / "06_shap_por_classe.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # ----------------------------------------------------
        # Tabela compacta das 5 variáveis mais relevantes por classe
        # ----------------------------------------------------

        top5_by_class = (
            shap_class_df
            .sort_values(
                by=["classe_original", "mean_abs_shap"],
                ascending=[True, False]
            )
            .groupby("classe_original")
            .head(5)
            .reset_index(drop=True)
        )

        top5_by_class.to_csv(
            RESULTS_TABLES / "06_shap_top5_por_classe.csv",
            index=False,
            encoding="utf-8-sig"
        )

        # ----------------------------------------------------
        # Figura tipo summary plot quando possível
        # ----------------------------------------------------

        try:
            plt.figure()

            if isinstance(raw_shap_values, list):
                shap.summary_plot(
                    raw_shap_values,
                    X_shap,
                    feature_names=feature_names,
                    show=False,
                    max_display=TOP_N_FEATURES
                )
            else:
                shap.summary_plot(
                    shap_values,
                    X_shap,
                    feature_names=feature_names,
                    show=False,
                    max_display=TOP_N_FEATURES
                )

            plt.tight_layout()

            plt.savefig(
                RESULTS_FIGURES / "06_shap_summary_plot.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        except Exception as erro:
            print(
                "\n[AVISO] Não foi possível gerar o summary plot do SHAP.\n"
                f"Motivo: {erro}\n"
            )

    except Exception as erro:
        print(
            "\n[AVISO] Falha ao executar SHAP. "
            "As demais análises foram preservadas.\n"
            f"Motivo: {erro}\n"
        )

        pd.DataFrame([{
            "status": "erro",
            "erro": str(erro)
        }]).to_csv(
            RESULTS_TABLES / "06_shap_error.csv",
            index=False,
            encoding="utf-8-sig"
        )

else:
    pd.DataFrame([{
        "status": "shap_nao_instalado",
        "mensagem": "Instale com: pip install shap"
    }]).to_csv(
        RESULTS_TABLES / "06_shap_status.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 13. Consolidação interpretativa para o artigo
# ============================================================

summary_rows = []

for source_name, df_importance, value_col in [
    ("permutation", permutation_df, "importance_mean"),
    ("xgboost_native", native_df, "xgb_feature_importance")
]:
    top_features = (
        df_importance
        .sort_values(value_col, ascending=False)
        .head(10)
    )

    for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
        summary_rows.append({
            "fonte": source_name,
            "rank": rank,
            "feature": row["feature"],
            "tipo": row["tipo"],
            "valor": row[value_col]
        })

if HAS_SHAP and (RESULTS_TABLES / "06_shap_global_importance.csv").exists():
    shap_global_loaded = pd.read_csv(
        RESULTS_TABLES / "06_shap_global_importance.csv"
    )

    top_features = (
        shap_global_loaded
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
    )

    for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
        summary_rows.append({
            "fonte": "shap_global",
            "rank": rank,
            "feature": row["feature"],
            "tipo": row["tipo"],
            "valor": row["mean_abs_shap"]
        })

summary_importance_df = pd.DataFrame(summary_rows)

summary_importance_df.to_csv(
    RESULTS_TABLES / "06_resumo_top_atributos.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. Informações da execução
# ============================================================

split_info = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_total_original": len(df_raw),
    "n_total_deduplicado": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "modelo": "XGBoost",
    "atributos": "CBC_original_plus_derivados",
    "balanceamento": "Borderline_SMOTE",
    "interpretabilidade": "permutation_importance; xgboost_native_importance; SHAP_se_disponivel",
    "shap_disponivel": HAS_SHAP,
    "permutation_repeats": PERMUTATION_REPEATS,
    "best_params": json.dumps(best_params, ensure_ascii=False)
}

pd.DataFrame([split_info]).to_csv(
    RESULTS_TABLES / "06_split_info.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. Encerramento
# ============================================================

print("\nArquivos gerados:")
print(f"- {RESULTS_TABLES / '06_modelo_lider_metricas.csv'}")
print(f"- {RESULTS_TABLES / '06_classification_report_modelo_lider.csv'}")
print(f"- {RESULTS_TABLES / '06_feature_names_pos_preprocessamento.csv'}")
print(f"- {RESULTS_TABLES / '06_permutation_importance.csv'}")
print(f"- {RESULTS_TABLES / '06_xgboost_native_importance.csv'}")
print(f"- {RESULTS_TABLES / '06_resumo_top_atributos.csv'}")
print(f"- {RESULTS_TABLES / '06_split_info.csv'}")
print(f"- {RESULTS_FIGURES / '06_permutation_importance_top20.png'}")
print(f"- {RESULTS_FIGURES / '06_xgboost_native_importance_top20.png'}")

if HAS_SHAP:
    print(f"- {RESULTS_TABLES / '06_shap_global_importance.csv'}")
    print(f"- {RESULTS_TABLES / '06_shap_por_classe.csv'}")
    print(f"- {RESULTS_TABLES / '06_shap_top5_por_classe.csv'}")
    print(f"- {RESULTS_FIGURES / '06_shap_global_importance_top20.png'}")
    print(f"- {RESULTS_FIGURES / '06_shap_summary_plot.png'}")
else:
    print(f"- {RESULTS_TABLES / '06_shap_status.csv'}")

print(f"- {RESULTS_MODELS / '06_modelo_lider_xgboost_borderline_smote.joblib'}")

print("\nInterpretabilidade finalizada com sucesso.")
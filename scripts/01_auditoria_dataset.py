# -*- coding: utf-8 -*-
"""
Script 01 - Auditoria inicial do conjunto de dados de anemia.

Objetivo:
    - Ler a base original.
    - Verificar dimensões, tipos, valores ausentes e duplicatas.
    - Gerar distribuição das classes.
    - Gerar estatísticas descritivas.
    - Identificar possíveis valores extremos ou clinicamente suspeitos.
    - Salvar tabelas em results/tables.

Autor:
    Vinicius de Souza Santos
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. Configurações gerais
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "AnemiaTypesClassification_data.csv"
RESULTS_TABLES = BASE_DIR / "results" / "tables"

RESULTS_TABLES.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Leitura da base
# ============================================================

df = pd.read_csv(DATA_PATH)

target_col = "Diagnosis"
numeric_cols = [col for col in df.columns if col != target_col]


# ============================================================
# 3. Auditoria geral
# ============================================================

overview = pd.DataFrame({
    "Indicador": [
        "Total de linhas",
        "Total de colunas",
        "Total de variáveis preditoras",
        "Variável-alvo",
        "Total de valores ausentes",
        "Total de duplicatas exatas",
        "Linhas após remoção de duplicatas"
    ],
    "Valor": [
        df.shape[0],
        df.shape[1],
        len(numeric_cols),
        target_col,
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
        int(df.drop_duplicates().shape[0])
    ]
})

overview.to_csv(
    RESULTS_TABLES / "01_dataset_overview.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. Tipos das colunas
# ============================================================

dtypes = pd.DataFrame({
    "Coluna": df.columns,
    "Tipo": [str(df[col].dtype) for col in df.columns],
    "Valores ausentes": [int(df[col].isna().sum()) for col in df.columns],
    "Valores únicos": [int(df[col].nunique()) for col in df.columns]
})

dtypes.to_csv(
    RESULTS_TABLES / "01_column_types.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. Distribuição das classes - base original
# ============================================================

class_distribution_raw = (
    df[target_col]
    .value_counts()
    .rename_axis("Classe")
    .reset_index(name="Quantidade")
)

class_distribution_raw["Percentual"] = (
    class_distribution_raw["Quantidade"] / class_distribution_raw["Quantidade"].sum()
) * 100

class_distribution_raw.to_csv(
    RESULTS_TABLES / "01_class_distribution_raw.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. Distribuição das classes - após remoção de duplicatas
# ============================================================

df_dedup = df.drop_duplicates().copy()

class_distribution_dedup = (
    df_dedup[target_col]
    .value_counts()
    .rename_axis("Classe")
    .reset_index(name="Quantidade")
)

class_distribution_dedup["Percentual"] = (
    class_distribution_dedup["Quantidade"] / class_distribution_dedup["Quantidade"].sum()
) * 100

class_distribution_dedup.to_csv(
    RESULTS_TABLES / "01_class_distribution_dedup.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. Estatísticas descritivas das variáveis numéricas
# ============================================================

numeric_summary = df[numeric_cols].describe().T.reset_index()
numeric_summary = numeric_summary.rename(columns={"index": "Variável"})

numeric_summary.to_csv(
    RESULTS_TABLES / "01_numeric_summary_raw.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. Detecção de possíveis valores extremos por IQR
#    Observação:
#    Aqui NÃO removemos nada. Apenas relatamos.
# ============================================================

iqr_report = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    n_lower = int((df[col] < lower).sum())
    n_upper = int((df[col] > upper).sum())
    n_total = n_lower + n_upper

    iqr_report.append({
        "Variável": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Limite inferior IQR": lower,
        "Limite superior IQR": upper,
        "Valores abaixo do limite": n_lower,
        "Valores acima do limite": n_upper,
        "Total de potenciais outliers": n_total
    })

iqr_report = pd.DataFrame(iqr_report)

iqr_report.to_csv(
    RESULTS_TABLES / "01_iqr_outlier_report.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. Auditoria de valores clinicamente suspeitos
#    Observação:
#    As regras abaixo são amplas e servem apenas para auditoria.
#    Não são usadas para diagnóstico nem para exclusão automática.
# ============================================================

sanity_rules = {
    "WBC":  {"min": 0.1,  "max": 100},
    "LYMp": {"min": 0,    "max": 100},
    "NEUTp": {"min": 0,   "max": 100},
    "LYMn": {"min": 0,    "max": 100},
    "NEUTn": {"min": 0,   "max": 100},
    "RBC":  {"min": 0.1,  "max": 10},
    "HGB":  {"min": 0,    "max": 25},
    "HCT":  {"min": 0,    "max": 75},
    "MCV":  {"min": 40,   "max": 150},
    "MCH":  {"min": 5,    "max": 60},
    "MCHC": {"min": 10,   "max": 60},
    "PLT":  {"min": 1,    "max": 1000},
    "PDW":  {"min": 0,    "max": 50},
    "PCT":  {"min": 0,    "max": 2}
}

sanity_report = []

for col, limits in sanity_rules.items():
    if col in df.columns:
        below = int((df[col] < limits["min"]).sum())
        above = int((df[col] > limits["max"]).sum())
        sanity_report.append({
            "Variável": col,
            "Mínimo observado": df[col].min(),
            "Máximo observado": df[col].max(),
            "Limite mínimo adotado": limits["min"],
            "Limite máximo adotado": limits["max"],
            "Abaixo do limite": below,
            "Acima do limite": above,
            "Total suspeito": below + above
        })

sanity_report = pd.DataFrame(sanity_report)

sanity_report.to_csv(
    RESULTS_TABLES / "01_sanity_check_report.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Impressão resumida no terminal
# ============================================================

print("\n================ AUDITORIA DO DATASET ================\n")
print(overview.to_string(index=False))

print("\n================ DISTRIBUIÇÃO DAS CLASSES - ORIGINAL ================\n")
print(class_distribution_raw.to_string(index=False))

print("\n================ DISTRIBUIÇÃO DAS CLASSES - SEM DUPLICATAS ================\n")
print(class_distribution_dedup.to_string(index=False))

print("\nArquivos gerados em:")
print(RESULTS_TABLES)
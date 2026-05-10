# 🧬 AnemiaClassifierML

> **Triagem fenotípica de anemias baseada em hemograma completo com Machine Learning**  
> Protocolo calibrado, interpretável e sensível ao desbalanceamento de classes para apoio à decisão clínica.

---

## 📌 Visão Geral

O **AnemiaClassifierML** é um projeto de aprendizado de máquina aplicado à **triagem fenotípica de anemias** a partir de variáveis do **hemograma completo** (*Complete Blood Count — CBC*).

O objetivo do projeto não é substituir o diagnóstico clínico, mas oferecer um protocolo computacional de **apoio à decisão**, combinando:

- auditoria e limpeza do conjunto de dados;
- prevenção de vazamento de informação;
- engenharia de atributos hematológicos derivados;
- comparação de classificadores supervisionados;
- estratégias de tratamento do desbalanceamento de classes;
- calibração probabilística;
- limiares por classe;
- abstenção assistida para casos incertos;
- interpretabilidade com importância por permutação e SHAP.

O projeto foi desenvolvido como base experimental para o artigo:

> **Triagem fenotípica de anemias baseada em hemograma completo: um protocolo calibrado, interpretável e sensível ao desbalanceamento de classes**

---

## 🧠 O que é Anemia?

A anemia é uma condição clínica caracterizada pela redução da concentração de hemoglobina, da quantidade de eritrócitos circulantes ou da capacidade global de transporte de oxigênio pelo sangue.

Ela pode estar associada a diferentes causas, como:

- 🔬 deficiência de ferro;
- 🧪 deficiência de vitamina B12 ou folato;
- 🩸 perdas sanguíneas;
- 🧬 alterações genéticas;
- 🦠 doenças crônicas ou inflamatórias;
- 🧫 distúrbios hematológicos específicos.

Neste projeto, a classificação é tratada como **triagem fenotípica baseada exclusivamente no CBC**, pois a base utilizada não contém ferritina, vitamina B12, folato, ferro sérico, transferrina ou dados clínicos complementares.

---

## 🎯 Objetivos do Projeto

| Tipo de Objetivo | Descrição |
|------------------|-----------|
| 🎯 Geral | Construir e avaliar um protocolo de triagem fenotípica de anemias baseado em hemograma completo. |
| 🔍 Específicos | Auditar o conjunto de dados, remover duplicatas, avaliar modelos supervisionados, criar atributos hematológicos derivados, tratar desbalanceamento, calibrar probabilidades, testar limiares, aplicar abstenção assistida e interpretar o modelo líder. |
| 🧪 Experimental | Comparar CBC original, CBC com atributos derivados, estratégias de balanceamento, calibração probabilística, limiares por classe e políticas de abstenção. |
| 🏥 Clínico-computacional | Apoiar a priorização de revisão humana em casos incertos, classes raras e regiões de sobreposição fenotípica. |

---

## 🗃️ Dataset

O projeto utiliza o conjunto público **Anemia Types Classification**, disponível na plataforma Kaggle.

### Variável-alvo

```text
Diagnosis

# Wine Quality Classification 🍷

Tech Challenge — Fase 2 | POSTECH — Data Analytics

## Sobre o projeto

Este projeto tem como objetivo desenvolver um modelo de **classificação binária** capaz de prever a qualidade de um vinho a partir de suas características físico-químicas (acidez, teor alcoólico, densidade, dióxido de enxofre, entre outras), utilizando o **Wine Quality Dataset** (Kaggle).

Tradicionalmente, a avaliação da qualidade de um vinho depende de análise sensorial feita por especialistas — um processo subjetivo e demorado. Este trabalho explora como técnicas de ciência de dados e aprendizado de máquina podem apoiar essa decisão de forma mais objetiva e escalável.

### Definição do problema

A variável original `quality` (nota atribuída por especialistas) foi transformada em uma classificação binária:

- **1 — Alta Qualidade**: nota ≥ 8
- **0 — Baixa/Média Qualidade**: nota < 8

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/              # Base de dados utilizada (Wine Quality Dataset)
├── notebooks/         # Notebook com a análise exploratória e modelagem
├── src/               # Scripts auxiliares (pré-processamento e modelagem)
├── results/           # Gráficos e métricas dos modelos
├── requirements.txt   # Bibliotecas utilizadas
└── README.md          # Descrição do projeto
```

## Base de dados

- **Fonte:** [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **Variáveis preditoras:**
  - Acidez fixa (fixed acidity)
  - Acidez volátil (volatile acidity)
  - Ácido cítrico (citric acid)
  - Açúcar residual (residual sugar)
  - Cloretos (chlorides)
  - Dióxido de enxofre livre (free sulfur dioxide)
  - Dióxido de enxofre total (total sulfur dioxide)
  - Densidade (density)
  - pH
  - Sulfatos (sulphates)
  - Teor alcoólico (alcohol)
- **Variável alvo:** Qualidade do vinho (quality) → transformada em classificação binária

## Metodologia

O desenvolvimento seguiu as seguintes etapas:

1. **Compreensão do problema**
   Interpretação do contexto, definição da variável alvo e transformação em classificação binária.

2. **Análise Exploratória de Dados (EDA)**
   - Distribuição das variáveis
   - Correlação entre variáveis (com justificativa)
   - Identificação de outliers e valores inconsistentes
   - Análise do balanceamento das classes

3. **Pré-processamento de dados**
   - Tratamento de dados faltantes (se houver)
   - Normalização/padronização das variáveis numéricas
   - Criação de novas features (quando relevante)

4. **Desenvolvimento dos modelos**
   Treinamento de pelo menos dois modelos de classificação para comparação.

5. **Avaliação dos modelos**
   Avaliação com métricas adequadas ao problema (acurácia, precisão, recall, F1-score, matriz de confusão) e comparação entre os modelos testados.

6. **Interpretação dos resultados**
   - Identificação das variáveis com maior influência na qualidade do vinho
   - Discussão de possíveis implicações para o processo de produção

## Como executar o projeto

1. Clone o repositório:
   ```bash
   git clone <link-do-repositorio>
   cd wine-quality-classification
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Baixe o dataset do Kaggle e coloque o arquivo `.csv` na pasta `data/`.

5. Execute o notebook principal:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## Modelos utilizados

| Modelo | Descrição |
|---|---|
| *A preencher* | *A preencher* |
| *A preencher* | *A preencher* |

## Resultados

| Modelo | Acurácia | Precisão | Recall | F1-score |
|---|---|---|---|---|
| *A preencher* | | | | |
| *A preencher* | | | | |

Gráficos e demais métricas estão disponíveis na pasta `results/`.

## Principais insights

- *A preencher após a EDA e modelagem* (ex: variáveis com maior correlação com a qualidade, comportamento por faixa de teor alcoólico, etc.)

## Entregáveis do Tech Challenge

- [x] Repositório GitHub com os códigos utilizados
- [ ] Apresentação executiva (storytelling da EDA) — formato PPT ou PDF, disponível neste repositório
- [ ] Vídeo executivo (até 5 minutos), em linguagem não técnica

## Autores

- Pâmela Cristina da Silva
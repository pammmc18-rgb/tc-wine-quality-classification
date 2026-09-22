# Wine Quality Classification

Projeto de classificação da qualidade de vinhos utilizando técnicas de Machine Learning.

## Objetivo

Aplicar um pipeline de análise exploratória, pré-processamento e modelagem para prever a qualidade de vinhos a partir de suas características físico-químicas.

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/              # Base de dados utilizada
├── notebooks/         # Notebook com a análise e modelagem
├── src/               # Scripts auxiliares (pré-processamento ou modelagem)
├── results/           # Gráficos e métricas dos modelos
├── requirements.txt   # Bibliotecas utilizadas
└── README.md          # Descrição do projeto
```

## Base de dados

Descrever aqui a origem do dataset (ex: Wine Quality Dataset - UCI Machine Learning Repository), quantidade de amostras, variáveis e o que a variável alvo representa.

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

4. Coloque o arquivo de dados na pasta `data/`.

5. Abra o notebook principal:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## Metodologia

- Análise exploratória dos dados (EDA)
- Pré-processamento (tratamento de nulos, normalização, encoding)
- Treinamento de modelos de classificação
- Avaliação com métricas (acurácia, precisão, recall, F1-score, matriz de confusão)

## Resultados

Resumir aqui os principais resultados obtidos, com referência aos gráficos salvos em `results/`.

## Autor

Pâmela Cristina da Silva 

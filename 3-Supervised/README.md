<br />
<p align="center">
  <h1 align="center">Pipeline de Classificação Supervisionada com Múltiplos Modelos</h1>
</p>

<!-- TABLE OF CONTENTS -->

## Tabela de Conteúdo

- [Tabela de Conteúdo](#tabela-de-conteúdo)
- [Sobre o Projeto](#sobre-o-projeto)
  - [Modelos Implementados](#modelos-implementados)
  - [Feito Com](#feito-com)
- [Começando](#começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Estrutura de Arquivos](#estrutura-de-arquivos)
  - [Instalação](#instalação)
  - [Configuração do Ambiente Virtual](#configuração-do-ambiente-virtual)
    - [Linux/Ubuntu](#linuxubuntu)
    - [Windows (Command Prompt)](#windows-command-prompt)
  - [Configuração do arquivo config.ini](#configuração-do-arquivo-configini)
    - [Seção \[LOG\]](#seção-log)
    - [Seção \[DATA\]](#seção-data)
    - [Seção \[MODEL\]](#seção-model)
    - [Seções de Hiperparâmetros dos Modelos](#seções-de-hiperparâmetros-dos-modelos)
    - [Seção \[PREPROCESSING\]](#seção-preprocessing)
    - [Seção \[PCA\]](#seção-pca)
    - [Seção \[METRIC\]](#seção-metric)
  - [Execução do Pipeline](#execução-do-pipeline)
    - [Linux/Ubuntu](#linuxubuntu-1)
    - [Windows (Command Prompt)](#windows-command-prompt-1)
  - [Descrição dos Módulos](#descrição-dos-módulos)
    - [`src/experiment/`](#srcexperiment)
    - [`src/utils/`](#srcutils)
    - [Arquivos na raiz](#arquivos-na-raiz)
    - [Pasta `report/`](#pasta-report)
- [Interpretando os Resultados](#interpretando-os-resultados)

<!-- ABOUT THE PROJECT -->

## Sobre o Projeto

Este projeto implementa um **pipeline automatizado de machine learning** para classificação supervisionada com suporte a múltiplos modelos, execuções com diferentes seeds, e análise comparativa detalhada.

### Modelos Implementados

O pipeline inclui 9 modelos de classificação prontos para uso:

1. **RandomForest**;
2. **XGBoost**;
3. **LightGBM**;
4. **CatBoost**;
5. **LogisticRegression**;
6. **AdaBoost** ;
7. **MLP (Multi-Layer Perceptron)**;
8. **SVM (Support Vector Machine)**;
9. **NaiveBayes**.
### Feito Com

Abaixo segue o que foi utilizado na criação e preparação do ambiente de desenvolvimento:

- [Python 3.13+](https://www.python.org/) - Linguagem de programação.
- [Pandas](https://pandas.pydata.org/) - Biblioteca para manipulação e análise de dados em Python.
- [NumPy](https://numpy.org/) - Biblioteca fundamental para computação científica em Python.
- [Scikit-learn](https://scikit-learn.org/) - Biblioteca de machine learning (modelos, pré-processamento, métricas).
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting otimizado.
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting rápido para grandes datasets.
- [CatBoost](https://catboost.ai/) - Gradient boosting robusto para dados categóricos.
- [Matplotlib](https://matplotlib.org/) - Biblioteca para visualização de dados.
- [Seaborn](https://seaborn.pydata.org/) - Visualizações estatísticas.
- [Joblib](https://joblib.readthedocs.io/) - Serialização eficiente de objetos Python.
- [ConfigParser](https://docs.python.org/3/library/configparser.html) - Parser para arquivos de configuração.
- [Logging](https://docs.python.org/3/library/logging.html) - Sistema de logging nativo do Python.

<!-- GETTING STARTED -->

## Começando

Para começar a utilizar este projeto, é necessário ter alguns pré-requisitos de ambiente.

### Pré-requisitos

1. A utilização do ambiente requer a instalação do Python 3.13 ou superior. Para instalá-lo, acesse o site oficial do [Python](https://www.python.org/downloads/) e siga as instruções de instalação para o seu sistema operacional.
2. A documentação de cada biblioteca aqui utilizada pode ser encontrada nos links fornecidos em [Feito Com](#feito-com).
3. Foi desenvolvido e testado em sistemas Windows e Linux, então recomenda-se tais sistemas operacionais para melhor compatibilidade.
4. O pipeline espera um arquivo CSV em `./data/raw/dataset.csv` com uma coluna de target (configurável em `config.ini`).
### Estrutura de Arquivos

A estrutura de arquivos está da seguinte maneira:

```bash
3-Supervised/
  ├── src/
  │   ├── experiment/
  │   │   ├── config/
  │   │   │   └── config_reader.py
  │   │   ├── data_handling/
  │   │   │   ├── data_manager.py
  │   │   │   ├── preprocessor.py
  │   │   │   └── dimensionality_reducer.py
  │   │   ├── models/
  │   │   │   ├── base_model.py
  │   │   │   ├── model_factory.py
  │   │   │   ├── random_forest_model.py
  │   │   │   ├── xgboost_model.py
  │   │   │   ├── lightgbm_model.py
  │   │   │   ├── catboost_model.py
  │   │   │   ├── logistic_regression_model.py
  │   │   │   ├── adaboost_model.py
  │   │   │   ├── mlp_model.py
  │   │   │   ├── svm_model.py
  │   │   │   └── naive_bayes_model.py
  │   │   ├── evaluation/
  │   │   │   ├── evaluator.py
  │   │   │   ├── comparison_report.py
  │   │   │   └── comparison_plots.py
  │   │   ├── context.py
  │   │   ├── experiment.py
  │   │   └── metrics.py
  │   └── utils/
  │       ├── logging_utils.py
  │       ├── filepath_validation.py
  │       └── remove_temp_files.py
  ├── data/
  │   └── raw/
  │       └── dataset.csv
  ├── report/
  │   └── experiment_YYYYMMDD_HHMMSS/
  │       ├── logs/experiment.log
  │       ├── data_ids/
  │       ├── [ModelName]/run_1/
  │       └── comparison/
  ├── config.ini
  ├── main.py
  ├── requirements.txt
  ├── requirements-dev.txt
  └── README.md
```

### Instalação

1. Para instalar e utilizar esse projeto, basta clonar o repositório:

```bash
git clone https://github.com/gustavomello-source/IA---ENTREGA-FINAL.git
cd 3-Supervised
```

2. Certifique-se de que você está no diretório do projeto:

```bash
cd 3-Supervised
```
### Configuração do Ambiente Virtual

Para criar o ambiente virtual, ativá-lo e instalar as dependências, utilize os seguintes comandos baseados no seu sistema operacional:

#### Linux/Ubuntu

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

#### Windows (Command Prompt)

```cmd
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```
### Configuração do arquivo config.ini

Antes de executar o treinamento, configure o arquivo `config.ini` com os parâmetros desejados. O arquivo está organizado em seções:

#### Seção [LOG]

```ini
[LOG]
report_path = ./report/
```

- **report_path**: Diretório onde os relatórios serão salvos

#### Seção [DATA]

```ini
[DATA]
data_path = ./data/raw/dataset.csv
target_column = Y
```

- **data_path**: Caminho para o arquivo CSV de entrada
- **target_column**: Nome da coluna target (variável resposta)

#### Seção [MODEL]

```ini
[MODEL]
models = RandomForest, XGBoost, LogisticRegression, LightGBM, AdaBoost, CatBoost, MLP, NaiveBayes
n_runs = 10
```

- **models**: Lista de modelos a serem treinados (separados por vírgula). Remova ou adicione modelos conforme necessário
- **n_runs**: Número de execuções por modelo com seeds diferentes (para avaliar estabilidade)

#### Seções de Hiperparâmetros dos Modelos

Cada modelo tem sua própria seção com hiperparâmetros específicos. Exemplos:

```ini
[RANDOMFOREST]
n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = sqrt
class_weight = balanced
random_state = 1
n_jobs = -1

[XGBOOST]
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = auto
random_state = 1
n_jobs = -1

[LIGHTGBM]
n_estimators = 100
learning_rate = 0.1
num_leaves = 31
max_depth = -1
scale_pos_weight = auto
random_state = 1
n_jobs = -1
verbose = -1

[MLP]
hidden_layer_sizes = 100, 50
activation = relu
alpha = 0.0001
learning_rate_init = 0.001
max_iter = 200
early_stopping = true
random_state = 1
```

**Dicas de configuração**:
- `class_weight = balanced` (RandomForest, LogisticRegression, SVM) e `scale_pos_weight = auto` (XGBoost, LightGBM, CatBoost) ajudam com dados desbalanceados
- `n_jobs = -1` usa todos os núcleos da CPU
- `random_state = 1` é substituído automaticamente por 1, 2, 3... em cada run
#### Seção [PREPROCESSING]

```ini
[PREPROCESSING]
id_column = ID
sentinel_values = -9999, -9998, -9997, -999, -998, -997
drop_near_constant = true
near_constant_threshold = 0.99
drop_high_missing = true
high_missing_threshold = 0.95
imputation_strategy = median
add_missing_indicator = false
one_hot_encode = true
low_cardinality_max = 10
scaler = robust
```

Parâmetros principais:
- **id_column**: Coluna de identificador (removida antes do treinamento)
- **sentinel_values**: Códigos que representam valores ausentes (convertidos para NaN)
- **drop_near_constant**: Remove colunas onde um valor aparece em >99% das amostras
- **drop_high_missing**: Remove colunas com >95% de valores ausentes
- **imputation_strategy**: Estratégia de imputação (`mean`, `median`, `most_frequent`)
- **one_hot_encode**: Converte colunas categóricas (≤10 valores únicos) em one-hot
- **scaler**: Escalonamento das features numéricas (`robust`, `standard`, `none`)

#### Seção [PCA]

```ini
[PCA]
enabled = true
n_components = 0.95
standardize_before = true
svd_solver = auto
whiten = false
random_state = 1
```

- **enabled**: Ativa/desativa redução de dimensionalidade
- **n_components**: `0.95` retém 95% da variância; um inteiro retém N componentes
- **standardize_before**: Padroniza features antes do PCA (recomendado)

#### Seção [METRIC]

```ini
[METRIC]
primary = f1_macro
minority_class = auto
```

- **primary**: Métrica principal para comparação (não utilizada atualmente; sempre usa `f1_macro`)
- **minority_class**: `auto` detecta automaticamente a classe minoritária; ou especifique explicitamente (ex: `0`, `1`)
### Execução do Pipeline

Para executar o projeto no ambiente virtual, ative o ambiente e execute o script principal:

#### Linux/Ubuntu

```bash
source .venv/bin/activate

python main.py
```

#### Windows (Command Prompt)

```cmd
.venv\Scripts\activate

python main.py
```

**O que acontece durante a execução**:

1. **Carregamento dos dados**: Lê o CSV e faz split estratificado 80/20 (train/test)
2. **Pré-processamento**: Aplica limpeza, imputação, encoding e escalonamento no conjunto de treino; transforma o conjunto de teste com os mesmos parâmetros
3. **Redução de dimensionalidade**: (Opcional) Aplica PCA
4. **Treinamento multi-run**: Para cada modelo configurado:
   - Treina N vezes com seeds 1, 2, 3, ..., N
   - Salva cada modelo treinado em `report/.../[ModelName]/run_X/`
   - Mede tempo de treinamento
5. **Avaliação**: Para cada run de cada modelo:
   - Calcula métricas completas no conjunto de teste
   - Gera matriz de confusão
   - Identifica falsos positivos e falsos negativos
6. **Comparação**: Agrega todas as métricas:
   - Calcula média ± desvio padrão por modelo
   - Gera tabelas CSV com resultados agregados e por run
   - Gera gráficos de distribuição (boxplots, barplots)
   - Identifica o melhor modelo por F1 macro

### Descrição dos Módulos

Esta seção descreve brevemente cada diretório e arquivo do projeto:

#### `src/experiment/`

- **config/config_reader.py** - Lê e parseia o arquivo `config.ini`
- **data/data_manager.py** - Carrega CSV, faz split train/test estratificado, salva IDs das amostras
- **data/preprocessor.py** - Pipeline de pré-processamento configurável (limpeza, imputação, encoding, escalonamento)
- **data/dimensionality_reducer.py** - Redução de dimensionalidade via PCA
- **models/base_model.py** - Classe base abstrata que todos os modelos herdam
- **models/model_factory.py** - Factory com auto-discovery — escaneia a pasta `models/` e registra automaticamente qualquer subclasse de `BaseModel` com atributo `MODEL_NAME`
- **models/[nome]_model.py** - Implementações concretas dos 9 modelos

- **evaluation/evaluator.py** - Calcula métricas (acurácia, precisão, recall, F1, ROC-AUC, matriz de confusão, análise de erros)
- **evaluation/comparison_report.py** - Agrega métricas de múltiplos runs, calcula média/std, gera tabelas CSV e relatório textual
- **evaluation/comparison_plots.py** - Gera boxplots e barplots para visualização de distribuições

- **context.py** - Contexto compartilhado que armazena dados, configuração, modelos treinados e artefatos entre as etapas do pipeline
- **experiment.py** - Orquestrador principal que coordena todas as etapas (preprocess → PCA → train → compare)
- **metrics.py** - Funções auxiliares para detecção de classe minoritária e cálculo de F1

#### `src/utils/`

- **logging_utils.py** - Configura logging estruturado com saída para console e arquivo
- **filepath_validation.py** - Valida e cria diretórios de saída
- **remove_temp_files.py** - Limpeza de arquivos temporários ao final

#### Arquivos na raiz

- **main.py** - Script principal que lê `config.ini`, configura logging, carrega dados, cria o contexto e executa o experimento
- **config.ini** - Arquivo de configuração com hiperparâmetros e configurações do sistema
- **requirements.txt** - Lista de dependências Python necessárias
- **requirements-dev.txt** - Dependências de desenvolvimento
- **README.md** - Documentação do projeto

#### Pasta `report/`

Diretório onde são salvos os resultados dos experimentos. Cada execução cria uma pasta única timestampada:

- **experiment_[timestamp]/** - Pasta única para cada execução contendo:
  - **logs/experiment.log** - Log completo da execução
  - **data_ids/** - IDs das amostras de treino e teste
  - **[ModelName]/run_X/** - Modelos treinados e métricas por run
  - **comparison/** - Tabelas agregadas, gráficos e sumário textual

## Interpretando os Resultados

Após a execução, os principais arquivos de saída são:

1. **`report/.../comparison/metrics_table_aggregated.csv`**: Tabela com média ± desvio padrão de cada métrica por modelo. Use para comparação rápida.
2. **`report/.../comparison/summary.txt`**: Resumo textual identificando o melhor modelo.
3. **`report/.../comparison/plots/`**: Gráficos de distribuição (boxplots mostram variabilidade entre runs).
4. **`report/.../[ModelName]/confusion_matrix.png`**: Visualização da matriz de confusão.
5. **`report/.../[ModelName]/errors_false_positives.csv` e `errors_false_negatives.csv`**: IDs das amostras classificadas incorretamente para análise de erros.

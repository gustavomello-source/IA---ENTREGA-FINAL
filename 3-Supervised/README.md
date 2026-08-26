<br />
<p align="center">
  <h1 align="center">NOME DO PROJETO</h1>
</p>

<!-- TABLE OF CONTENTS -->

## Tabela de Conteúdo

- [Tabela de Conteúdo](#tabela-de-conteúdo)
- [Sobre o Projeto](#sobre-o-projeto)
  - [Feito Com](#feito-com)
- [Começando](#começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Estrutura de Arquivos](#estrutura-de-arquivos)
  - [Instalação](#instalação)
  - [Configuração do Ambiente Virtual](#configuração-do-ambiente-virtual)
    - [Linux/Ubuntu](#linuxubuntu)
    - [Windows (Command Prompt)](#windows-command-prompt)
  - [Configuração do arquivo config.ini](#configuração-do-arquivo-configini)
  - [Configuração do arquivo .env](#configuração-do-arquivo-env)
  - [Execução do Script de Treinamento](#execução-do-script-de-treinamento)
    - [Linux/Ubuntu](#linuxubuntu-1)
    - [Windows (Command Prompt)](#windows-command-prompt-1)
  - [Edição](#edição)

<!-- ABOUT THE PROJECT -->

## Sobre o Projeto

Este projeto implementa um

O sistema é capaz de:
- F

### Feito Com

Abaixo segue o que foi utilizado na criação e preparação do ambiente de desenvolvimento:

- [Pandas](https://pandas.pydata.org/) - Biblioteca para manipulação e análise de dados em Python.
- [NumPy](https://numpy.org/) - Biblioteca fundamental para computação científica em Python.
- [Scikit-learn](https://scikit-learn.org/) - Biblioteca de machine learning para Python com algoritmos de classificação, regressão e clustering.
- [Joblib](https://joblib.readthedocs.io/) - Biblioteca para serialização eficiente de objetos Python, especialmente arrays NumPy.
- [ConfigParser](https://docs.python.org/3/library/configparser.html) - Parser para arquivos de configuração.
- [OpenPyXL](https://openpyxl.readthedocs.io/) - Biblioteca para leitura e escrita de arquivos Excel.
- [Logging](https://docs.python.org/3/library/logging.html) - Sistema de logging nativo do Python.

<!-- GETTING STARTED -->

## Começando

Para começar a utilizar este projeto, é necessário ter alguns pré-requisitos de ambiente.

### Pré-requisitos

1. A utilização do ambiente requer a instalação do Python 3.11 ou superior. Para instalá-lo, acesse o site oficial do [Python](https://www.python.org/downloads/) e siga as instruções de instalação para o seu sistema operacional.
2. A documentação de cada biblioteca aqui utilizada pode ser encontrada nos links fornecidos em [Feito Com](#feito-com).
3. Foi desenvolvido e testado em sistemas Windows e Linux, então recomenda-se tais sistemas operacionais para melhor compatibilidade.

### Estrutura de Arquivos

A estrutura de arquivos está da seguinte maneira:

```bash
project/
  ├── utils/
  │   ├── __init__.py
  │   └── content.py
  ├── report/
  │   └── experiment_[timestamp]/
  │       └── training_report_[nome]_[timestamp].json
  ├── .env.example
  ├── requirements.txt
  ├── README.md
  ├── train_script.py
  └── train_script_config.ini
```

### Instalação

1. Para instalar e utilizar esse projeto, basta clonar o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd project
```

2. Certifique-se de que você está no diretório do projeto:

```bash
cd project
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

Antes de executar o treinamento, configure o arquivo `config.ini` com os parâmetros desejados:

```ini

[MODEL 1]
n_estimators = 289
```

As principais configurações do modelo são:

- **n_estimators**: Número de árvores no Random Forest

### Configuração do arquivo .env

Crie um arquivo chamado `.env` na raiz do projeto e adicione as seguintes variáveis de ambiente, como exemplificado em `.env.example`:

```ini
[Database]
host =
port =
database =
user =
password =

[API]
API_URL =
API_TOKEN =
```

As variáveis de ambiente a serem configuradas são as seguintes:

- **host**: Endereço do servidor de banco de dados
- **port**: Porta do servidor de banco de dados
- **database**: Nome do banco de dados
- **user**: Usuário para conexão
- **password**: Senha para conexão
- **API_URL**: URL da API, se aplicável
- **API_TOKEN**: Token de autenticação da API, se aplicável

### Execução do Script de Treinamento

Para executar o projeto no ambiente virtual, ative o ambiente e execute o script de treinamento:

#### Linux/Ubuntu

```bash
source .venv/bin/activate

python train_script.py
```

#### Windows (Command Prompt)

```cmd
.venv\Scripts\activate

python train_script.py
```

### Edição

Esta seção descreve brevemente cada diretório e arquivo do projeto:

- **utils/** - Módulos utilitários do sistema:
  - **content.py** - Exemplo de módulo utilitário que contém funções auxiliares para o projeto

- **report/** - Diretório onde são salvos os resultados dos experimentos:
  - **experiment_[timestamp]/** - Pasta única para cada execução contendo:
    - **training_report_[nome]_[timestamp].json** - Relatório em JSON com métricas

- **.env.example** - Exemplo de arquivo de variáveis de ambiente
- **README.md** - Documentação do projeto
- **requirements.txt** - Lista de dependências Python necessárias
- **train_script.py** - Script principal que coordena todo o pipeline de treinamento
- **train_script_config.ini** - Arquivo de configuração com hiperparâmetros e configurações do sistema

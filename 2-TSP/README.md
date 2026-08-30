<br />
<p align="center">
  <h1 align="center">Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)</h1>
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
  - [Disposição dos dados](#disposição-dos-dados)
  - [Configuração do arquivo config.ini](#configuração-do-arquivo-configini)
  - [Execução](#execução)
    - [Linux/Ubuntu](#linuxubuntu-1)
    - [Windows (Command Prompt)](#windows-command-prompt-1)
  - [Edição](#edição)

<!-- ABOUT THE PROJECT -->

## Sobre o Projeto

Este projeto implementa um algoritmo genético para resolver o Problema do Caixeiro Viajante (TSP), utilizando instâncias no formato TSPLIB. O objetivo é encontrar um percurso de menor distância que visite todas as cidades exatamente uma vez e retorne à cidade de origem.

O sistema é capaz de:
- Ler instâncias TSPLIB (`.tsp`) e tours de referência (`.opt.tour`).
- Construir a matriz de distâncias, com suporte aos tipos `EUC_2D` (euclidiana) e `ATT` (pseudo-euclidiana).
- Avaliar a distância total de um tour.
- Evoluir soluções por meio de um algoritmo genético com seleção por torneio, order crossover (OX), mutação por troca (swap) e elitismo.
- Registrar o progresso de cada geração em um arquivo de log dentro da pasta `results/`.
- Imprimir o melhor tour encontrado e sua distância ao final da execução.

### Feito Com

Abaixo segue o que foi utilizado na criação e preparação do ambiente de desenvolvimento:

- [NumPy](https://numpy.org/) - Biblioteca fundamental para computação científica em Python, utilizada na construção da matriz de distâncias.
- [ConfigParser](https://docs.python.org/3/library/configparser.html) - Parser nativo do Python para leitura do arquivo de configuração.
- [Logging](https://docs.python.org/3/library/logging.html) - Sistema de logging nativo do Python, utilizado para registrar o progresso por geração.
- [dataclasses](https://docs.python.org/3/library/dataclasses.html) - Módulo nativo do Python utilizado na modelagem das entidades do domínio.

<!-- GETTING STARTED -->

## Começando

Para começar a utilizar este projeto, é necessário ter alguns pré-requisitos de ambiente.

### Pré-requisitos

1. A utilização do ambiente requer a instalação do Python 3.11 ou superior. Para instalá-lo, acesse o site oficial do [Python](https://www.python.org/downloads/) e siga as instruções de instalação para o seu sistema operacional.
2. A documentação de cada biblioteca aqui utilizada pode ser encontrada nos links fornecidos em [Feito Com](#feito-com).
3. Foi desenvolvido e testado em sistemas Windows e Linux, então recomenda-se tais sistemas operacionais para melhor compatibilidade.
4. Dados de instâncias TSPLIB, dispostos dentro de uma pasta chamada `data/` no diretório raiz do projeto, as quais podem ser obtidas no site oficial do [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

### Estrutura de Arquivos

A estrutura de arquivos está da seguinte maneira:

```bash
2-TSP/
  ├── config/
  │   └── config.ini
  ├── data/                       # Instâncias TSPLIB (.tsp) e tours de referência (.opt.tour)
  ├── results/                    # Logs por execução (criado automaticamente)
  ├── src/
  │   ├── __init__.py
  │   ├── main.py
  │   ├── configuration/
  │   │   ├── __init__.py
  │   │   └── config_manager.py
  │   ├── genetic_algorithm/
  │   │   ├── __init__.py
  │   │   ├── individual.py
  │   │   ├── population.py
  │   │   ├── genetic_algorithm.py
  │   │   └── operators/
  │   │       ├── __init__.py
  │   │       ├── selection.py
  │   │       ├── crossover.py
  │   │       └── mutation.py
  │   ├── tsp/
  │   │   ├── __init__.py
  │   │   ├── city.py
  │   │   ├── tsp_parser.py
  │   │   ├── tour_parser.py
  │   │   └── tsp_instance.py
  │   └── utils/
  │       ├── __init__.py
  │       └── distances.py
  ├── requirements.txt
  └── README.md
```

### Instalação

1. Para instalar e utilizar esse projeto, basta clonar o repositório:

```bash
git clone https://github.com/gustavomello-source/IA---ENTREGA-FINAL
cd 2-TSP
```

2. Certifique-se de que você está no diretório do projeto:

```bash
cd 2-TSP
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

### Disposição dos dados
É necessário que os dados das instâncias TSPLIB estejam dispostos dentro de uma pasta chamada `data/` no diretório raiz do projeto. As instâncias podem ser obtidas no site oficial do [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).
Os dados devem estar no formato `.tsp` e, caso haja tours de referência, no formato `.opt.tour`.

### Configuração do arquivo config.ini

Antes de executar o algoritmo, configure o arquivo `config/config.ini` com os parâmetros desejados:

```ini
[GENETIC_ALGORITHM]
population_size = 100
generations = 500
mutation_rate = 0.02
crossover_rate = 0.90
elitism_size = 2
tournament_size = 5
random_seed = 1

[TSP]
dataset = data/att48.tsp
tour_file = data/att48.opt.tour
```

As principais configurações são:

- **population_size**: Número de indivíduos (tours) em cada geração.
- **generations**: Número de gerações a serem evoluídas.
- **mutation_rate**: Probabilidade de aplicar a mutação por troca (valor entre 0.0 e 1.0).
- **crossover_rate**: Probabilidade de aplicar o order crossover (valor entre 0.0 e 1.0).
- **elitism_size**: Quantidade de melhores indivíduos preservados intactos a cada geração.
- **tournament_size**: Número de competidores em cada torneio de seleção.
- **random_seed**: Semente do gerador de números aleatórios, garantindo a reprodutibilidade das execuções.
- **dataset**: Caminho para a instância TSPLIB (`.tsp`) a ser resolvida.

### Execução

Para executar o projeto no ambiente virtual, ative o ambiente e execute o módulo principal:

#### Linux/Ubuntu

```bash
source .venv/bin/activate

python3 -m src.main
```

#### Windows (Command Prompt)

```cmd
.venv\Scripts\activate

python -m src.main
```

Ao final da execução, o melhor tour encontrado e sua distância são impressos no console. Além disso, o progresso de cada geração é registrado em um arquivo de log dentro da pasta `results/`, nomeado como `<instancia>_<timestamp>.log` (a pasta é criada automaticamente caso não exista).

### Edição

Esta seção descreve brevemente cada diretório e arquivo do projeto:

- **config/** - Arquivos de configuração do projeto:
  - **config.ini** - Parâmetros do algoritmo genético e definição da instância TSP a ser resolvida.

- **data/** - Instâncias TSPLIB (`.tsp`) e tours de referência (`.opt.tour`).

- **results/** - Diretório onde são salvos os logs de cada execução (criado automaticamente):
  - **[instancia]_[timestamp].log** - Registro do progresso por geração.

- **src/** - Código-fonte do projeto:
  - **main.py** - Ponto de entrada que carrega a configuração, monta os operadores, injeta as dependências no algoritmo genético e executa a evolução.
  - **configuration/config_manager.py** - Carrega e valida os parâmetros do arquivo `config.ini`.
  - **genetic_algorithm/individual.py** - Representa um indivíduo (tour candidato) e sua distância.
  - **genetic_algorithm/population.py** - Representa uma população de indivíduos e oferece operações de busca e ordenação por aptidão.
  - **genetic_algorithm/genetic_algorithm.py** - Orquestra o laço evolutivo com elitismo e registra o progresso em log.
  - **genetic_algorithm/operators/selection.py** - Seleção por torneio.
  - **genetic_algorithm/operators/crossover.py** - Order Crossover (OX), que preserva a validade da permutação.
  - **genetic_algorithm/operators/mutation.py** - Mutação por troca (swap).
  - **tsp/city.py** - Representa uma cidade (identificador e coordenadas).
  - **tsp/tsp_parser.py** - Leitor de instâncias TSPLIB (`.tsp`).
  - **tsp/tour_parser.py** - Leitor de tours de referência (`.opt.tour`).
  - **tsp/tsp_instance.py** - Representa a instância do problema e constrói a matriz de distâncias.
  - **utils/distances.py** - Funções de cálculo das distâncias (`EUC_2D` e `ATT`).

- **README.md** - Documentação do projeto.
- **requirements.txt** - Lista de dependências Python necessárias.

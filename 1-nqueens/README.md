# N-Queens Solver

Implementação do problema das **N-Rainhas (N-Queens)** utilizando **backtracking recursivo** com uma **heurística** para priorizar as melhores posições.

## Requisitos

* Python 3.13 ou superior
* Nenhuma biblioteca externa é necessária.

## Estrutura do projeto

```text
1-nqueens/
├── main.py
├── nqueens.py
├── read_input.py
└── input.txt
```

* `main.py`: ponto de entrada do programa.
* `nqueens.py`: implementação do algoritmo de solução.
* `read_input.py`: responsável pela leitura do arquivo de entrada.
* `input.txt`: arquivo contendo a dimensão do tabuleiro.

## Arquivo de entrada

O arquivo de entrada deve conter, na primeira linha, um número inteiro `N`, representando o tamanho do tabuleiro.

Exemplo de `input.txt`:

```text
8
```

Isso representa um tabuleiro **8 × 8** e o problema das **8-Rainhas**.

## Como executar

### 1. Usando o arquivo padrão

Se o arquivo se chamar `input.txt` e estiver na mesma pasta do `main.py`, execute:

#### Windows (Command Prompt):
```bash
python main.py
```

#### Linux/Ubuntu:
```bash
python3 main.py
```

Caso nenhum arquivo seja informado, o programa tentará utilizar automaticamente:

```text
./input.txt
```

### 2. Informando um arquivo específico

Também é possível informar o caminho do arquivo através do argumento `-f` ou `--file_path`:

```bash
python main.py -f input.txt
```

Ou:

```bash
python main.py --file_path input.txt
```

Exemplo utilizando outro arquivo:

```bash
python main.py -f inputs/nqueens_20.txt
```

## Saída

Ao executar o programa, serão exibidos:

* Identificação do solver;
* Tabuleiro com a solução encontrada;
* Número de iterações realizadas;
* Tempo de execução.

Exemplo:

```text
Solver: NQueensSolver(n=8)

Solution:

Q . . . . . . .
. . . . Q . . .
. . . . . . . Q
. . . . . Q . .
. . Q . . . . .
. . . . . . Q .
. Q . . . . . .
. . . Q . . . .

Iterations: 9
Execution time: 0.000123 seconds
```

Onde:

* `Q` representa uma rainha;
* `.` representa uma posição vazia.

## Funcionamento resumido

O algoritmo utiliza:

1. **Backtracking recursivo** para testar diferentes posições das rainhas.
2. **Controle de linhas e diagonais** para evitar posições inválidas.
3. **Heurística** para ordenar as posições candidatas e tentar primeiro aquelas que deixam mais possibilidades para as próximas rainhas.
4. **Contagem de iterações** para medir o número de chamadas realizadas durante a busca.
5. **Medição do tempo de execução** para avaliar o desempenho.

## Exemplo rápido

Para resolver o problema das **8-Rainhas**:

**input.txt**

```text
8
```

Execute:

```bash
python main.py
```

O programa irá calcular uma solução e exibir o tabuleiro no terminal.

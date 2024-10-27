# Análise de Dados de Seguro de Saúde

## Introdução

Este projeto realiza a análise de um conjunto de dados de seguro de saúde utilizando várias técnicas de regressão. O código lê um arquivo CSV contendo os dados e executa correlações, regressões lineares múltiplas e outras análises estatísticas.

## Bibliotecas Utilizadas

- `time`
- `random`
- `numpy`
- `pandas`
- `seaborn`
- `math`
- `matplotlib`
- `sklearn`
- `statistics`
- `tabulate`
- `statsmodels`

## Descrição dos Dados

Os dados utilizados no projeto são lidos a partir de um arquivo CSV localizado no caminho `C:\insurance.csv`. Este conjunto de dados contém as seguintes colunas:

- `age`: Idade do beneficiário
- `sex`: Sexo do beneficiário
- `bmi`: Índice de Massa Corporal
- `children`: Número de filhos cobertos pelo seguro
- `smoker`: Se o beneficiário é fumante ou não
- `region`: Região onde o beneficiário reside
- `charges`: Custos médicos incorridos

## Funções Utilizadas

### `strToBool(str)`

Converte valores de string para booleanos.

### `strToInt(str)`

Converte valores de string para inteiros, representando as diferentes regiões.

## Análise de Correlação

Correlação entre as variáveis `age`, `sex`, `bmi`, `children`, `smoker`, `region` e `charges`.

## Regressão Linear Múltipla

Executa a regressão linear múltipla utilizando as variáveis `age`, `bmi`, `children` e `smoker` como preditores para `charges`.

### Avaliação do Modelo

- Erro quadrático médio (RMSE)
- Desvio padrão dos erros

## Exportação de Resultados

Os resultados das correlações e determinações são exportados para um arquivo Excel `dados_pro_relatorio_ana_interprete.xlsx`.

## Como Executar

1. Clone o repositório.
2. Certifique-se de ter todas as bibliotecas necessárias instaladas.
3. Execute o script em um ambiente Python configurado corretamente.
4. Verifique o arquivo `dados_pro_relatorio_ana_interprete.xlsx` para os resultados das análises.

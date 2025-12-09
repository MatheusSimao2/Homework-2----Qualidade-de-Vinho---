# A Superioridade de Redes Neurais na Predição da Qualidade do Vinho: Uma Análise Comparativa

## Descrição do Projeto
Trabalho acadêmico que implementa e compara diferentes modelos de regressão para prever a qualidade de vinhos portugueses a partir de características físico-químicas. Desenvolvido como parte do Homework 2 da disciplina de Inteligência Computacional Aplicada, o trabalho inclui implementações manuais e com bibliotecas especializadas de quatro abordagens: Regressão Linear Ordinária (OLS), Regressão Ridge (L2), Regressão PLS e Rede Neural Artificial.

## Dataset
- Conjunto de Dados: Wine Quality
- Fonte: UCI Machine Learning Repository
- Amostras: 6.497 observações (1.599 vinhos tintos + 4.898 vinhos brancos)
- Características: 11 atributos físico-químicos quantitativos
- Variável alvo: Qualidade (escala contínua de 3 a 9) - **problema de regressão**
- Divisão: 75% treino (4.872) / 25% teste (1.625)

## Tecnologias e Dependências
- Python 3.7 ou superior
- Bibliotecas principais:
  - numpy - Cálculos numéricos e álgebra linear
  - scipy - Estatísticas (skewness, correlação)
  - matplotlib - Visualização de gráficos
  - scikit-learn - Pré-processamento e modelos de referência
  - pandas - Manipulação de dados (para algumas análises)
  - tabulate - Formatação de tabelas no console

## Execução do Código
- Primeiro instale as dependências:
  - pip install numpy scipy matplotlib scikit-learn pandas tabulate
- Depois execute o script principal:
  - python codigo_hw2.py

- O script realizará automaticamente:
   - Download dos datasets (se não existirem)
   - Análise exploratória e pré-processamento
   - Treinamento dos 4 modelos com validação cruzada
   - Geração de gráficos e tabelas de resultados
   - Comparação final de desempenho

## Arquivos Gerados pelo Código
- outputs_hw2/: Contém resultados detalhados em CSV e TXT.
- figures_hw2/: Contém todos os gráficos em PNG (histogramas, boxsplots, heatmaps, etc).
# WINE QUALITY
Avaliação de uma base de dados de qualidade de vinhos utilizando métodos de machine learning.

Os estudos apresentados no jupyter-notebook que acompanha este projeto foram desenvolvidos considerando as seguintes ferramentas computacionais e respectivas versões:

- python versão 3.5
- anaconda versão 4.2.0
- pandas versão 0.23.4
- scikit-learn versão 0.20.3
- seaborn versão 0.9.0
- MORD versão 0.6

Caminho da base de dados utilizada: https://drive.google.com/open?id=1-oG5-kBt9xQ3Li4PEexpiA9_7RZhRM1f

### RESPOSTAS AOS PONTOS LEVANTADOS:

#### a. Como foi a definição da sua estratégia de modelagem?

O problema proposto de classificação de qualidade de vinhos a partir de propriedades físico químicas corresponde a um caso especial, em que os valores de score correspondem a uma distribuição ordinal de valores inteiros atribuídos a partir de critérios baseados em níveis de preferência humana em uma escala arbitrária. Dessa forma, tanto o tratamento deste sistema como um problema de classificação ou como de regressão não correspondem à abordagem correta, mas resultam somente em soluções aproximadas para construção de modelos preditivos. Ao tratar predição de valores ordinais como um problema de classificação, a natureza dos métodos de classificação exige que a ordem entre as múltiplas "classes" de score seja ignorada. Por outro lado, ao tratar o sistema como um problema de regressão, é nessário considerar que a escala de valores discretos seja uniforme. Ou seja, é necessário considerar que a "similaridade" ou os critérios para definição dos intervalos da escala da variável resposta seja homogênea e linearmente proporcional a uma escala contínua de valores. Apesar de em muitas aplicações práticas o uso de modelos de regressão se mostrar uma aproximação razoável e capaz de gerar resultados bastante satisfatórios, é preciso tomar bastante cuidado ao considerar que critérios de preferência atribuídos por humanos (como no caso da qualidade do vinho) são definidos de forma gradual e homogênea para incremento do score final.
Devido às características do problema de classificação ordinal serem bastante específicas e ignoradas por muitos no desenvolvimento de modelos, como já descrito, o número de ferramentas de machine learning disponíveis é ainda bastante escasso. Tendo esta observação como justificativa, o método inicialmente escolhido para este estudo foi o de regressão logística ordinal (all-threshold) proposto por Renniee Srebro (ver: https://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf) e implementado na biblioteca MORD (https://pypi.org/project/mord/).
A partir dos modelos iniciais com métricas de capacidade de predição pouco satisfatórias (MAE~0,5 e acurácia~0,56) obtidos com métodos   disponíveis na biblioteca MORD, estudos foram desenvolvidos para avaliar (i) se a baixa capacidade preditiva seria devido à falta de amostragem nas regiões limítrofes da escala de score como variável resposta e (ii) se os modelos implementados não apresentavam a capacidade necessária para identificar padrões mais complexos de dependência entre variáveis e variável resposta e (iii) se, após todos os testes, seria possível considerar a baixa consistência da base de dados fornecida como razão da baixa performance apresentada pelos modelos.
Para isso, modelos mais sofisticados foram implementados considerando a aproximação do problema para casos de classificação ou regressão.
Estes estudos permitiram chegar nas seguintes conclusões: (i) uma vez que métodos de random forest e gradient boosting resultaram em modelos com capacidade preditiva consideramente maior que os modelos iniciais, é possível que o conjunto de dados necessite considerar uma intricada relação entre as variáveis físico-químicas para estimativa de qualidade de vinhos; e (ii) a escala utilizada para score de vinhos é linear, monotônica e homogênea o suficiente para a aproximação do sistema por modelos de regressão.

#### b. Como foi definida a função de custo utilizada?

A função de custo (loss) definida para otimizar e avaliar a qualidade dos modelos preditivos foi o erro médio absoluto (neg_mean_absolute_error, valor negativo na implementação, já que valores mais próximos de zero correspondem a melhores modelos), uma vez que ele é bastante intuitivo, de fácil implementação e amplamente aplicado em métodos de machine-learning baseados em regressão.

#### c. Qual foi o critério utilizado na seleção do modelo final?

Para cada uma das técnicas utilizadas, parâmetros foram explorados dentro de faixas selecionadas de acordo com experiências prévias, a fim de encontrar o melhor modelo possível com capacidade preditiva real, baseando-se em métricas e validação externa. Uma vez que o sistema pode ser interpretado tanto como um problema de regressão como de classificação, as métricas principais utilizadas para avaliar a capacidade preditiva dos modelos foi o erro médio absoluto (mae) e a acurácia dos modelos. Outras métricas também avaliadas para estimar o quão o modelo está balanceado (sem viés para maior efetividade a um único grupo ou classe específica) foi o F1-score, correspondendo à média harmônica entre precisão e sensibilidade do modelo. O melhor modelo selecionado foi aquele que apresentou: o menor valor possível de MAE e os maiores valores possíveis de acurácia e F1-score para todas as categorias em geral.

#### d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Para validar cada um dos modelos, o processo escolhido foi: (i) a separação do conjunto total em um conjunto treino e um teste holdout, (ii) a otimização do modelo a partir de uma validação cruzada (3-fold) dividindo o conjunto treino em novos conjuntos treino e teste e (iii) a validação mais criteriosa do conjunto otimizado no conjunto teste inicialmente gerado. Este processo tem como finalidade garantir a seleção de um modelo com capacidade de generalização e de capacidade preditiva real para amostras desconhecidas. 

#### e. Quais evidências você possui de que seu modelo é suficientemente bom?

O melhor modelo obtido apresentou valores de MAE 0.43 e acurácia de 0.67. Por essas métricas podemos considerar que o modelo possui uma capacidade de descriminar o escore correto entre valores vizinhos, ou seja, com erro de estimativa menor que metade da diferença de valor entre scores vizinhos (i.e. MAE < 0,5). Além disso, podemos estimar que o modelo possui uma acurácia em identificar o score correto entre 2 valores bem acima da aleatoriedade (i.e. acurácia > 0,5).

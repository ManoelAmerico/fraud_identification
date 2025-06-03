# 🕵️‍♂️ Credit Card Fraud Detection: A Strategic Analysis and Predictive Modeling / 🕵️‍♂️ Detecção de Fraude em Cartão de Crédito: Uma Análise Estratégica e Modelagem Preditiva

## 🎯 Strategic Introduction / Introdução Estratégica

**English:**
Credit card fraud extends beyond mere financial loss, significantly impacting consumer trust and the reputation of financial institutions. This project delves into the complex task of identifying fraudulent transactions within a large, anonymized dataset. The core challenge is twofold: (1) the **extreme class imbalance**, where fraudulent transactions are rare events, necessitating specialized techniques to prevent modeling bias, and (2) the **anonymized nature of predictive features** (V1-V28), resulting from Principal Component Analysis (PCA), which imposes specific considerations on model interpretability and applicability. This document outlines the analytical journey, from in-depth data exploration to the rigorous construction and evaluation of predictive models, culminating in actionable insights and guidelines for future investigations.

**Português:**
A fraude em cartões de crédito transcende a mera perda financeira, impactando a confiança do consumidor e a reputação das instituições. Este projeto se debruça sobre a complexa tarefa de identificar transações fraudulentas em um conjunto de dados massivo e anonimizado. O cerne do desafio reside em duas frentes principais: (1) a **extrema desproporção de classes**, onde fraudes são eventos raros, exigindo técnicas especializadas para evitar vieses de modelagem, e (2) a **natureza anonimizada das features preditivas** (V1-V28), resultantes de uma Transformação de Análise de Componentes Principais (PCA), o que impõe considerações específicas sobre a interpretabilidade e aplicabilidade do modelo. Este documento delineia a jornada analítica, desde a exploração aprofundada dos dados até a construção e avaliação rigorosa de modelos preditivos, culminando em insights acionáveis e diretrizes para futuras investigações.

---

## 📁 Repository Structure: Foundations for Reproducibility and Collaboration / 📁 Estrutura do Repositório: Fundamentos para Reprodutibilidade e Colaboração

**English:**
Project organization is paramount to ensuring clarity, reproducibility, and scalability of the analysis. The adopted structure is as follows:

```
.
├── notebooks/ # Epicenter of analysis and modeling
│ ├── 1_Card_Fraud_EDA.ipynb # In-depth Exploratory Data Analysis (EDA)
│ └── 2_Model.ipynb # Model development, training, and evaluation
├── data/ # Centralized data management
│ ├── raw/ # Raw, unaltered data (e.g., creditcard.csv.zip)
│ └── processed/ # Transformed and prepared data (e.g., PROCESSED_DATA)
├── src/ # Support modules for optimization and standardization
│ ├── config.py # Global parameters and data paths
│ ├── graphics.py # Custom functions for impactful visualizations
│ ├── models.py # Utilities for model training and evaluation
│ └── utils.py # General-purpose helper functions
├── models/ # Persistence of trained models (e.g., FINAL_MODEL.joblib)
├── README.md # Central project documentation (this file)
└── ... # Other artifacts (e.g., .gitignore, requirements.txt, Dockerfile)
```

**Português:**
A organização do projeto é fundamental para garantir a clareza, reprodutibilidade e escalabilidade da análise. A estrutura adotada é a seguinte:

```
.
├── notebooks/ # Epicentro da análise e modelagem
│ ├── 1_Card_Fraud_EDA.ipynb # Análise Exploratória de Dados (AED) aprofundada
│ └── 2_Model.ipynb # Desenvolvimento, treinamento e avaliação de modelos
├── data/ # Gerenciamento centralizado dos dados
│ ├── raw/ # Dados brutos e inalterados (ex: creditcard.csv.zip)
│ └── processed/ # Dados transformados e preparados (ex: PROCESSED_DATA)
├── src/ # Módulos de suporte para otimização e padronização
│ ├── config.py # Parâmetros globais e caminhos de dados
│ ├── graphics.py # Funções customizadas para visualizações de impacto
│ ├── models.py # Utilitários para treinamento e avaliação de modelos
│ └── utils.py # Funções auxiliares de propósito geral
├── models/ # Persistência de modelos treinados (ex: FINAL_MODEL.joblib)
├── README.md # Documentação central do projeto (este arquivo)
└── ... # Demais arquivos (ex: .gitignore, requirements.txt, Dockerfile)
```

---

## 🗺️ Data Exploration and Discoveries (EDA): Unveiling Hidden Patterns / 🗺️ Exploração e Descobertas dos Dados (AED): Desvendando os Padrões Ocultos

**English:**
Exploratory Data Analysis (EDA), conducted in the `notebooks/1_Card_Fraud_EDA.ipynb` notebook, was the cornerstone for understanding transaction dynamics and formulating strategic hypotheses. The dataset comprises the `Time` feature (elapsed time since the first transaction), `Amount` (transaction value), and a set of 28 anonymized features (`V1` to `V28`) derived from PCA.

### 📊 Key Strategic EDA Findings:

1.  **Data Integrity and Quality:** A rigorous initial check confirmed the absence of missing values, ensuring a solid foundation for subsequent analyses.
2.  **Critical Class Imbalance:** The most impactful discovery was the severe imbalance in the target variable `Class`. Fraudulent transactions (`Class == 1`) represent a minimal fraction of the total volume (approximately 0.1727%). This finding is crucial, as models naively trained on imbalanced data tend to perform well on the majority class but fail to detect the minority class (frauds), which is the primary objective. Visualizing this disparity using a count plot with a logarithmic y-axis was essential for communicating the magnitude of the challenge.
    - `![Class Distribution Plot](URL_PLACEHOLDER_FOR_CLASS_DISTRIBUTION_PLOT)`
3.  **Visualization Strategy for Comparability:** Given the class disparity, direct comparisons of feature distributions would be overshadowed by the majority class. To circumvent this limitation and obtain reliable insights, a balanced subsample (`df_sample`) was created. This sample consisted of all 492 fraud instances and an equivalent number of randomly selected legitimate transactions. This approach allowed for an effective comparative analysis of the distinctive characteristics of fraudulent transactions.
4.  **Feature Discrimination by Class:** Using the balanced sample, comparative histograms were generated for all 30 features (`Time`, `Amount`, `V1-V28`). The analysis of these side-by-side distributions (Class 0 vs. Class 1) revealed that several PCA features, in addition to `Amount`, exhibit distinct patterns between legitimate and fraudulent transactions. These features were therefore identified as promising candidates for the modeling phase, possessing high predictive potential.
    - `![Histograms of Features by Class](URL_PLACEHOLDER_FOR_HISTOGRAMS_FEATURES_PLOT)`

**Português:**
A Análise Exploratória de Dados (AED), conduzida no notebook `notebooks/1_Card_Fraud_EDA.ipynb`, constituiu a pedra angular para a compreensão da dinâmica das transações e para a formulação de hipóteses estratégicas. O dataset é composto pelas features `Time` (tempo decorrido desde a primeira transação), `Amount` (valor da transação) e um conjunto de 28 features anonimizadas (`V1` a `V28`), oriundas de uma transformação PCA.

### 📊 Principais Achados Estratégicos da AED:

1.  **Integridade e Qualidade dos Dados:** Uma verificação inicial rigorosa confirmou a ausência de valores ausentes, assegurando uma base sólida para as análises subsequentes.
2.  **Desbalanceamento Crítico de Classes:** A descoberta mais impactante foi a severa desproporção na variável alvo `Class`. Transações fraudulentas (`Class == 1`) representam uma fração mínima do volume total (aproximadamente 0,1727%). Este achado é crucial, pois modelos treinados ingenuamente em dados desbalanceados tendem a performar bem na classe majoritária, mas falham em detectar a classe minoritária (fraudes), que é o objetivo primário. A visualização desta disparidade, utilizando um gráfico de contagem com eixo y em escala logarítmica, foi essencial para comunicar a magnitude do desafio.
    - `![Gráfico de Distribuição das Classes](URL_PLACEHOLDER_PARA_GRAFICO_DISTRIBUICAO_CLASSES)`
3.  **Estratégia de Visualização para Comparabilidade:** Dada a disparidade de classes, comparações diretas das distribuições de features seriam ofuscadas pela classe majoritária. Para contornar essa limitação e obter insights fidedignos, foi criada uma subamostra balanceada (`df_sample`). Esta amostra consistiu em todas as 492 instâncias de fraude e um número equivalente de transações legítimas, selecionadas aleatoriamente. Essa abordagem permitiu uma análise comparativa eficaz das características distintivas das transações fraudulentas.
4.  **Discriminação de Features por Classe:** Utilizando a amostra balanceada, foram gerados histogramas comparativos para todas as 30 features (`Time`, `Amount`, `V1-V28`). A análise dessas distribuições lado a lado (Classe 0 vs. Classe 1) revelou que diversas features PCA, além de `Amount`, exibem padrões distintos entre transações legítimas e fraudulentas. Essas features, portanto, foram identificadas como candidatas promissoras para a etapa de modelagem, possuindo alto potencial preditivo.
    - `![Histogramas das Features por Classe](URL_PLACEHOLDER_PARA_HISTOGRAMAS_FEATURES_PLOT)`

---

## ⚙️ Predictive Modeling: From Preparation to Strategic Model Selection / ⚙️ Modelagem Preditiva: Da Preparação à Seleção Estratégica de Modelos

**English:**
The modeling phase, detailed in `notebooks/2_Model.ipynb`, was conducted with a focus on building robust classifiers and critically evaluating their performance, especially in the context of the identified class imbalance.

### 🛠️ Preprocessing and Feature Engineering: Maximizing Data Potential

Proper data preparation is an indispensable precursor to successful modeling. The following transformations were strategically applied:

- **`Time`**: Scaled using `MinMaxScaler` to normalize the value range, ensuring that magnitude did not unduly influence scale-sensitive algorithms.
- **`Amount`**: Given its common skewness in financial data, it was transformed with `PowerTransformer` (specifically, the Yeo-Johnson transformation, which handles positive and negative values well, though `Amount` is positive here), aiming to approximate a normal distribution and stabilize variance.
- **PCA Features (`V1`-`V28`)**: Scaled with `RobustScaler`. This choice is strategic as `RobustScaler` is less sensitive to outliers, a potential characteristic in PCA-derived features and crucial in anomaly detection like fraud.
- **`ColumnTransformer`**: All transformations were encapsulated in a `ColumnTransformer`, ensuring a consistent, reproducible, and easily applicable preprocessing pipeline for new data (either in cross-validation or future deployment).

### 🧠 Model Selection and Evaluation: Methodological Rigor

A spectrum of classification algorithms was systematically trained and evaluated. Cross-validation was performed using `StratifiedKFold` (with 5 splits), an essential technique for imbalanced data as it preserves the original class proportions in each training and testing fold, ensuring a more realistic model performance assessment.

- **Models Considered:**
  - `DummyClassifier` (with `stratified` strategy): Served as a fundamental baseline, indicating the minimum expected performance.
  - `LogisticRegression` (with `class_weight="balanced"`): A robust linear model, adjusted to penalize errors in the minority class more heavily.
  - `DecisionTreeClassifier` (with `class_weight="balanced"`): A tree-based model, also adjusted for imbalance.
  - `LGBMClassifier` and `XGBClassifier`: Gradient boosting models known for their high performance. The `scale_pos_weight` parameter was strategically employed.
  - `KNeighborsClassifier`: An instance-based model to explore neighborhood patterns.
- **Strategies for Class Imbalance:**
  - For `LogisticRegression` and `DecisionTreeClassifier`, the `class_weight="balanced"` parameter automatically adjusts weights inversely proportional to class frequencies.
  - For the `LGBMClassifier` and `XGBClassifier` ensembles, the `scale_pos_weight` parameter was calculated as the ratio of non-fraudulent to fraudulent instances (approximately 520.5 in this dataset). This hyperparameter amplifies the importance of the minority class (fraud) during training, forcing the model to pay closer attention to these cases.
- **Key Performance Metrics:** Given the nature of the problem, simple accuracy is a misleading metric. Evaluation focused on indicators that offer a more complete and relevant perspective for fraud detection:
  - **Average Precision (AP) / AUCPR**: Particularly informative for imbalanced datasets, summarizing the Precision-Recall curve.
  - **ROC AUC**: Measures the model's ability to distinguish between classes.
  - **F1-Score**: Harmonic mean of Precision and Recall, seeking a balance between them.
  - **Recall (Sensitivity / True Positive Rate for Fraud)**: A critical metric, indicating the proportion of actual frauds the model successfully identified. Maximizing Recall is often a primary goal in fraud detection, even at the cost of an increase in false positives (which must be managed).
  - **Precision**: Proportion of transactions classified as fraud that are actually fraudulent.
  - **Balanced Accuracy**: Arithmetic mean of Recall for each class, useful in imbalanced scenarios.

The model with the most promising performance (`FINAL_MODEL`), holistically assessed through these metrics, was serialized for future use.

### 📊 Visualizing Model Performance: Clarity and Comparability

Interpretation of model performance was enriched through strategic visualizations:

- **Comparative Bar Chart:** A direct comparison of key metrics (e.g., Average Precision, ROC AUC, Recall for the fraud class) across all evaluated models, facilitating the identification of the superior candidate(s).
  - `![Model Comparison Plot](URL_PLACEHOLDER_FOR_MODEL_COMPARISON_PLOT)`

The `df_results` dataframe, generated in the `2_Model.ipynb` notebook, consolidates all these metrics, serving as a detailed record for auditing and future iterations.

**Português:**
A fase de modelagem, detalhada em `notebooks/2_Model.ipynb`, foi conduzida com foco na construção de classificadores robustos e na avaliação criteriosa de sua performance, especialmente no contexto do desbalanceamento de classes identificado.

### 🛠️ Pré-processamento e Engenharia de Features: Maximizando o Potencial dos Dados

A preparação adequada dos dados é um precursor indispensável para o sucesso da modelagem. As seguintes transformações foram aplicadas de forma estratégica:

- **`Time`**: Escalonado utilizando `MinMaxScaler` para normalizar o intervalo de valores, garantindo que a magnitude não influenciasse indevidamente os algoritmos sensíveis à escala.
- **`Amount`**: Dada sua assimetria comum em dados financeiros, foi transformado com `PowerTransformer` (especificamente, a transformação de Yeo-Johnson, que lida bem com valores positivos e negativos, embora aqui `Amount` seja positivo), buscando aproximar a distribuição de uma normal e estabilizar a variância.
- **Features PCA (`V1`-`V28`)**: Escalonadas com `RobustScaler`. Esta escolha é estratégica, pois o `RobustScaler` é menos sensível a outliers, uma característica potencialmente presente em features resultantes de PCA e crucial em detecção de anomalias como fraude.
- **`ColumnTransformer`**: Todas as transformações foram encapsuladas em um `ColumnTransformer`, assegurando um pipeline de pré-processamento consistente, reprodutível e facilmente aplicável a novos dados (seja em validação cruzada ou em um futuro deploy).

### 🧠 Seleção e Avaliação de Modelos: Rigor Metodológico

Um espectro de algoritmos de classificação foi sistematicamente treinado e avaliado. A validação cruzada foi realizada utilizando `StratifiedKFold` (com 5 folds), uma técnica essencial para dados desbalanceados, pois preserva a proporção original das classes em cada fold de treinamento e teste, garantindo uma avaliação mais realista do desempenho do modelo.

- **Modelos Considerados:**
  - `DummyClassifier` (com estratégia `stratified`): Serviu como um baseline fundamental, indicando o desempenho mínimo esperado.
  - `LogisticRegression` (com `class_weight="balanced"`): Um modelo linear robusto, ajustado para penalizar mais os erros na classe minoritária.
  - `DecisionTreeClassifier` (com `class_weight="balanced"`): Um modelo baseado em árvore, também ajustado para o desbalanceamento.
  - `LGBMClassifier` e `XGBClassifier`: Modelos de gradient boosting, conhecidos por sua alta performance. O parâmetro `scale_pos_weight` foi estrategicamente empregado.
  - `KNeighborsClassifier`: Um modelo baseado em instância, para explorar padrões de vizinhança.
- **Estratégias para Desbalanceamento de Classes:**
  - Para `LogisticRegression` e `DecisionTreeClassifier`, o parâmetro `class_weight="balanced"` ajusta automaticamente os pesos inversamente proporcionais às frequências das classes.
  - Para os ensemble `LGBMClassifier` e `XGBClassifier`, o parâmetro `scale_pos_weight` foi calculado como a razão entre o número de instâncias não fraudulentas e fraudulentas (aproximadamente 520,5 neste dataset). Esse hiperparâmetro amplifica a importância da classe minoritária (fraude) durante o processo de treinamento, forçando o modelo a dar maior atenção a esses casos.
- **Métricas de Performance Chave:** Dada a natureza do problema, a acurácia simples é uma métrica enganosa. A avaliação concentrou-se em indicadores que oferecem uma perspectiva mais completa e relevante para a detecção de fraude:
  - **Average Precision (AP) / AUCPR**: Particularmente informativa para datasets desbalanceados, resume a curva Precision-Recall.
  - **ROC AUC**: Mede a capacidade do modelo de distinguir entre as classes.
  - **F1-Score**: Média harmônica de Precision e Recall, buscando um equilíbrio entre ambos.
  - **Recall (Sensibilidade / Taxa de Verdadeiros Positivos para Fraude)**: Métrica crítica, pois indica a proporção de fraudes reais que o modelo conseguiu identificar. Maximizar o Recall é frequentemente um objetivo primário em detecção de fraude, mesmo que à custa de um aumento nos falsos positivos (que devem ser gerenciados).
  - **Precision**: Proporção de transações classificadas como fraude que são de fato fraudulentas.
  - **Balanced Accuracy**: Média aritmética do Recall para cada classe, útil em cenários desbalanceados.

O modelo com o desempenho mais promissor (`FINAL_MODEL`), avaliado holisticamente através dessas métricas, foi serializado para uso futuro.

### 📊 Desempenho Visual dos Modelos: Clareza e Comparabilidade

A interpretação do desempenho dos modelos foi enriquecida por meio de visualizações estratégicas:

- **Gráfico de Barras Comparativo:** Um comparativo direto das principais métricas (ex: Average Precision, ROC AUC, Recall para a classe fraude) entre todos os modelos avaliados, facilitando a identificação do(s) candidato(s) superior(es).
  - `![Gráfico de Comparação de Modelos](URL_PLACEHOLDER_PARA_GRAFICO_COMPARACAO_MODELOS)`

O dataframe `df_results`, gerado no notebook `2_Model.ipynb`, consolida todas essas métricas, servindo como um registro detalhado para auditoria e futuras iterações.

---

## 📝 Considerations on Model Applicability and PCA Implications / 📝 Considerações sobre a Aplicabilidade do Modelo e Implicações da PCA

**English:**
It is crucial to align expectations regarding the operationalization of the developed model. The primary characteristic of this dataset – the anonymization of features `V1` to `V28` via PCA – imposes important considerations:

- **Dependency on the Original PCA Transformation:** The trained model expects, as input, data that has undergone the _exact same PCA transformation_ applied to the original dataset. This includes the same mean, standard deviation (if standardization was part of the pre-PCA pipeline), and the same principal components.
- **Challenges in Applying to New Raw Data:** Without access to the original (pre-PCA) data or the parameters and components of the specific PCA transformation used to generate this dataset, it is not possible to convert new transactions (with their original features, such as merchant type, geolocation, etc.) into the `V1-V28` format required by the model.

Consequently, this project demonstrates the ability to build an effective fraud detector _given the anonymized data format_. Deployment in a production scenario with raw transactional data would require possession of, or the ability to replicate, the original PCA pipeline.

**Português:**
É crucial alinhar as expectativas quanto à operacionalização do modelo desenvolvido. A principal característica deste dataset – a anonimização das features `V1` a `V28` via PCA – impõe considerações importantes:

- **Dependência da Transformação PCA Original:** O modelo treinado espera, como entrada, dados que passaram pela _exata mesma transformação PCA_ aplicada ao dataset original. Isso inclui a mesma média, desvio padrão (se a padronização foi parte do pré-PCA) e os mesmos componentes principais.
- **Desafios na Aplicação em Novos Dados Brutos:** Sem acesso aos dados originais (pré-PCA) ou aos parâmetros e componentes da transformação PCA específica utilizada para gerar este dataset, não é possível converter novas transações (com suas features originais, como tipo de estabelecimento, geolocalização, etc.) para o formato `V1-V28` exigido pelo modelo.

Consequentemente, este projeto demonstra a capacidade de construir um detector de fraudes eficaz _dado o formato anonimizado dos dados_. A implantação em um cenário de produção com dados transacionais brutos exigiria a posse ou a capacidade de replicar o pipeline de PCA original.

---

## 🚀 Conclusion, Strategic Recommendations, and Next Steps / 🚀 Conclusão, Recomendações Estratégicas e Próximos Passos

**English:**
This project has successfully demonstrated the feasibility of fraud detection in a complex dataset characterized by PCA anonymization and severe class imbalance. EDA insights were crucial in guiding preprocessing and modeling strategies. The application of techniques such as robust scaling, power transformations, and, fundamentally, addressing class imbalance (via `class_weight` or `scale_pos_weight`) allowed models like XGBoost, LGBM, and Logistic Regression to achieve robust performance, especially on critical metrics like Average Precision and Recall for the minority class.

The anonymized nature of the data (PCA features) is the main limiting factor for a "plug-and-play" application of the model on raw transactional data.

**Português:**
Este projeto demonstrou com sucesso a viabilidade da detecção de fraude em um dataset complexo, caracterizado por anonimização via PCA e severo desbalanceamento de classes. Os insights da AED foram cruciais para direcionar as estratégias de pré-processing e modelagem. A aplicação de técnicas como escalonamento robusto, transformações de potência e, fundamentalmente, o tratamento do desbalanceamento de classes (via `class_weight` ou `scale_pos_weight`) permitiram que modelos como XGBoost, LGBM e Regressão Logística alcançassem um desempenho robusto, especialmente em métricas críticas como Average Precision e Recall para a classe minoritária.

A natureza anonimizada dos dados (features PCA) é o principal fator limitante para uma aplicação "plug-and-play" do modelo em dados transacionais brutos.

# Dicionário de dados

Data Source: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

| Column Name | Description                                                | Data Type |
| ----------- | ---------------------------------------------------------- | --------- |
| `Time`      | Interval in seconds between each transaction and the first | float     |
| `Vx`        | Result of PCA on the original variables                    | float     |
| `Amount`    | Value of each transaction                                  | float     |
| `Class`     | Transaction classification: genuine (0) or fraudulent (1)  | int       |

In the Kaggle dataset description, some explanations about the variables are provided:

- Due to confidentiality issues, the identification of many of the original variables is not available.
- Those represented as V1, V2,... V28 are the result of a Principal Component Analysis (PCA) transformation, a technique used to condense the information contained in several original variables into a smaller set of statistical variables (components) with minimal information loss. In this work, we will see some consequences of this transformation in our analysis.

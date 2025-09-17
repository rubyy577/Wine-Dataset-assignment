# Wine Quality Prediction Project

This project explores various machine learning techniques to predict the quality of Portuguese "Vinho Verde" wine based on its physicochemical properties. The analysis is conducted in two separate Jupyter Notebooks, each tackling the problem from a different angle: one as a binned classification task and the other as a multi-class/regression task.

---
## Dataset

The project uses the **Wine Quality Dataset** from the UCI Machine Learning Repository.

-   **Source**: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. In *Decision Support Systems*, 47(4), 547-553.
-   **Description**: The dataset contains information on both red and white wine samples. There are 11 input variables (physicochemical features like `fixed acidity`, `alcohol`, `pH`, etc.) and one output variable (`quality`, scored between 0 and 10).

---
## Notebooks

This repository contains two main notebooks, each representing a distinct modeling approach.

### 1. `wine_dataset_with_binning.ipynb` (Binned Classification)

This notebook frames the problem as a **3-class classification task**, aiming to categorize wine into 'Poor', 'Average', or 'Good' quality. This approach is effective for dealing with the natural class imbalance of the raw quality scores.

**Methodology:**
-   **Preprocessing**: The `quality` score is binned into three categories:
    -   **Poor**: score < 5
    -   **Average**: score is 5 or 6
    -   **Good**: score > 6
-   **Exploratory Data Analysis**: A correlation heatmap is generated to analyze relationships between features.
-   **Feature Selection**: An A/B test is conducted between `free sulfur dioxide` and `total sulfur dioxide` to handle multicollinearity. The analysis concludes that dropping `total sulfur dioxide` yields slightly better model performance.
-   **Model Training**: The following classification models are trained and evaluated:
    -   Logistic Regression
    -   Random Forest Classifier
    -   XGBoost Classifier
-   **Hyperparameter Tuning**: `GridSearchCV` is used to find the optimal hyperparameters for the Logistic Regression and XGBoost models.

**Key Findings:**
The **XGBoost Classifier** proved to be the most effective model for this approach, demonstrating higher overall accuracy and a significantly better ability to identify the crucial minority classes ('Poor' and 'Good' wines).

### 2. `wine_dataset_without_binning.ipynb` 

This notebook explores predicting the raw `quality` score (ranging from 3 to 9 in the data) without binning. Two main strategies are investigated.

**Methodology:**
-   **Approach 1: Multi-class Classification**
    -   Each integer quality score (3, 4, 5, etc.) is treated as a separate class.
    -   `RandomForestClassifier` and `XGBClassifier` are trained. `LabelEncoder` is used to handle the zero-indexing requirement for XGBoost.
-   **Approach 2: Regression**
    -   The quality score is treated as a continuous value.
    -   A `RandomForestRegressor` is trained to predict the score directly.
    -   Evaluation is performed using regression metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

**Key Findings:**
The multi-class classification approach struggles due to the severe class imbalance (very few wines have scores of 3, 4, 8, or 9). The regression approach provides an alternative perspective, evaluating the models based on their average prediction error rather than strict classification accuracy.

---
## Conclusion

The project demonstrates that framing the problem as a binned classification task and using a powerful gradient boosting model like XGBoost provides the most practical and effective solution for predicting wine quality from the given features.
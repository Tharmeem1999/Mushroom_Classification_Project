# Mushroom Classification Project

This project uses the Mushroom Classification dataset from Kaggle to predict whether a mushroom is **edible** or **poisonous** based on its physical characteristics. Two machine learning models, a **Decision Tree** and a **Logistic Regression**, are implemented to perform this binary classification task.

## Table of Contents

- Project Overview
- Dataset
- Project Steps
- Python Packages Used
- Models
- Results
- Key Insights
- Future Improvements

## Project Overview

The goal of this project is to build and compare two machine learning models to classify mushrooms as **edible (e)** or **poisonous (p)** using features like cap shape, odor, and gill color. The project is designed to be beginner-friendly, using Python and scikit-learn, with visualizations to understand the data and model decisions.

## Dataset

The dataset (`mushrooms.csv`) contains 8,124 mushrooms with 23 categorical features, including:

- **Target Variable**: `class` (e = edible, p = poisonous)
- **Features**: 22 attributes like `cap-shape`, `odor`, `gill-color`, etc.
- **Key Notes**:
  - All features are categorical (text-based).
  - The `stalk-root` column contains missing values denoted by `?`, which are handled by replacing with `unknown`.
  - The dataset is balanced, with roughly equal numbers of edible and poisonous mushrooms.

Data source: Kaggle Mushroom Classification

## Project Steps

1. **Data Loading**: Loaded the dataset using pandas to read `mushrooms.csv`.
2. **Exploratory Data Analysis (EDA)**: Visualized the distribution of the `class` variable and explored relationships between features (e.g., `cap-color`, `odor`) and the target using seaborn count plots.
3. **Data Preprocessing**:
   - Replaced `?` in `stalk-root` with `unknown`.
   - For **Model A (Decision Tree)**: Applied Label Encoding to convert categorical features to numerical values.
   - For **Model B (Logistic Regression)**: Applied One-Hot Encoding to create binary columns for categorical features.
   - Split data into 80% training and 20% testing sets.
4. **Model Training**:
   - Trained a Decision Tree Classifier for Model A.
   - Trained a Logistic Regression model for Model B.
5. **Model Evaluation**: Evaluated both models using accuracy, confusion matrix, and classification report.
6. **Model Interpretation**:
   - For Model A: Visualized the Decision Tree to understand key splits.
   - For Model B: Analyzed classification metrics to confirm performance.

## Python Packages Used

- **pandas**: For data loading and manipulation.
- **scikit-learn**: For preprocessing (LabelEncoder, get_dummies), model training (DecisionTreeClassifier, LogisticRegression), and evaluation (accuracy_score, classification_report, confusion_matrix).
- **matplotlib**: For creating visualizations like the Decision Tree plot.
- **seaborn**: For creating count plots to explore feature-class relationships.

## Models

### Model A: Decision Tree (`Model_A.ipynb`)

- **Algorithm**: Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)
- **Preprocessing**:
  - Replaced `?` in `stalk-root` with `unknown`.
  - Applied **Label Encoding** to convert categorical features to numerical values.
  - Split data into 80% training and 20% testing sets.
- **Training**: Trained with default parameters and `random_state=42`.
- **Evaluation**: Visualized the top 3 levels of the tree using `plot_tree`.
- **Why Used**: Decision Trees are intuitive, handle categorical data well, and are interpretable.

### Model B: Logistic Regression (`Model_B.ipynb`)

- **Algorithm**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Preprocessing**:
  - Replaced `?` in `stalk-root` with `unknown`.
  - Applied **One-Hot Encoding** to convert categorical features to binary columns, suitable for Logistic Regression.
  - Split data into 80% training and 20% testing sets.
- **Training**: Trained with `max_iter=1000` and `random_state=42`.
- **Evaluation**: Analyzed performance using accuracy and classification report.
- **Why Used**: Logistic Regression is a simple, interpretable linear model that performs well on this dataset after one-hot encoding.

## Results

- **Decision Tree (Model A)**:

  - Accuracy: 100% on the test set.
  - The model perfectly classifies all test samples, likely due to strong feature-class correlations.
  - Visualization: The Decision Tree plot shows key features like `odor` or `gill-color` as top splits.

- **Logistic Regression (Model B)**:

  - Accuracy: 100% on the test set.
  - Classification Report:

    ```
              precision    recall  f1-score   support
    Edible       1.00      1.00      1.00       843
    Poisonous    1.00      1.00      1.00       782
    accuracy                           1.00      1625
    ```
  - The model perfectly classifies all test samples, leveraging one-hot encoded features.

## Key Insights

- **Feature Importance**: Features like `odor`, `gill-color`, and `spore-print-color` are critical predictors, as seen in the Decision Tree’s top splits and confirmed by the dataset’s strong feature-class correlations.
- **Model Performance**: Both models achieve 100% accuracy, indicating the dataset has clear patterns (e.g., certain odors like foul are strongly associated with poisonous mushrooms).
- **Preprocessing Impact**: Label Encoding works well for Decision Trees, while One-Hot Encoding is necessary for Logistic Regression to avoid assuming ordinal relationships.
- **Dataset Characteristics**: The balanced dataset and lack of significant noise contribute to perfect classification performance.

## Future Improvements

- **Feature Selection**: Use feature importance scores (e.g., `model.feature_importances_` for Decision Tree) to reduce model complexity by dropping less impactful features.
- **Cross-Validation**: Implement k-fold cross-validation to ensure robust performance across different data splits.
- **Other Models**: Experiment with Random Forest or Support Vector Machines (SVM) for comparison.
- **Hyperparameter Tuning**: Tune Decision Tree parameters (`max_depth`, `min_samples_split`) or Logistic Regression’s regularization (`C`) for potential improvements.
- **Missing Value Handling**: Explore imputation techniques for `stalk-root` instead of using `unknown` to assess impact on performance.

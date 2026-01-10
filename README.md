# Titanic Survival Prediction — Logistic Regression


This project demonstrates an end-to-end machine learning workflow, 
including exploratory data analysis, feature engineering, pipeline-based modeling, 
and evaluation on a real-world classification problem.


 **Kaggle Notebook:** https://www.kaggle.com/code/selimsnmn/titanic-survival-prediction-logistic-regression


---

## Project Summary

- **Project Objective:** To predict whether Titanic passengers survived or not.
- **Prediction Type:** Binary classification (survived = 0 or 1).  
- **Model:** Logistic Regression classifier.  
- **Data Source:** Kaggle Titanic dataset.

---

## Data Set Description

The dataset contains the following information about the passengers on the Titanic:

- **Survived:** Target variable indicating whether the passenger survived
(0 = No, 1 = Yes)
- **Pclass:** Passenger class
(1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- **Name:** Full name of the passenger
- **Sex:** Gender of the passenger
- **Age:** Age of the passenger in years
- **SibSp:** Number of siblings or spouses aboard the Titanic
- **Ticket:** Ticket number
- **Fare:** Passenger fare
- **Cabin:** Cabin number assigned to the passenger
- **Embarked:** Port of embarkation
(C = Cherbourg, Q = Queenstown, S = Southampton)


---

## Exploratory Data Analysis (EDA)

The dataset consists of 12 variables pertaining to 891 passengers. In the initial review, data types, missing values, and general distributions were analyzed.

- **Missing Values:**
  A high proportion of missing data was detected in the Age and especially Cabin variables. The Cabin variable was excluded from the model.
- **Survival Distribution:**
  The number of passengers who did not survive is higher; however, the data set is not excessively unbalanced.
- **Age Distribution:**
  It is observed that most passengers are between the ages of 20 and 40. The number of passengers gradually decreases with increasing age.
- **Survival Count by Sex:**
  Female passengers have a higher survival rate compared to male passengers. This observation suggests that gender may be an important factor related to survival and may reflect evacuation priorities during disasters.
- **Age Distribution by Survival Status:**
  The age distributions for survivors and non-survivors show noticeable overlap. However, younger passengers, particularly children, appear to have a relatively higher likelihood of survival.
- **Survival Rate by Pclass:**
  Survival rates vary across passenger classes, with first-class passengers showing higher survival rates compared to second- and third-class passengers. This pattern suggests a possible association between socioeconomic status and survival.
- **Fare Distribution by Survival Status:**
  Passengers who paid higher fares tend to exhibit higher survival rates. Fare may therefore serve as a proxy for socioeconomic status and access to safer locations on the ship.

These analyses have guided the data preprocessing and the establishment of the Logistic Regression model.


---

## Feature Engineering

The following feature engineering steps were applied to improve model performance and capture meaningful structures in the data:

- **Title Extraction (Name → Title):**
  Due to the high cardinality of the `name` variable, it was not used directly; instead, passenger titles (`Mr.`, `Mrs.`, `Miss`, etc.) were extracted.
  Rare titles were grouped under a single “`Rare`” category to reduce sparsity.
- **Age Imputation (Group-Based):**
  Missing age values were imputed using the median age within each title group. This group-based approach helps preserve demographic structure and provides a more realistic imputation compared to global statistics.
- **Embarked Imputation:**
  Missing values in the Embarked feature were filled using the most frequent category (mode). This approach is appropriate for categorical variables and helps maintain the original distribution.
- **TicketGroupSize:**
  A new feature, TicketGroupSize, was created to capture the number of passengers sharing the same ticket. This feature may reflect group or family travel patterns, which could be associated with survival outcomes.
- **FamilySize & IsAlone:**
  `FamilySize` was constructed by combining the number of siblings/spouses and parents/children aboard.

  Based on this feature, a binary `IsAlone` indicator was derived to distinguish passengers traveling alone from those traveling with family members.
- **Feature Reduction:**
  Several features were removed after feature engineering to reduce redundancy and noise. These included identifiers, high-cardinality features, and variables whose information was captured by engineered features.

As a result of these steps, a smaller, more meaningful feature set that is more suitable for the Logistic Regression model has been obtained.

---


## Modeling & Evaluation

During the modeling process, scikit-learn Pipeline was used to prevent data leakage and ensure a consistent structure.

- **Feature Separation:**
  Numeric (`Pclass`, `Age`, `Fare`, `TicketGroupSize`, `FamilySize`, `IsAlone`) and categorical (`Sex`, `Embarked`, `Title`) variables have been defined separately.
- **Train–Test Split:**
  The dataset is split into `80%` training and `20%` testing, with reproducibility ensured using `random_state`.
- **Preprocessing Pipeline:**
  - One-Hot Encoding for Categorical Variables
  - Standard Scaling for Numerical Variables

  These steps have been combined into a single pipeline using ColumnTransformer.
- **Model:**
  Logistic Regression has been preferred as a powerful and interpretable baseline model for tabular and binary classification problems.


---

## Model Performance

Basic metrics obtained on the test set:

  - Accuracy: ~0.81
  - F1-Score: ~0.77
  - ROC-AUC: ~0.89

The confusion matrix shows that the model classifies both survivors and non-survivors with a reasonable balance.


---

## Hyperparameter Tuning

- Method: RandomizedSearchCV
- Validation: Stratified K-Fold Cross-Validation
- Result:
  Hyperparameter optimization has provided limited improvement compared to the baseline model. This indicates that Logistic Regression is already a strong starting model for this problem.


---

## Conclusion

  - The Logistic Regression model produced simple, interpretable, and consistent results on the Titanic dataset.
  - Demographic and socio-economic characteristics were observed to be decisive for survival. Tree-based models can be evaluated for further performance improvement.
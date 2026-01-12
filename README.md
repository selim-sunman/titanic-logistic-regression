# Titanic Survival Prediction

This project is a simple end-to-end machine learning pipeline built on the Kaggle Titanic dataset.  
The main goal is to practice writing clean, modular ML code and to demonstrate a basic training and tuning workflow.

I focused on code readability and structure rather than complex modeling.

---

## Project Overview

In this project, I:
- Learn how to structure an ML project using Python modules
- Apply basic feature engineering
- Train a Logistic Regression model using a scikit-learn Pipeline
- Optionally tune hyperparameters with RandomizedSearchCV
- Evaluate the model using common classification metrics

---

## File Descriptions

- `data.py`: loads the dataset and handles basic cleaning  
- `features.py`: feature engineering (Title extraction, FamilySize, IsAlone, etc.)  
- `model.py`: defines the ML pipeline and RandomizedSearchCV  
- `evaluation.py`: evaluation metrics and reports  
- `utils.py`: helper functions (seed setting, column checks, saving models)  
- `train.py`: baseline training script  
- `tune.py`: training with RandomizedSearchCV  

---

## Setup

First, create and activate a virtual environment.

### Windows
```powershell```
python -m venv venv
venv\Scripts\activate


### macOS / Linux
python -m venv venv
source venv/bin/activate


#### Install the required packages:
pip install -r requirements.txt

---

## How to Run

#### Baseline training
Runs the model without hyperparameter tuning.

python -m src.train


#### Hyperparameter tuning
Runs RandomizedSearchCV to search for better hyperparameters.

python -m src.tune


The trained model is saved under:

models/titanic_model_tuned.joblib

---

## Model Details

- Model: Logistic Regression
- Preprocessing:
  - Numerical features are scaled using StandardScaler
  - Categorical features are encoded using OneHotEncoder
- Cross-validation:
  - StratifiedKFold with 5 splits
- Hyperparameter tuning:
  - RandomizedSearchCV
  - Optimized metric: F1-score

---

## Results

Example results from a tuned run:

  - Accuracy: ~0.83
  - ROC-AUC: ~0.87

Results may vary slightly depending on the random split and environment.

---

## Notes

This project was built mainly for learning and portfolio purposes.

I focused on writing clean and readable code and keeping the overall pipeline simple and reproducible

---

## Possible Improvements
- Try different classification models and compare their performance
- Add more feature engineering and analyze their impact on the model
- Perform more systematic hyperparameter tuning
- Improve model evaluation with cross-validation on the final estimator
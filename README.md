# Titanic Survival Prediction

This project is a beginner-friendly machine learning pipeline built using the Kaggle Titanic dataset.

The main purpose of this project is to practice core machine learning concepts and to learn how to structure a small ML project using clean and readable Python code.

---

## Project Goals

In this project, I aimed to:

Understand how to structure a machine learning project using Python modules

Apply basic feature engineering techniques

Train a classification model using scikit-learn

Learn how to evaluate model performance using common metrics

Practice reproducible ML workflows

---

## Dataset

- Dataset: Kaggle Titanic Dataset
- Task: Binary classification
- Target variable: Survived

---


## Project Structure

src/

├── data.py        # Load and clean the dataset<br>
├── features.py    # Feature engineering<br>
├── model.py       # Model and pipeline definition<br>
├── evaluation.py  # Evaluation utilities<br>
├── utils.py       # Helper functions<br>
├── train.py       # Baseline training script<br>
└── tune.py        # Hyperparameter tuning script<br>

---

## Features

Some simple feature engineering steps include:
- Extracting passenger titles from names
- Creating FamilySize and IsAlone features
- Encoding categorical variables
- Scaling numerical variables


---

## Model

- Model used: Logistic Regression
- Preprocessing:
  - Numerical features are scaled using StandardScaler
  - Categorical features are encoded using OneHotEncoder
- Training:
  - StratifiedKFold cross-validation
  - Optional hyperparameter tuning using RandomizedSearchCV
- Optimization metric: F1-score

Logistic Regression was chosen as a simple and interpretable baseline model.


---

## Setup

Create and activate a virtual environment.

### Windows

```
python -m venv venv
venv\Scripts\activate
```


### macOS / Linux
```
python -m venv venv
source venv/bin/activate
```

#### Install the required packages:
```
pip install -r requirements.txt
```
---

## How to Run

### Train baseline model
```
python -m src.train
```

### Train model with hyperparameter tuning
```
python -m src.tune
```

#### The trained model is saved under:
```
models/titanic_model_tuned.joblib
```

---


## Results

Example results from a tuned run:

  - Accuracy: ~0.83
  - ROC-AUC: ~0.87

Results may vary depending on the random seed and data split.

---

## Notes

This project was created for learning and portfolio purposes.

The focus is on code clarity, project structure, and understanding the end-to-end machine learning workflow rather than achieving the best possible model performance.

---

## Possible Improvements
- Try different classification models and compare their performance
- Add more feature engineering and analyze their impact on the model
- Perform more systematic hyperparameter tuning
- Improve model evaluation with cross-validation on the final estimator
# Spaceship Titanic: Passenger Transportation Prediction

**End-to-end binary classification project | Python · scikit-learn · CatBoost · SHAP**

## Project overview

This project uses the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic) to predict whether a passenger was transported to an alternate dimension. The target, `Transported`, is binary (`True`/`False`). The workflow covers exploratory data analysis (EDA), missing-value handling, categorical encoding, feature engineering, model comparison, cross-validation, interpretation, and generation of a competition-ready submission.

**Goal:** Develop a reproducible classification workflow, evaluate different modelling choices, and communicate what the selected model learns from passenger attributes. This is a learning and portfolio project, not a deployed prediction service.

## Data

|File|Records|Role|
|-|-:|-|
|`train.csv`|8,693|Labelled development data|
|`test.csv`|4,277|Unlabelled competition test data|
|`sample\_submission.csv`|4,277|Submission format template|

Data is provided through the Kaggle competition. The original CSV datasets are intentionally **not committed** to this repository; see [Getting started](#getting-started).

## Workflow

1. **Explore the data:** inspect features, target balance, missingness, spending, CryoSleep, HomePlanet, age, and cabin information.
2. **Preprocess:** compare median/mode imputation and one-hot encoding with an alternative strategy retaining missing-value indicators. CatBoost handles missing numeric values directly and accepts specified categorical predictors.
3. **Establish baselines:** evaluate Logistic Regression and Random Forest, then compare CatBoost, KNN, and an RBF Support Vector Classifier.
4. **Engineer features:** extract `CabinDeck`, `CabinNumber`, and `CabinSide`; test spending aggregates and passenger-group features.
5. **Validate and tune:** use a stratified 80/20 development split and five-fold cross-validation on the development-training subset. Compare a separate group-aware cross-validation strategy.
6. **Interpret:** inspect CatBoost `PredictionValuesChange` feature importance and SHAP contributions.
7. **Submit:** fit the selected CatBoost configuration on all labelled competition data and generate a Kaggle-compatible CSV.

## Results

The main model-comparison figures below are **mean five-fold development cross-validation accuracy**. Models use their respective documented preprocessing and tested configurations; this is not a statistically definitive ranking of all possible hyperparameter settings.

|Model / configuration|Mean CV accuracy|
|-|-:|
|**CatBoost (selected)**|**81.12%**|
|Regularized Random Forest|80.21%|
|Random Forest (original)|79.85%|
|RBF Support Vector Classifier|79.68%|
|Logistic Regression (missingness-aware preprocessing)|79.16%|
|Logistic Regression (original preprocessing)|78.99%|
|K-Nearest Neighbors (`k=5`)|77.32%|

**Selected CatBoost parameters:** `iterations=500`, `depth=6`, `learning\_rate=0.05`, `loss\_function="Logloss"`, `random\_seed=42`.

**Additional checks**

|Evaluation|Result|
|-|-:|
|Five-fold *group-aware* development CV accuracy|80.85%|
|Kaggle **public leaderboard** accuracy (baseline submission)|**80.383%**|

The Kaggle public leaderboard score is a competition result, **not** a private leaderboard score or an independent final test estimate. Group-aware and ordinary cross-validation use different partitions and answer slightly different generalization questions.

### What the experiments showed

* Adding spending-based features produced **81.00%** mean CV accuracy, slightly below the 13-feature CatBoost baseline.
* Adding passenger-group-size and passenger-number features produced **80.33%** mean CV accuracy.
* The best tested depth, learning rate, and iteration combination remained the original CatBoost configuration.
* Preserving numerical missingness indicators and explicit categorical `Missing` values increased Logistic Regression's mean CV accuracy from **78.99% to 79.16%**. Both preprocessing changes were introduced together, so this does not isolate their individual effects.
* In built-in CatBoost feature importance, `HomePlanet` ranked highest. In mean absolute SHAP importance computed on a sample of 300 development-training passengers, `Spa` ranked highest, followed by `CryoSleep`.
* SHAP contributions for `CryoSleep=True` were positive on average in that sample and negative for `CryoSleep=False`. These are **model explanations, not causal effects**.

## Visualizations

### CatBoost feature importance

!\[CatBoost feature importance](figures/catboost\_feature\_importance.png)

### Mean absolute SHAP importance

!\[Mean absolute SHAP importance](figures/shap\_feature\_importance.png)

### Spa spending and its SHAP contribution

!\[Spa SHAP analysis](figures/spa\_shap\_analysis.png)

The figures are exported from the model interpretation notebook. They should be committed under `figures/` so GitHub can display them here.

## Repository structure

```text
spaceship-titanic/
├── notebooks/
│   ├── 01\_data\_understanding.ipynb
│   ├── 02\_model\_development.ipynb
│   ├── 03\_kaggle\_submission.ipynb
│   ├── 04\_feature\_engineering.ipynb
│   └── 05\_advanced\_experiments.ipynb
├── figures/
│   ├── catboost\_feature\_importance.png
│   ├── shap\_feature\_importance.png
│   └── spa\_shap\_analysis.png
├── submissions/
│   └── catboost\_baseline\_submission.csv
├── README.md
├── requirements.txt
└── .gitignore
```

The data CSV files live locally in the project root but are excluded from Git. Older notebooks should remain in `original\_notebooks\_backup/`, which is also excluded from Git.

## Getting started

1. Download the competition data from [Spaceship Titanic on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) under its applicable terms.
2. Place `train.csv`, `test.csv`, and `sample\_submission.csv` in the **project root**, alongside `README.md`, not in `notebooks/`.
3. In a terminal opened at the project root, create and activate a virtual environment and install the requirements:

   **Windows PowerShell**

```powershell
   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   python -m pip install -r requirements.txt
   jupyter notebook
   ```

   **macOS / Linux**

```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   jupyter notebook
   ```

4. Open the notebooks in the listed order. The revised notebooks locate CSV files from the current project directory or its parent; run Jupyter from the project root.
5. Notebook 3 regenerates `submissions/catboost\_baseline\_submission.csv`. Notebook 5 contains the model interpretation and exported charts.

Package versions may influence results. For an exact environment snapshot, record the installed versions with `python -m pip freeze` after testing the environment.

## Evaluation limitations

* Several experiments used the same five development folds, so the best observed CV score may reflect **model-selection optimism**.
* Earlier analysis explored the labelled dataset and development validation outcomes; do not describe the holdout as entirely untouched.
* Related passengers may share characteristics. Group-aware CV offers a complementary evaluation but differs from the competition's hidden test composition.
* Default and lightly tuned configurations are compared; a lower-ranked algorithm might improve with targeted tuning.
* Feature importance and SHAP measure contributions in a trained model; neither establishes causality.
* The public leaderboard reports performance only on its public portion; final ranking can differ on the private leaderboard.

## Skills demonstrated

Python, pandas, NumPy, Jupyter, exploratory data analysis, preprocessing pipelines, imputation, one-hot encoding, scaling, feature engineering, Logistic Regression, Random Forest, KNN, SVC, gradient boosting with CatBoost, cross-validation, group-aware evaluation, hyperparameter experiments, and model interpretation.

## Data attribution

Competition data and problem statement: [Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). Review the competition's rules before redistributing data or derived competition assets.

\## Kaggle Competition and Published Notebook



This project was developed for the Kaggle Spaceship Titanic competition.



The selected CatBoost model achieved:



\- \*\*Five-fold cross-validation accuracy:\*\* 81.12%

\- \*\*Group-aware cross-validation accuracy:\*\* 80.85%

\- \*\*Official Kaggle leaderboard accuracy:\*\* 80.383%

\- \*\*Leaderboard position:\*\* 788 (November 3, 2026)



\### Kaggle Resources



\- \[Published Kaggle Notebook](https://www.kaggle.com/code/basiruharuna/spaceship-titanic-catboost-baseline-80-38-acc)

\- \[Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic)



The published Kaggle notebook reproduces the final CatBoost

training and prediction workflow using the official competition data.


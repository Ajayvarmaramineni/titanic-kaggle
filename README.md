# Titanic - Machine Learning from Disaster

**Kaggle Competition:** https://www.kaggle.com/competitions/titanic

**Best Score:** 0.78708 (Top 35% on public leaderboard)

## Project Overview

This project applies machine learning techniques to predict passenger survival on the Titanic using the famous Kaggle dataset. The goal is to classify passengers as survivors or non-survivors based on features like age, class, gender, and family relationships.

## Approach

### 1. Data Preprocessing & Feature Engineering
- **Title Extraction:** Extracted and grouped passenger titles (Mr, Mrs, Miss, Master, Rare)
- **Family Features:** Created FamilySize, IsAlone, and FamilyId features to capture family relationships
- **Ticket Analysis:** Grouped passengers by ticket prefix to identify travel groups
- **Cabin & Deck:** Extracted deck information from cabin numbers
- **Fare Engineering:** Applied log transformation and created fare bins
- **Age Imputation:** Used RandomForestRegressor for model-based age imputation (by Pclass & Sex)
- **Missingness Flags:** Added indicators for imputed fields

### 2. Models Used
- **Random Forest:** 200 trees, max_depth=12, 83.73% OOF accuracy
- **Gradient Boosting:** 150 estimators, max_depth=5, 82.15% OOF accuracy
- **XGBoost:** 150 estimators, max_depth=6, 82.49% OOF accuracy

### 3. Validation Strategy
- Stratified 5-Fold Cross-Validation to prevent data leakage
- Out-of-Fold (OOF) predictions to track generalization
- Threshold optimization on validation set

### 4. Ensemble Approach
Tested multiple ensemble strategies:
- **Stacking with Meta-Learner:** Used Logistic Regression on base model OOF predictions (OOF: 0.8361)
- **Simple Weighted Ensemble:** Weighted average of base models (RF: 35%, GB: 25%, XGB: 40%) - **BEST GENERALIZER**

**Key Insight:** Simpler weighted ensemble generalized better to the public leaderboard (0.78708) than complex stacking (0.78229), demonstrating the importance of validation strategy in preventing overfitting.

## Results

| Model | OOF Score | Public LB Score |
|-------|-----------|-----------------|
| Random Forest | 0.8373 | N/A |
| Gradient Boosting | 0.8215 | N/A |
| XGBoost | 0.8249 | N/A |
| Stacking Ensemble | 0.8361 | 0.78229 |
| Simple Weighted Ensemble | N/A | **0.78708** ✅ |

**Final Ranking:** Top 35% on Kaggle public leaderboard

## Key Learnings

1. **Feature Engineering Matters:** Advanced features (title grouping, family analysis, model-based imputation) provided significant lift
2. **OOF vs Public Score Gap:** Large gaps between OOF and public scores indicate overfitting to CV folds - simpler models often generalize better
3. **Ensemble Trade-offs:** While stacking improved OOF accuracy to 83.61%, the simple weighted average (78.71%) better represented actual test performance
4. **Threshold Tuning:** Optimizing decision threshold for accuracy metric yielded 0.5-1% improvements
5. **Stratified Validation:** Proper stratified cross-validation is critical to avoid data leakage

## Files

- `titanic_solution.py` - Complete pipeline with all preprocessing, feature engineering, modeling, and ensemble code
- `submission_best.csv` - Best submission (score: 0.78708)
- `README.md` - This file

## Technologies Used

- **Python 3.7+**
- **Libraries:** pandas, numpy, scikit-learn, xgboost
- **Techniques:** Feature engineering, cross-validation, ensemble learning, threshold optimization

## How to Use

1. Download `train.csv` and `test.csv` from Kaggle
2. Run `titanic_solution.py`
3. Submit `submission_best.csv` to Kaggle

## Future Improvements

- Hyperparameter tuning with Optuna
- Feature interaction engineering (Pclass × Sex, Title × AgeBin)
- Neural network approach (Deep Learning)
- More sophisticated imputation (KNN, MICE)
- Ensemble with additional base models (CatBoost, LightGBM)

## Competition Details

- **Dataset:** 891 training samples, 418 test samples
- **Features:** 11 (Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked, Name, Ticket)
- **Target:** Binary classification (Survived: 0/1)
- **Metric:** Accuracy

---

**Author:** Ajay Ramineni
**Date:** March 2026
**Status:** Completed - Top 35% Kaggle submission

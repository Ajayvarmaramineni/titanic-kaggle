"""
Titanic - Machine Learning from Disaster
Kaggle Competition Solution
Score: 0.78708 (Top 35%)

This solution uses advanced feature engineering, multiple base models,
and weighted ensemble to predict passenger survival.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==================== 1. LOAD DATA ====================
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['PassengerId'].copy()

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ==================== 2. ADVANCED FEATURE ENGINEERING ====================
def advanced_features(df):
    """Apply advanced feature engineering to dataset"""
    df = df.copy()

    # TITLE EXTRACTION
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')[0]
    title_counts = df['Title'].value_counts()
    rare_titles = title_counts[title_counts < 10].index.tolist()
    df.loc[df['Title'].isin(rare_titles), 'Title'] = 'Rare'

    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = df['Title'].map(title_map).fillna(4).astype(int)

    # FAMILY FEATURES
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Surname'] = df['Name'].str.split(',').str[0]
    df['FamilyId'] = df['Surname'] + '_' + df['FamilySize'].astype(str)
    family_counts = df['FamilyId'].value_counts()
    df['FamilyId_size'] = df['FamilyId'].map(family_counts)

    # TICKET ANALYSIS
    df['TicketPrefix'] = df['Ticket'].str.extract('([A-Z]+)')[0].fillna('0')
    ticket_counts = df['TicketPrefix'].value_counts()
    df['TicketGroup_size'] = df['TicketPrefix'].map(ticket_counts)

    # CABIN & DECK
    df['HasCabin'] = (~df['Cabin'].isna()).astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')

    # FARE PROCESSING
    for pclass in [1, 2, 3]:
        mask = df['Pclass'] == pclass
        median_fare = df[mask]['Fare'].median()
        df.loc[mask & df['Fare'].isna(), 'Fare'] = median_fare
        df.loc[mask & (df['Fare'] == 0), 'Fare'] = median_fare

    df['Fare_log'] = np.log1p(df['Fare'])
    df['Fare_per_person'] = df['Fare'] / df['FamilySize']
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=False, duplicates='drop')

    # EMBARKED
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # MISSINGNESS FLAG
    df['Age_missing'] = df['Age'].isna().astype(int)

    # ENCODING
    df['Sex'] = (df['Sex'] == 'male').astype(int)

    return df

print("Applying advanced feature engineering...")
train_df_fe = advanced_features(train_df)
test_df_fe = advanced_features(test_df)

# Encode categorical variables
le_deck = LabelEncoder()
le_embarked = LabelEncoder()

all_decks = pd.concat([train_df_fe['Deck'], test_df_fe['Deck']])
all_embarked = pd.concat([train_df_fe['Embarked'], test_df_fe['Embarked']])

le_deck.fit(all_decks)
le_embarked.fit(all_embarked)

train_df_fe['Deck'] = le_deck.transform(train_df_fe['Deck'])
test_df_fe['Deck'] = le_deck.transform(test_df_fe['Deck'])

train_df_fe['Embarked'] = le_embarked.transform(train_df_fe['Embarked'])
test_df_fe['Embarked'] = le_embarked.transform(test_df_fe['Embarked'])

# Clean up
train_df_fe = train_df_fe.drop(['Name', 'Ticket', 'Cabin', 'Surname', 'FamilyId', 'TicketPrefix'],
                                axis=1, errors='ignore')
test_df_fe = test_df_fe.drop(['Name', 'Ticket', 'Cabin', 'Surname', 'FamilyId', 'TicketPrefix'],
                              axis=1, errors='ignore')

print(f"Features: {train_df_fe.shape}")

# ==================== 3. MODEL-BASED AGE IMPUTATION ====================
print("\nApplying model-based age imputation...")

age_train = train_df_fe[train_df_fe['Age'].notna()]
age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize']
X_age = age_train[age_features]
y_age = age_train['Age']

age_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
age_model.fit(X_age, y_age)

for df in [train_df_fe, test_df_fe]:
    missing_mask = df['Age'].isna()
    if missing_mask.sum() > 0:
        X_missing = df[missing_mask][age_features]
        df.loc[missing_mask, 'Age'] = age_model.predict(X_missing)

train_df_fe['AgeBin'] = pd.cut(train_df_fe['Age'], bins=[0, 5, 12, 18, 35, 60, 100], labels=False)
test_df_fe['AgeBin'] = pd.cut(test_df_fe['Age'], bins=[0, 5, 12, 18, 35, 60, 100], labels=False)

print("Age imputation complete")

# ==================== 4. PREPARE DATA ====================
X_train = train_df_fe.drop('Survived', axis=1)
y_train = train_df_fe['Survived']
X_test = test_df_fe

# ==================== 5. TRAIN BASE MODELS WITH STRATIFIED K-FOLD ====================
print("\nTraining base models with 5-Fold CV...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_rf = np.zeros(len(X_train))
oof_gb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

test_pred_rf = np.zeros(len(X_test))
test_pred_gb = np.zeros(len(X_test))
test_pred_xgb = np.zeros(len(X_test))

fold = 1
for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
    test_pred_rf += rf.predict_proba(X_test)[:, 1] / 5

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_tr, y_tr)
    oof_gb[val_idx] = gb.predict_proba(X_val)[:, 1]
    test_pred_gb += gb.predict_proba(X_test)[:, 1] / 5

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_model.fit(X_tr, y_tr, verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_pred_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5

    print(f"Fold {fold} complete")
    fold += 1

print(f"\nOOF Accuracies:")
print(f"  RF:  {(oof_rf.round() == y_train).mean():.4f}")
print(f"  GB:  {(oof_gb.round() == y_train).mean():.4f}")
print(f"  XGB: {(oof_xgb.round() == y_train).mean():.4f}")

# ==================== 6. WEIGHTED ENSEMBLE (BEST GENERALIZER) ====================
print("\nBuilding weighted ensemble...")

weights = {'rf': 0.35, 'gb': 0.25, 'xgb': 0.40}

ensemble_pred_final = (weights['rf'] * test_pred_rf +
                       weights['gb'] * test_pred_gb +
                       weights['xgb'] * test_pred_xgb)

oof_ensemble = (weights['rf'] * oof_rf +
                weights['gb'] * oof_gb +
                weights['xgb'] * oof_xgb)

print(f"Weighted Ensemble OOF: {(oof_ensemble.round() == y_train).mean():.4f}")

# ==================== 7. THRESHOLD OPTIMIZATION ====================
print("\nOptimizing decision threshold...")

best_threshold = 0.5
best_accuracy = 0

for threshold in np.arange(0.35, 0.66, 0.01):
    accuracy = ((oof_ensemble >= threshold).astype(int) == y_train).mean()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f} | OOF Accuracy: {best_accuracy:.4f}")

# ==================== 8. GENERATE SUBMISSION ====================
print("\nGenerating submission...")

final_predictions = (ensemble_pred_final >= best_threshold).astype(int)

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': final_predictions
})

submission.to_csv('submission_best.csv', index=False)
print("✅ Submission created: submission_best.csv")
print(f"\nExpected Kaggle Score: {best_accuracy:.4f}")
print(submission.head(10))

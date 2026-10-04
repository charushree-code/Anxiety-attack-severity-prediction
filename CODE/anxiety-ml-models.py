import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Generate sample data based on the given structure
# In a real scenario, you would load the actual dataset
np.random.seed(42)
n_samples = 1000

# Create the dataset with relevant features
df = pd.DataFrame({
    'Age': np.random.randint(18, 75, n_samples),
    'Gender': np.random.randint(0, 2, n_samples),
    'Panic_Attack_Frequency': np.random.randint(1, 15, n_samples),
    'Duration_Minutes': np.random.randint(1, 60, n_samples),
    'Trigger': np.random.randint(0, 5, n_samples),
    'Heart_Rate': np.random.randint(70, 160, n_samples),
    'Sweating': np.random.randint(0, 2, n_samples),
    'Shortness_of_Breath': np.random.randint(0, 2, n_samples),
    'Dizziness': np.random.randint(0, 2, n_samples),
    'Chest_Pain': np.random.randint(0, 2, n_samples),
    'Trembling': np.random.randint(0, 2, n_samples),
    'Medical_History': np.random.randint(0, 5, n_samples),
    'Medication': np.random.randint(0, 2, n_samples),
    'Caffeine_Intake': np.random.randint(0, 5, n_samples),
    'Exercise_Frequency': np.random.randint(0, 7, n_samples),
    'Sleep_Hours': np.random.uniform(4, 10, n_samples),
    'Alcohol_Consumption': np.random.randint(0, 10, n_samples),
    'Smoking': np.random.randint(0, 2, n_samples),
    'Therapy': np.random.randint(0, 2, n_samples)
})

# Feature Engineering (as done previously)
df['Symptom_Score'] = df['Sweating'] + df['Shortness_of_Breath'] + df['Dizziness'] + df['Chest_Pain'] + df['Trembling']
df['Lifestyle_Score'] = (
    df['Exercise_Frequency'] * 0.3 +
    (df['Sleep_Hours'] - 6) * 0.3 +
    (4 - df['Caffeine_Intake']) * 0.15 +
    (10 - df['Alcohol_Consumption']) * 0.15 +
    (1 - df['Smoking']) * 0.1
)
df['Stress_Vulnerability'] = df['Panic_Attack_Frequency'] * 0.4 + (5 - df['Lifestyle_Score']) * 0.4 + df['Trigger'] * 0.2
df['Has_Treatment'] = np.where((df['Medication'] == 1) | (df['Therapy'] == 1), 1, 0)
df['Combined_Treatment'] = np.where((df['Medication'] == 1) & (df['Therapy'] == 1), 1, 0)

# Create Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 75], labels=[0, 1, 2, 3])

# Create a target variable for classification: High Frequency Attacks (≥10 per month)
df['High_Frequency'] = np.where(df['Panic_Attack_Frequency'] >= 10, 1, 0)

# For regression, we'll predict the Panic_Attack_Frequency
# For classification, we'll predict the High_Frequency

# Define features
categorical_features = ['Gender', 'Trigger', 'Medical_History', 'Age_Group']
numerical_features = [
    'Age', 'Heart_Rate', 'Sweating', 'Shortness_of_Breath', 
    'Dizziness', 'Chest_Pain', 'Trembling', 'Medication', 
    'Caffeine_Intake', 'Exercise_Frequency', 'Sleep_Hours', 
    'Alcohol_Consumption', 'Smoking', 'Therapy', 
    'Symptom_Score', 'Lifestyle_Score', 'Stress_Vulnerability',
    'Has_Treatment', 'Combined_Treatment'
]

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model evaluation function
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance (handle different model types)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(X_train.columns)  # Placeholder for models without feature_importances_
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Importance': importances
    }

def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance (handle different model types)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(X_train.columns)  # Placeholder
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': report['1']['precision'],  # For the positive class
        'Recall': report['1']['recall'],
        'F1': report['1']['f1-score'],
        'Importance': importances
    }

# Split the data for regression
X_reg = df[numerical_features + categorical_features]
y_reg = df['Panic_Attack_Frequency']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

# Split the data for classification
X_class = df[numerical_features + categorical_features]
y_class = df['High_Frequency']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.25, random_state=42)

# Define models for regression
rf_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

xgb_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

cat_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', CatBoostRegressor(n_estimators=100, learning_rate=0.1, verbose=0, random_state=42))
])

# Define models for classification
rf_class = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

xgb_class = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
])

cat_class = Pipeline([
    ('preprocessor', preprocessor),
    ('model', CatBoostClassifier(n_estimators=100, learning_rate=0.1, verbose=0, random_state=42))
])

# Evaluate regression models
reg_models = [
    (rf_reg, "Random Forest"),
    (xgb_reg, "XGBoost"),
    (cat_reg, "CatBoost")
]

reg_results = []
for model, name in reg_models:
    result = evaluate_regression_model(model, X_train_reg, X_test_reg, y_train_reg, y_test_reg, name)
    reg_results.append(result)

# Evaluate classification models
class_models = [
    (rf_class, "Random Forest"),
    (xgb_class, "XGBoost"),
    (cat_class, "CatBoost")
]

class_results = []
for model, name in class_models:
    result = evaluate_classification_model(model, X_train_class, X_test_class, y_train_class, y_test_class, name)
    class_results.append(result)

# Create comparison tables
reg_df = pd.DataFrame(reg_results)[['Model', 'MSE', 'RMSE', 'R²']]
class_df = pd.DataFrame(class_results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']]

print("Regression Model Comparison:")
print(reg_df)
print("\nClassification Model Comparison:")
print(class_df)

# Visualize regression results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='RMSE', data=reg_df)
plt.title('Regression Models: RMSE (lower is better)')
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R²', data=reg_df)
plt.title('Regression Models: R² (higher is better)')
plt.tight_layout()
plt.savefig('regression_comparison.png')
plt.close()

# Visualize classification results
plt.figure(figsize=(14, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
for i, metric in enumerate(metrics):
    plt.subplot(1, 4, i+1)
    sns.barplot(x='Model', y=metric, data=class_df)
    plt.title(f'Classification Models: {metric}')
    plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('classification_comparison.png')
plt.close()

# Cross-validation for regression models
cv_scores = {}
for model, name in reg_models:
    scores = cross_val_score(model, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error')
    cv_scores[name] = -scores.mean()

# Cross-validation for classification models
cv_scores_class = {}
for model, name in class_models:
    scores = cross_val_score(model, X_class, y_class, cv=5, scoring='accuracy')
    cv_scores_class[name] = scores.mean()

# Visualize cross-validation results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(cv_scores.keys(), cv_scores.values())
plt.title('5-Fold CV: MSE for Regression Models (lower is better)')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
plt.bar(cv_scores_class.keys(), cv_scores_class.values())
plt.title('5-Fold CV: Accuracy for Classification Models (higher is better)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('cross_validation_results.png')
plt.close()

# Find the best model for both tasks
best_reg_model = min(cv_scores.items(), key=lambda x: x[1])[0]
best_class_model = max(cv_scores_class.items(), key=lambda x: x[1])[0]

print(f"\nBest regression model based on cross-validation: {best_reg_model}")
print(f"Best classification model based on cross-validation: {best_class_model}")

# Hyperparameter tuning for the best models
if best_reg_model == "Random Forest":
    param_grid_reg = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }
    best_reg_pipeline = rf_reg
elif best_reg_model == "XGBoost":
    param_grid_reg = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
    best_reg_pipeline = xgb_reg
else:  # CatBoost
    param_grid_reg = {
        'model__iterations': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__depth': [4, 6, 8]
    }
    best_reg_pipeline = cat_reg

if best_class_model == "Random Forest":
    param_grid_class = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }
    best_class_pipeline = rf_class
elif best_class_model == "XGBoost":
    param_grid_class = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
    best_class_pipeline = xgb_class
else:  # CatBoost
    param_grid_class = {
        'model__iterations': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__depth': [4, 6, 8]
    }
    best_class_pipeline = cat_class

# Perform grid search for regression
grid_search_reg = GridSearchCV(best_reg_pipeline, param_grid_reg, cv=3, scoring='neg_mean_squared_error')
grid_search_reg.fit(X_reg, y_reg)

# Perform grid search for classification
grid_search_class = GridSearchCV(best_class_pipeline, param_grid_class, cv=3, scoring='accuracy')
grid_search_class.fit(X_class, y_class)

print(f"\nBest parameters for {best_reg_model} (regression):")
print(grid_search_reg.best_params_)
print(f"Best cross-validation score: {-grid_search_reg.best_score_:.4f} MSE")

print(f"\nBest parameters for {best_class_model} (classification):")
print(grid_search_class.best_params_)
print(f"Best cross-validation score: {grid_search_class.best_score_:.4f} accuracy")

# Feature importance for the best tuned models (using the best model from GridSearchCV)
best_reg_model_tuned = grid_search_reg.best_estimator_
best_class_model_tuned = grid_search_class.best_estimator_

# For regression: Get feature names after preprocessing
# This is complex due to preprocessing transformations, so we'll use a simpler approach
# Train a model directly without pipeline to get feature importance
if best_reg_model == "Random Forest":
    direct_model = RandomForestRegressor(**{k.replace('model__', ''): v for k, v in grid_search_reg.best_params_.items()})
elif best_reg_model == "XGBoost":
    direct_model = XGBRegressor(**{k.replace('model__', ''): v for k, v in grid_search_reg.best_params_.items()})
else:  # CatBoost
    direct_model = CatBoostRegressor(**{k.replace('model__', ''): v for k, v in grid_search_reg.best_params_.items()}, verbose=0)

# One-hot encode categorical features for direct model
X_encoded = pd.get_dummies(X_reg, columns=categorical_features)
direct_model.fit(X_encoded, y_reg)

# Get feature importance
if hasattr(direct_model, 'feature_importances_'):
    feature_importance = direct_model.feature_importances_
    feature_names = X_encoded.columns
    
    # Create dataframe for feature importance
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(15)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
    plt.title(f'Top 15 Feature Importances for {best_reg_model} (Regression)')
    plt.tight_layout()
    plt.savefig('feature_importance_regression.png')
    plt.close()
    
    print("\nTop 10 Features for Predicting Panic Attack Frequency:")
    print(feat_imp_df.head(10))

# For classification: similar approach
if best_class_model == "Random Forest":
    direct_model_class = RandomForestClassifier(**{k.replace('model__', ''): v for k, v in grid_search_class.best_params_.items()})
elif best_class_model == "XGBoost":
    direct_model_class = XGBClassifier(**{k.replace('model__', ''): v for k, v in grid_search_class.best_params_.items()})
else:  # CatBoost
    direct_model_class = CatBoostClassifier(**{k.replace('model__', ''): v for k, v in grid_search_class.best_params_.items()}, verbose=0)

direct_model_class.fit(X_encoded, y_class)

# Get feature importance for classification
if hasattr(direct_model_class, 'feature_importances_'):
    feature_importance_class = direct_model_class.feature_importances_
    
    # Create dataframe for feature importance
    feat_imp_df_class = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance_class
    }).sort_values('Importance', ascending=False).head(15)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df_class)
    plt.title(f'Top 15 Feature Importances for {best_class_model} (Classification)')
    plt.tight_layout()
    plt.savefig('feature_importance_classification.png')
    plt.close()
    
    print("\nTop 10 Features for Predicting High Frequency Attacks:")
    print(feat_imp_df_class.head(10))

# Generate confusion matrix for the best classification model
best_class_model_tuned.fit(X_train_class, y_train_class)
y_pred_class = best_class_model_tuned.predict(X_test_class)
cm = confusion_matrix(y_test_class, y_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {best_class_model}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Summary document
summary = f"""
# Machine Learning Model Comparison for Anxiety Attack Prediction

## Regression Task: Predicting Panic Attack Frequency

| Model | MSE | RMSE | R² |
|-------|-----|------|---|
{' | '.join(reg_df.iloc[0].values.astype(str))}
{' | '.join(reg_df.iloc[1].values.astype(str))}
{' | '.join(reg_df.iloc[2].values.astype(str))}

Best model: {best_reg_model} with CV MSE of {min(cv_scores.values()):.4f}

## Classification Task: Predicting High Frequency Attacks

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|---|
{' | '.join(class_df.iloc[0].values.astype(str))}
{' | '.join(class_df.iloc[1].values.astype(str))}
{' | '.join(class_df.iloc[2].values.astype(str))}

Best model: {best_class_model} with CV accuracy of {max(cv_scores_class.values()):.4f}

## Key Findings

1. {best_reg_model} performed best for predicting the exact frequency of panic attacks
2. {best_class_model} performed best for classifying high vs. low frequency attacks
3. The most important features for prediction were:
   - For regression: {', '.join(feat_imp_df['Feature'].head(5).values)}
   - For classification: {', '.join(feat_imp_df_class['Feature'].head(5).values)}
4. Hyperparameter tuning improved performance by {((grid_search_reg.best_score_ - (-min(cv_scores.values())))/(-min(cv_scores.values())))*100:.1f}% for regression and {((grid_search_class.best_score_ - max(cv_scores_class.values()))/max(cv_scores_class.values()))*100:.1f}% for classification

## Recommendations

1. Use {best_reg_model} for predicting exact attack frequency with the following parameters:
   {grid_search_reg.best_params_}

2. Use {best_class_model} for risk classification with the following parameters:
   {grid_search_class.best_params_}

3. Focus on the top features identified for monitoring and intervention
"""

print("\nSummary of Model Comparison:")
print(summary)

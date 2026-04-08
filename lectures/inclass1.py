# InClass 1 - Insurance Dataset
# 1. (3 pts) Load dataset, display first 5 rows
# 2. (3 pts) SweetViz EDA, describe trends
# 3. (4 pts) Define X and y, split into train/test (80/20)
# 4. (4 pts) Scale input variables
# 5. (5 pts) Baseline LR and RF, report R2 and MSE
# 6. (5 pts) GridSearchCV to tune RF hyperparameters
# 7. (5 pts) Look at top 20 models, discuss which hyperparams to pick
# 8. (3 pts) Compare train vs test performance, check for overfitting

import pandas as pd
import numpy as np
import sweetviz as sv
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1 - load and display
df = pd.read_csv('/Users/sahilbains/Downloads/AdvML/data/insurance.csv')
print(df.head(5))

# 2 - sweetviz eda
# report = sv.analyze(df)
# report.show_html('insurance_report.html')

# takeaways from the report:
# - charges is right skewed, most people have lower costs but some have really high ones
# - smoker status has the biggest impact on charges by far
# - age and bmi are positively correlated with charges, bmi especially matters more for smokers

# 3 - encoding categorical vars, defining X and y, splitting
insurance_encoded = pd.get_dummies(df, drop_first=True).astype(int)

X = insurance_encoded.drop('charges', axis=1)
y = insurance_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4 - scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5 - baseline models
baseline_lr = LinearRegression()
baseline_lr.fit(X_train_scaled, y_train)
y_pred_lr = baseline_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

baseline_rf = RandomForestRegressor(random_state=42)
baseline_rf.fit(X_train, y_train)
y_pred_rf = baseline_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Baseline Linear Regression MSE:', mse_lr)
print('Baseline Linear Regression R2:', r2_lr)
print('Baseline Random Forest MSE:', mse_rf)
print('Baseline Random Forest R2:', r2_rf)

# the linear regression got an R2 of about 0.784 and MSE around 33.5M which isnt great
# random forest did a lot better with R2 of 0.857 and MSE around 22.1M
# makes sense since RF can pick up on nonlinear stuff that LR cant, like the interaction between smoking and bmi

# 6 - gridsearch cv
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf_gscv = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
rf_gscv.fit(X_train, y_train)

print('Best hyperparams:', rf_gscv.best_params_)
print('Best score:', rf_gscv.best_score_)

# 7 - top 20 models
# looking at the top 20 models they all score pretty similarly around 0.84 on the cv folds
# the main differences are in max_depth (10 vs 5 vs None) and min_samples_split
# i would pick something with max_depth=10 and a higher min_samples_split like 5
# because using None for max_depth lets trees grow really deep which can overfit
# constraining it a bit should generalize better on new data

# 8 - overfitting check
best_rf = rf_gscv.best_estimator_

y_pred_train = best_rf.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)

y_pred_test = best_rf.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training R2: {r2_train:.4f}")
print(f"Testing R2: {r2_test:.4f}")

# training R2 is about 0.93 and testing is about 0.87 so theres definitely some overfitting
# the model fits the training data really well but drops off on the test set
# this is pretty normal for random forests though, you could try increasing min_samples_split
# or lowering max_depth more to reduce the gap

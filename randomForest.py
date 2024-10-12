# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# Load the dataset
url = "/Users/amoghsood/Desktop/UTA2024/crop_yield.csv"
df = pd.read_csv(url)

# Data preprocessing (encoding categorical variables)
label_encoder = LabelEncoder()
df['Region'] = label_encoder.fit_transform(df['Region'])
df['Soil_Type'] = label_encoder.fit_transform(df['Soil_Type'])
df['Crop'] = label_encoder.fit_transform(df['Crop'])
df['Weather_Condition'] = label_encoder.fit_transform(df['Weather_Condition'])

# Selecting features and target variable
X = df[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
        'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest']]
y = df['Yield_tons_per_hectare']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Make predictions and evaluate
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
print(f"Best Model Mean Squared Error: {mse_best:.2f}")
print(f"Best Model R-squared Score: {r2_best:.2f}")

# Feature importance plot
feature_importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.show()

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mean_mse = -cv_scores.mean()
cv_std_mse = cv_scores.std()
print(f"Cross-Validated Mean Squared Error: {cv_mean_mse:.2f}")
print(f"Cross-Validated Standard Deviation of MSE: {cv_std_mse:.2f}")

# Save and load the model
joblib.dump(best_model, 'best_random_forest_model.pkl')
loaded_model = joblib.load('best_random_forest_model.pkl')
y_pred_loaded = loaded_model.predict(X_test)
mse_loaded = mean_squared_error(y_test, y_pred_loaded)
r2_loaded = r2_score(y_test, y_pred_loaded)
print(f"Loaded Model Mean Squared Error: {mse_loaded:.2f}")
print(f"Loaded Model R-squared Score: {r2_loaded:.2f}")


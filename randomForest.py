# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "your_path_to_agriculture_crop_yield.csv"  # Update with actual path
df = pd.read_csv(url)

# Preview the dataset
print(df.head())

# Data preprocessing (encoding categorical variables)
label_encoder = LabelEncoder()

# Encoding categorical columns
df['Region'] = label_encoder.fit_transform(df['Region'])
df['Soil_Type'] = label_encoder.fit_transform(df['Soil_Type'])
df['Crop'] = label_encoder.fit_transform(df['Crop'])
df['Weather_Condition'] = label_encoder.fit_transform(df['Weather_Condition'])

# Selecting features and target variable
X = df[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius', 
        'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest']]
y = df['Yield']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")


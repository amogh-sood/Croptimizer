import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle


# Load the dataset
df = pd.read_csv('/Users/amoghsood/Desktop/UTA2024/backend/crop_yield.csv')

# Separate features and target variable
X = df.drop(columns='Yield_tons_per_hectare')
y = df['Yield_tons_per_hectare']

# Label encoding for categorical variables
label_encoders = {}
for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
scaler = StandardScaler()
# Scaling numerical features
X[['Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Days_to_Harvest']] = scaler.fit_transform(
    X[['Rainfall_mm', 'Temperature_Celsius', 'Fertilizer_Used', 'Days_to_Harvest']]
)

def runModel():
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the neural network model
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

    # Save the trained model
    model.save('agriculture_yield_prediction_model_V2.h5')

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)

    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    print(label_encoders)

    with open("label_encoders.pkl", "wb") as f:  # "wb" means write binary
        pickle.dump(label_encoders, f)
        
if __name__ == '__main__':
    runModel()
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from neuralNetwork import scaler
import pickle

with open("label_encoders.pkl", "rb") as f:  # "rb" means read binary
    label_encoders = pickle.load(f)
    
print(label_encoders)
print("*"*25)

# Load the model with the 'mse' loss function
model = keras.models.load_model('agriculture_yield_prediction_model_V2.h5', custom_objects={'mse': MeanSquaredError()})


model.summary()

# Example: A single new input with Region, Soil_Type, etc.
new_data = {
    'Region': 'West',
    'Soil_Type': 'Peaty',
    'Crop': 'Maize',
    'Rainfall_mm': 189.5,
    'Temperature_Celsius': 15.5,
    'Fertilizer_Used': False,
    'Irrigation_Used': False,
    'Weather_Condition': 'Sunny',
    'Days_to_Harvest': 147
}

# Example preprocessing: assuming you used LabelEncoder and StandardScaler
# Apply label encoding and scaling (using the same encoder/scaler from training)

encoded_region = label_encoders['Region'].transform([new_data['Region']])[0]
encoded_soil_type = label_encoders['Soil_Type'].transform([new_data['Soil_Type']])[0]
encoded_crop = label_encoders['Crop'].transform([new_data['Crop']])[0]
encoded_weather = label_encoders['Weather_Condition'].transform([new_data['Weather_Condition']])[0]

scaled_rainfall = scaler.transform([[new_data['Rainfall_mm'], new_data['Temperature_Celsius'], new_data['Fertilizer_Used'], new_data['Days_to_Harvest']]])

# Now combine the preprocessed features into an array
input_data = np.array([[encoded_region, encoded_soil_type, encoded_crop, 
                        scaled_rainfall[0][0], scaled_rainfall[0][1], 
                        scaled_rainfall[0][2], new_data['Irrigation_Used'], 
                        encoded_weather, scaled_rainfall[0][3]]])


prediction = model.predict(input_data)

# Output the prediction
print(f"Predicted Yield (tons per hectare): {prediction[0][0]}")
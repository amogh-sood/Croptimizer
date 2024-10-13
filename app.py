from flask import Flask, render_template, request
import numpy as np 
import pickle 
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
from neuralNetwork import scaler

app = Flask(__name__)

model = keras.models.load_model('agriculture_yield_prediction_model_V2.h5', custom_objects={'mse': MeanSquaredError()})

with open("label_encoders.pkl", "rb") as f:  # "rb" means read binary
    label_encoders = pickle.load(f)
    
    def index():
        return render_template('index.html')
    
    def predict():
        
        region = request.form['Region']
        soil_type = request.form['Soil_Type']
        crop = request.form['Crop']
        rainfall = float(request.form['Rainfall_mm'])
        temperature = float(request.form['Temperature_Celsius'])
        fertilizer = request.form['Fertilizer_Used']
        irrigation = request.form['Irrigation_Used']
        weather = request.form['Weather_Condition']
        days_to_harvest = int(request.form['Days_to_Harvest'])
        
        encoded_region = label_encoders['Region'].transform([region])[0]
        encoded_soil_type = label_encoders['Soil_Type'].transform([soil_type])[0]
        encoded_crop = label_encoders['Crop'].transform([crop])[0]
        encoded_weather = label_encoders['Weather_Condition'].transform([weather])[0]
        
        scaled_data = scaler.transform([[rainfall, temperature, fertilizer, days_to_harvest]])
        
        input_data = np.array([[encoded_region, encoded_soil_type, encoded_crop, 
                                scaled_data[0][0], scaled_data[0][1], 
                                scaled_data[0][2], irrigation, 
                                encoded_weather, scaled_data[0][3]]])
        
        prediction = model.predict(input_data)
        prediction_yeild = prediction[0][0]
        
        return render_template('index.html', prediction_text = f"Predicted Yeild (tons per hectare): {prediction_yeild}")
    
    if __name__ == "__main__":
        app.run(debug=True)
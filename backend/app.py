from flask import Flask, render_template, request, jsonify
import numpy as np 
import pickle 
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
from neuralNetwork import scaler
from flask_cors import CORS

app = Flask(__name__, template_folder='../frontend')

CORS(app)

model = keras.models.load_model('agriculture_yield_prediction_model_V2.h5', custom_objects={'mse': MeanSquaredError()})

with open("./label_encoders.pkl", "rb") as f:  # "rb" means read binary
    label_encoders = pickle.load(f)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/submit", methods = ['POST'])
def do():
    data = request.json
    print(data)
    region = data['Region']
    soil_type = data['Soil_Type']
    crop = data['Crop']
    rainfall = float(data['Rainfall_mm'])
    temperature = float(data['Temperature_Celsius'])
    fertilizer = data['Fertilizer_Used']
    irrigation = data['Irrigation_Used']
    weather = data['Weather_Condition']
    days_to_harvest = int(data['Days_to_Harvest'])
    fertilizer = True if fertilizer == 'True' else False
    irrigation = True if irrigation == 'True' else False
    
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
    prediction_yield = prediction[0][0]
    
    return jsonify({"prediction": f"{prediction_yield:.2f}"})

if __name__ == "__main__":
    app.run(debug=True)
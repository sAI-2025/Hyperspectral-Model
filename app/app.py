from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load Pretrained Models
encoder = load_model("encoder_model.h5")  # Load the trained encoder
cnn_model = load_model("cnn_model.h5", compile=False)  # Load the trained CNN model
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Load the Target Variable Scaler
scaler_y = joblib.load("scaler_y.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form (Hyperspectral data)
        input_features = [float(x) for x in request.form.values()]
        input_features = np.array(input_features).reshape(1, -1)  # Reshape for a single prediction
        
        # Encode Features using Pretrained Encoder
        encoded_features = encoder.predict(input_features)
        encoded_features = np.expand_dims(encoded_features, axis=-1)  # Reshape for CNN
        
        # Predict Using Pretrained CNN Model
        y_pred = cnn_model.predict(encoded_features)
        
        # Inverse transform to original scale
        y_pred = scaler_y.inverse_transform(y_pred)  # Convert to Original Scale
        
        return render_template('index.html', prediction_text=f"Predicted Value: {y_pred[0][0]:.4f}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

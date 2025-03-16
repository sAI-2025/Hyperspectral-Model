# Hyperspectral-Model
## **1. Introduction**
This project focuses on predicting **vomitoxin_ppb** levels from hyperspectral data using advanced **machine learning** and **deep learning** models. Hyperspectral imaging produces high-dimensional data with **449 spectral features**, requiring effective **feature selection, dimensionality reduction, and model experimentation** to improve prediction accuracy.

By addressing challenges such as the **curse of dimensionality and spatial feature dependencies**, we explored various approaches including **Linear Regression, Random Forest, XGBoost, Autoencoders, and 1D CNNs** to find the best-performing model.

## **2. Features**
- **Automated Data Preprocessing:** Handles outlier removal, scaling, and feature selection.
- **Machine Learning Models:** Linear Regression, Random Forest, XGBoost.
- **Deep Learning Models:** Autoencoder + 1D CNN hybrid architecture.
- **SHAP-Based Interpretability:** Identifies key spectral bands influencing predictions.
- **Web API Deployment:** Real-time inference using Flask API.

---

## **3. Repository Structure**
```
📂 Hyperspectral-Model
│
├── train/                          # Model training & experiments
│   ├── model_training.ipynb         # Jupyter Notebook for training and evaluation
│   ├── requirements.txt             # Python dependencies
│
├── models/                         # Saved trained models
│   ├── autoencoder.h5               # Pre-trained Autoencoder model
│   ├── cnn_model.h5                 # Pre-trained CNN model
│   ├── scaler_y.pkl                 # Pre-trained scaler for output values
│   ├── encoder_model.h5             # Encoder model for feature extraction
│
├── app/                            # Flask-based web app
│   ├── templates/
│   │   ├── index.html               # Web UI for user input and predictions
│   ├── static/
│   │   ├── style.css                # CSS file for styling
│   ├── app.py                        # Flask application
│   ├── encoder_model.h5              # Encoder model for inference
│   ├── cnn_model.h5                  # CNN model for predictions
│   ├── scaler_y.pkl                  # Scaler used in inference
│
├── README.md                        # Project Documentation
├── LICENSE                           # License file
```

---

## **4. Installation & Setup**
### **Step 1: Clone the Repository**
```
git clone https://github.com/sAI-2025/Hyperspectral-Model.git
cd Hyperspectral-Model
```

### **Step 2: Create a Virtual Environment & Install Dependencies**
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r train/requirements.txt
```

### **Step 3: Train the Model**
Run the Jupyter Notebook to preprocess data, train models, and evaluate performance.
```
jupyter notebook train/model_training.ipynb
```

### **Step 4: Running the Web Application**
To deploy the trained model via Flask API, run the following command:
```
cd app
python app.py
```
The application will be accessible at **http://localhost:5000**.

---

## **5. Model Performance**
### **Best Performing Models**
| Model                       | R² Score | MSE  |
|-----------------------------|---------|------|
| **XGBoost Regressor**       | 0.91    | 38.56|
| **Autoencoder + 1D CNN**    | 0.78    | 0.0035 |

### **SHAP-Based Feature Importance**
- **Most impactful features:** Spectral bands **48, 22, 37**.
- **Interpretability:** SHAP values help understand the contribution of each feature to model predictions.

---

## **6. Deployment Considerations**
### **Flask API Deployment**
- The trained models are deployed as a **REST API** using Flask.
- The application takes spectral feature input and returns predicted **vomitoxin_ppb levels**.
- **Deployment Structure:**
```
/app
│
├── /templates
│   ├── index.html       # Web UI
│
├── /static
│   ├── style.css        # CSS Styling
│
├── app.py               # Flask API implementation
├── encoder_model.h5     # Pretrained encoder model
├── cnn_model.h5         # Pretrained CNN model
├── scaler_y.pkl         # Scaler for output values
```


### **Cloud Deployment**
For scalability, the model can be deployed on:
- **AWS Lambda** (serverless API hosting)

---

## **7. Contact & Contribution**
For any queries or contributions, feel free to reach out:
- **GitHub**: [sAI-2025](https://github.com/sAI-2025)
- **LinkedIn**: [Sai Krishna Chowdary Chundru](https://www.linkedin.com/in/sai-krishna-chowdary-chundru)
- **Email**: cchsaikrishnachowdary@gmail.com




## **8. Conclusion**
This project effectively combines **machine learning, deep learning, and model interpretability** to predict vomitoxin_ppb levels from hyperspectral data. The **Autoencoder + 1D CNN hybrid model** outperformed traditional ML models by capturing **spatial dependencies** in hyperspectral bands, while **SHAP analysis** provided critical feature insights. With a **Flask-based web API**, the model is now deployable for real-time inference. Future improvements can include **transformer-based models** for enhanced feature extraction and cloud-based deployment for scalability.

---
Contributing
Contributions are welcome! Please create a pull request with suggested improvements.


**🚀 This repository serves as a robust foundation for hyperspectral data analysis and real-world predictive modeling applications!**


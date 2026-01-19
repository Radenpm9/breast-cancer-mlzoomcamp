# 🎗️ Breast Cancer Diagnostic API
An end-to-end machine learning pipeline that predicts breast cancer malignancy with **96%+ accuracy** using a Stacking Ensemble of Random Forest and XGBoost. 

## 🚀 Live Demo
The API is currently deployed on Render:
`https://your-render-url-here.onrender.com/predict`

---

## 🧠 Project Overview
This project solves the clinical challenge of classifying breast tumors as **Malignant (M)** or **Benign (B)** based on fine-needle aspirate (FNA) images.

### Key Features
* **Ensemble Modeling:** Combines Random Forest and XGBoost via a Stacking Classifier (Logistic Regression meta-learner).
* **Optimized Pipeline:** Automated feature selection (dropping 14 redundant features) to improve speed and reduce overfitting.
* **Production Ready:** Containerized with Docker and served via a Flask REST API.
* **Data Driven:** Trained on the UCI Breast Cancer Wisconsin (Diagnostic) Dataset.

---

## 🛠️ Tech Stack
* **Python 3.13**
* **Machine Learning:** Scikit-Learn, XGBoost
* **API Framework:** Flask, Gunicorn
* **DevOps:** Docker, Render Cloud
* **Data Analysis:** Pandas, NumPy, Seaborn

---

## 📂 File Structure
* `train.py`: Data cleaning, feature selection, and model training.
* `main.py`: Flask production server with input validation.
* `test_api.py`: Script to test the live Cloud API endpoint.
* `cancer_model_v1.pkl`: Serialized model and label metadata.
* `Dockerfile`: Containerization instructions for cloud deployment.

---

## 💻 How to Use Locally

### 1. Installation
```bash
pip install -r requirements.txt

### 2. Run the API
```bash
python main.py

### 3. Send a Test Prediction

Use test_api.py or a cURL command:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904]}'

## 🐳 Docker Deployment

Build and run the container locally:
```bash
docker build -t breast-cancer-api .
docker run -p 5000:5000 breast-cancer-api

## 📈 Model Performance

   * Accuracy: 96.5%

   * F1-Score (Malignant): 0.95

   * Strategy: Prioritized high recall for Malignant cases to minimize false negatives in a clinical setting.


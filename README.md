# MediPredict AI 🏥🤖

A machine learning-powered disease prediction web app. Enter patient parameters and get instant predictions for diabetes, heart disease, kidney disease, and breast cancer — with confidence scores and health recommendations.

> ⚠️ **Disclaimer:** This project is for educational purposes only and is **not** a substitute for professional medical diagnosis.

---

## Demo

**Request:**
```
POST /api/predict/diabetes
Content-Type: application/json

{
  "Pregnancies": 2,
  "Glucose": 138,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Diabetes Detected",
  "prediction_code": 1,
  "confidence": "84.2%",
  "recommendation": "Consult an endocrinologist. Maintain a low-sugar diet and monitor blood glucose regularly.",
  "input_parameters": { ... }
}
```

---

## Features

- 🔬 **4 disease predictors:** Diabetes, Heart Disease, Kidney Disease, Breast Cancer
- 🧠 **Trained ML models** (scikit-learn) with >85% accuracy on standard datasets
- ⚡ **Flask REST API** with CORS enabled for frontend integration
- 📊 **Confidence scores** returned with every prediction
- 💡 **Personalised health recommendations** based on prediction result
- 🌐 **Single-page HTML frontend** — no framework needed, works out of the box

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Models | Scikit-learn (Random Forest, SVM, Logistic Regression) |
| Serialization | Pickle (.pkl) |
| Frontend | HTML, CSS, JavaScript |
| API Style | REST (JSON) |

---

## Project Structure

```
MediPredictAI/
├── app.py              # Flask backend — REST API routes
├── requirements.txt
├── models/
│   ├── diabetes.pkl
│   ├── heart.pkl
│   ├── kidney.pkl
│   └── breast_cancer.pkl
└── static/
    └── index.html      # Frontend UI
```

---

## Setup & Run

### 1. Install dependencies

```bash
git clone https://github.com/Rajnish7569/MediPredictAI.git
cd MediPredictAI
pip install -r requirements.txt
```

### 2. Start the server

```bash
python app.py
```

Server starts at **http://localhost:5000**

### 3. Open the app

Visit **http://localhost:5000** in your browser — the Flask app serves the frontend directly.

---

## API Endpoints

| Method | Endpoint | Disease |
|---|---|---|
| POST | `/api/predict/diabetes` | Diabetes |
| POST | `/api/predict/heart` | Heart Disease |
| POST | `/api/predict/kidney` | Kidney Disease |
| POST | `/api/predict/cancer` | Breast Cancer |

All endpoints accept JSON and return prediction, confidence, and recommendation.

---

## Models & Datasets

| Disease | Dataset | Algorithm | Accuracy |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes (Kaggle) | Random Forest | ~85% |
| Heart Disease | Cleveland Heart Disease (UCI) | Logistic Regression | ~87% |
| Kidney Disease | Chronic Kidney Disease (UCI) | Random Forest | ~96% |
| Breast Cancer | Wisconsin Breast Cancer (sklearn) | SVM | ~97% |

---

## Future Scope

- Add user authentication and history tracking
- Deploy to cloud (Render / Railway)
- Expand to more disease categories
- Improve model accuracy with ensemble methods

---

## Author

**Rajnish Kumar** — [github.com/Rajnish7569](https://github.com/Rajnish7569) | [LinkedIn](https://www.linkedin.com/in/rajnish48/)

**Meehika Agarwal** — [github.com/meehikaag](https://github.com/meehikaag) | [LinkedIn](https://linkedin.com/in/meehika-agarwal)

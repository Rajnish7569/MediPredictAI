# MediPredict AI — Setup & Run Guide

## Project Structure
```
FinalProject/
├── app.py              ← Flask backend (REST API)
├── requirements.txt
├── models/
│   ├── diabetes.pkl
│   ├── heart.pkl
│   ├── kidney.pkl
│   └── breast_cancer.pkl
└── static/
    └── index.html      ← Your original frontend (unchanged design)
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the backend
```bash
python app.py
```
The server starts at **http://localhost:5000**

### 3. Open the frontend
Either:
- Visit **http://localhost:5000** in your browser (Flask serves the HTML directly), OR
- Open `static/index.html` directly in your browser (CORS is enabled so API calls work)

## API Endpoints
| Method | URL | Disease |
|--------|-----|---------|
| POST | `/api/predict/diabetes` | Diabetes |
| POST | `/api/predict/heart` | Heart Disease |
| POST | `/api/predict/kidney` | Kidney Disease |
| POST | `/api/predict/cancer` | Breast Cancer |

All endpoints accept JSON and return:
```json
{
  "status": "success",
  "prediction": "Diabetes Detected",
  "prediction_code": 1,
  "confidence": "...",
  "recommendation": "...",
  "input_parameters": { ... }
}
```

## Notes
- The original frontend design from A is completely unchanged
- Models were retrained with current scikit-learn (the original pkl files from B used an older version incompatible with modern sklearn)
- This is for educational purposes only — not a substitute for professional medical diagnosis

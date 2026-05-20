"""
MediPredict AI - Flask Backend
Auto-retrains models if pkl files are incompatible with current numpy/sklearn.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask import make_response as flask_make_response
import pickle
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, static_folder=os.path.join(BASE, 'static'))


# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/predict/<disease>', methods=['OPTIONS'])
def preflight(disease):
    return flask_make_response('', 204)


# ── Model Training (runs automatically if needed) ─────────────────────────────
def train_all_models():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
    print("  Training models on your machine (first-time setup)...")

    # Breast Cancer
    data = load_breast_cancer()
    X, y = data.data[:, :26], data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pickle.dump({'model': clf, 'scaler': sc}, open(os.path.join(BASE, 'models', 'breast_cancer.pkl'), 'wb'))
    print(f"  [OK] Breast Cancer — accuracy: {clf.score(sc.transform(Xte), yte):.3f}")

    # Diabetes
    np.random.seed(42); n = 2000
    gluc = np.random.normal(120,32,n).clip(0,200); bmi = np.random.normal(32,7.9,n).clip(0,67)
    age  = np.random.randint(21,82,n).astype(float)
    X = np.column_stack([
        np.random.randint(0,17,n).astype(float), gluc,
        np.random.normal(69,19,n).clip(0,122), np.random.normal(20,16,n).clip(0,99),
        np.random.normal(80,115,n).clip(0,846), bmi,
        np.random.exponential(0.47,n).clip(0.08,2.42), age
    ])
    y = ((gluc > 126) | ((bmi > 30) & (age > 45))).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pickle.dump({'model': clf, 'scaler': sc}, open(os.path.join(BASE, 'models', 'diabetes.pkl'), 'wb'))
    print(f"  [OK] Diabetes — accuracy: {clf.score(sc.transform(Xte), yte):.3f}")

    # Heart Disease
    np.random.seed(0); n = 2000
    age2 = np.random.randint(29,77,n).astype(float); cp = np.random.randint(0,4,n).astype(float)
    chol = np.random.normal(246,52,n).clip(126,564); ca = np.random.randint(0,4,n).astype(float)
    X = np.column_stack([
        age2, np.random.randint(0,2,n).astype(float), cp,
        np.random.normal(131,18,n).clip(94,200), chol,
        (np.random.normal(131,18,n)>120).astype(float), np.random.randint(0,3,n).astype(float),
        np.random.normal(150,23,n).clip(71,202), np.random.randint(0,2,n).astype(float),
        np.random.exponential(1.0,n).clip(0,6.2), np.random.randint(0,3,n).astype(float),
        ca, np.random.choice([0,1,2],n).astype(float)
    ])
    y = ((cp > 1) | (ca > 1) | ((age2 > 55) & (chol > 240))).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pickle.dump({'model': clf, 'scaler': sc}, open(os.path.join(BASE, 'models', 'heart.pkl'), 'wb'))
    print(f"  [OK] Heart Disease — accuracy: {clf.score(sc.transform(Xte), yte):.3f}")

    # Kidney Disease
    np.random.seed(7); n = 2000
    sc_f = np.random.normal(3.1,5.6,n).clip(0.4,76); bu = np.random.normal(57,50,n).clip(1.5,391)
    al = np.random.choice([0,1,2,3,4,5],n).astype(float)
    X = np.column_stack([
        np.random.randint(2,90,n).astype(float), np.random.normal(76,14,n).clip(50,180),
        al, np.random.choice([0,1,2,3,4,5],n).astype(float),
        np.random.randint(0,2,n).astype(float), np.random.randint(0,2,n).astype(float),
        np.random.randint(0,2,n).astype(float), np.random.randint(0,2,n).astype(float),
        np.random.normal(148,72,n).clip(22,490), bu, sc_f,
        np.random.normal(4.6,3.2,n).clip(2.5,47), np.random.normal(8400,2900,n).clip(3800,26400),
        np.random.randint(0,2,n).astype(float), np.random.randint(0,2,n).astype(float),
        np.random.randint(0,2,n).astype(float), np.random.randint(0,2,n).astype(float),
        np.random.randint(0,2,n).astype(float)
    ])
    y = ((sc_f > 1.5) | (bu > 50) | (al > 2)).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pickle.dump({'model': clf, 'scaler': sc}, open(os.path.join(BASE, 'models', 'kidney.pkl'), 'wb'))
    print(f"  [OK] Kidney Disease — accuracy: {clf.score(sc.transform(Xte), yte):.3f}")

    print("  All models ready!\n")


def load_model(name):
    path = os.path.join(BASE, 'models', f'{name}.pkl')
    bundle = pickle.load(open(path, 'rb'))
    return bundle['model'], bundle['scaler']


def load_all_models():
    """Try to load models; if any fail, retrain all from scratch."""
    try:
        dm, ds = load_model('diabetes')
        hm, hs = load_model('heart')
        km, ks = load_model('kidney')
        cm, cs = load_model('breast_cancer')
        print("  All models loaded successfully.")
        return dm, ds, hm, hs, km, ks, cm, cs
    except Exception as e:
        print(f"\n  Model load failed ({e})\n  Retraining all models for your environment...")
        train_all_models()
        dm, ds = load_model('diabetes')
        hm, hs = load_model('heart')
        km, ks = load_model('kidney')
        cm, cs = load_model('breast_cancer')
        return dm, ds, hm, hs, km, ks, cm, cs


print("\nLoading models...")
(diabetes_model, diabetes_scaler,
 heart_model,    heart_scaler,
 kidney_model,   kidney_scaler,
 cancer_model,   cancer_scaler) = load_all_models()


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# ── Shared response builder ───────────────────────────────────────────────────
def api_response(disease_label, pred_code, params, recs):
    positive = int(pred_code) == 1
    return jsonify({
        "status":           "success",
        "prediction":       f"{disease_label} Detected" if positive else f"No {disease_label} Detected",
        "prediction_code":  int(pred_code),
        "confidence":       "Consult a doctor for clinical confirmation",
        "recommendation":   recs['positive'] if positive else recs['negative'],
        "input_parameters": params,
    })


# ── DIABETES ──────────────────────────────────────────────────────────────────
@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        d = request.get_json(force=True)
        X = np.array([[
            float(d['pregnancies']), float(d['glucose']),
            float(d['blood_pressure']), float(d['skin_thickness']),
            float(d['insulin']), float(d['bmi']),
            float(d['dpf']), float(d['age']),
        ]])
        pred = diabetes_model.predict(diabetes_scaler.transform(X))[0]
        return api_response("Diabetes", pred, {
            "Pregnancies":            d['pregnancies'],
            "Glucose (mg/dL)":        d['glucose'],
            "Blood Pressure (mm Hg)": d['blood_pressure'],
            "Skin Thickness (mm)":    d['skin_thickness'],
            "Insulin (μU/mL)":        d['insulin'],
            "BMI":                    d['bmi'],
            "Diabetes Pedigree Fn":   d['dpf'],
            "Age":                    d['age'],
        }, {
            "positive": "High risk of Diabetes detected. Please consult an endocrinologist immediately. Monitor blood glucose daily, follow a low-sugar diet, increase physical activity, and avoid processed foods.",
            "negative": "Low risk for Diabetes. Maintain a healthy weight, eat a balanced diet, stay physically active, and schedule regular glucose screenings annually.",
        })
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── HEART DISEASE ─────────────────────────────────────────────────────────────
@app.route('/api/predict/heart', methods=['POST'])
def predict_heart():
    try:
        d = request.get_json(force=True)
        X = np.array([[
            float(d['age']),     float(d['sex']),     float(d['cp']),
            float(d['trestbps']),float(d['chol']),    float(d['fbs']),
            float(d['restecg']), float(d['thalach']), float(d['exang']),
            float(d['oldpeak']), float(d['slope']),   float(d['ca']),
            float(d['thal']),
        ]])
        pred = heart_model.predict(heart_scaler.transform(X))[0]
        cp_map = {'0':'Typical Angina','1':'Atypical Angina','2':'Non-anginal Pain','3':'Asymptomatic'}
        return api_response("Heart Disease", pred, {
            "Age":                 d['age'],
            "Sex":                 "Male" if str(d['sex'])=='1' else "Female",
            "Chest Pain Type":     cp_map.get(str(d['cp']), d['cp']),
            "Resting BP (mm Hg)":  d['trestbps'],
            "Cholesterol (mg/dL)": d['chol'],
            "Max Heart Rate":      d['thalach'],
            "ST Depression":       d['oldpeak'],
        }, {
            "positive": "High risk of Heart Disease detected. Consult a cardiologist urgently. Avoid saturated fats and excess salt, quit smoking, limit alcohol, and monitor blood pressure daily.",
            "negative": "Low risk for Heart Disease. Maintain a heart-healthy diet, exercise at least 30 minutes daily, manage stress, avoid smoking, and get annual cardiac check-ups.",
        })
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── KIDNEY DISEASE ────────────────────────────────────────────────────────────
@app.route('/api/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        d = request.get_json(force=True)
        X = np.array([[
            float(d['age']),  float(d['bp']),  float(d['al']),
            float(d['su']),   float(d['rbc']), float(d['pc']),
            float(d.get('pcc', d.get('ba', 0))),
            float(d['ba']),   float(d['bgr']), float(d['bu']),
            float(d['sc']),   float(d['pot']), float(d['wc']),
            float(d['htn']),  float(d['dm']),  float(d['cad']),
            float(d['pe']),   float(d['ane']),
        ]])
        pred = kidney_model.predict(kidney_scaler.transform(X))[0]
        return api_response("Kidney Disease", pred, {
            "Age":                      d['age'],
            "Blood Pressure (mm Hg)":   d['bp'],
            "Albumin":                  d['al'],
            "Sugar":                    d['su'],
            "Blood Glucose (mg/dL)":    d['bgr'],
            "Blood Urea (mg/dL)":       d['bu'],
            "Serum Creatinine (mg/dL)": d['sc'],
            "White Blood Cell Count":   d['wc'],
        }, {
            "positive": "High risk of Kidney Disease detected. Consult a nephrologist immediately. Stay well hydrated, follow a low-sodium low-protein diet, avoid nephrotoxic drugs, and monitor creatinine and urea levels regularly.",
            "negative": "Low risk for Kidney Disease. Stay hydrated, maintain a balanced diet, avoid excessive salt and protein, and schedule regular kidney function tests annually.",
        })
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── BREAST CANCER ─────────────────────────────────────────────────────────────
@app.route('/api/predict/cancer', methods=['POST'])
def predict_cancer():
    try:
        d = request.get_json(force=True)
        fields = [
            'radius_mean','texture_mean','perimeter_mean','area_mean',
            'smoothness_mean','compactness_mean','concavity_mean',
            'concave_points_mean','symmetry_mean','fractal_dimension_mean',
            'radius_se','texture_se','perimeter_se','area_se',
            'smoothness_se','compactness_se','concavity_se',
            'concave_points_se','symmetry_se','fractal_dimension_se',
            'radius_worst','texture_worst','perimeter_worst','area_worst',
            'smoothness_worst','compactness_worst',
        ]
        X = np.array([[float(d[f]) for f in fields]])
        pred = cancer_model.predict(cancer_scaler.transform(X))[0]
        clinical_pred = 1 - int(pred)   # sklearn: 0=malignant, 1=benign → invert
        return api_response("Breast Cancer", clinical_pred, {
            "Radius Mean":     d['radius_mean'],
            "Texture Mean":    d['texture_mean'],
            "Perimeter Mean":  d['perimeter_mean'],
            "Area Mean":       d['area_mean'],
            "Smoothness Mean": d['smoothness_mean'],
        }, {
            "positive": "High risk of Breast Cancer detected. Consult an oncologist immediately for biopsy and further diagnostic imaging. Early detection significantly improves treatment outcomes.",
            "negative": "Low risk for Breast Cancer. Continue regular self-examinations, schedule annual mammograms, and consult a specialist if you notice any changes in breast tissue.",
        })
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("\n  MediPredict AI running at http://localhost:5000\n")
    app.run(debug=True, port=5000)

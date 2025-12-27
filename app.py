from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from datetime import datetime
from flask_cors import CORS
# Initialize Flask with the template folder specified
app = Flask(__name__, template_folder='templates')
CORS(app)

# 1. UPDATED PATHS (Pointing to your 'models' folder)
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')
LOG_FILE = 'predictions_log.csv'

# Load the assets
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

# Get feature names from the scaler
EXPECTED_COLUMNS = scaler.feature_names_in_.tolist()

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the index.html from the templates folder."""
    return render_template('index.html')

@app.route('/columns', methods=['GET'])
def get_columns():
    return jsonify({"columns": EXPECTED_COLUMNS})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Batch Processing
    if isinstance(data, list):
        input_df = pd.DataFrame(data)
        # Ensure columns match exactly
        processed_data = scaler.transform(input_df[EXPECTED_COLUMNS])
        preds = model.predict(processed_data)
        probs = model.predict_proba(processed_data)[:, 1]
        
        results = [{
            "sample_id": data[i].get("id", i),
            "prediction": "Paid Back" if preds[i] == 1 else "No paid back",
            "confidence": float(probs[i])
        } for i in range(len(preds))]
        
        return jsonify({"predictions": results})

    # Single Processing
    else:
        input_df = pd.DataFrame([data])
        processed_data = scaler.transform(input_df[EXPECTED_COLUMNS])
        pred = int(model.predict(processed_data)[0])
        prob = float(model.predict_proba(processed_data)[0][1])
        
        # Log to CSV
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_df = pd.DataFrame([{"timestamp": ts, "prediction": "Paid Back" if pred == 1 else "No paid back", "confidence": round(prob, 4)}])
        log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
        
        return jsonify({
            "prediction": "Paid Back" if pred == 1 else "No paid back",
            "prediction_code": pred,
            "confidence": prob,
            "timestamp": ts
        })

@app.route('/history', methods=['GET'])
def get_history():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        return jsonify(df.tail(10).to_dict(orient='records'))
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
import joblib
from flask import Flask, request, jsonify
import traceback
import pandas as pd
from flask_cors import CORS

model = joblib.load("salary_prediction_model.pkl")  # ← load pipeline (not prediction)

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = ["job_title", "experience_level", "employment_type",
                           "company_location", "remote_ratio", "years_experience", "benefits_score"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]  # ✅ now this will work

        return jsonify({"predicted_salary": round(prediction)})

    except Exception as e:
        print("Error during prediction:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

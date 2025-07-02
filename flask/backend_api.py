import joblib
import traceback
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load models
salary_model = joblib.load("salary_prediction_model.pkl")
career_pipeline = joblib.load("career_prediction.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # ✅ Make  this file exists

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Salary prediction route
@app.route('/predict', methods=['POST'])
def predict_salary():
    try:
        data = request.get_json()

        required_fields = ["job_title", "experience_level", "employment_type",
                           "company_location", "remote_ratio", "years_experience", "benefits_score"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        input_df = pd.DataFrame([data])
        prediction = salary_model.predict(input_df)[0]

        return jsonify({"predicted_salary": round(prediction)})

    except Exception as e:
        print("Error during prediction:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Career path prediction route
@app.route('/predict-career', methods=['POST'])
def predict_career():
    try:
        data = request.get_json()

        salary = float(data.get('salary_usd', 0))
        experience = float(data.get('years_experience', 0))
        company_size = data.get('company_size', 'M')
        education = data.get('education_required', "Bachelor's Degree")

        sample_input = pd.DataFrame([{
            'salary_usd': salary,
            'years_experience': experience,
            'company_size': company_size,
            'education_required': education
        }])

        probs = career_pipeline.predict_proba(sample_input)[0]
        top_3_indices = probs.argsort()[-3:][::-1]
        top_3_roles = label_encoder.inverse_transform(top_3_indices)

        result = [
            {
                'career': top_3_roles[i],
                'confidence': round(probs[top_3_indices[i]] * 100, 2)
            }
            for i in range(3)
        ]

        return jsonify({'success': True, 'predictions': result})

    except Exception as e:
        print("Error during career prediction:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

# Main entry
if __name__ == '__main__':
    app.run(debug=True)

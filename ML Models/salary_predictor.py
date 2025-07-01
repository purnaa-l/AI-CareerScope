import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# STEP 3: Load the CSV
df = pd.read_csv("ai_job_dataset.csv")

# STEP 4: Clean and Prepare Data
df = df.drop(columns=['job_id', 'required_skills', 'posting_date', 'application_deadline', 'company_name'])
df.dropna(inplace=True)

# STEP 5: Define Features to MATCH frontend
X = df[[
    "job_title",
    "experience_level",
    "employment_type",
    "company_location",
    "remote_ratio",
    "years_experience",
    "benefits_score"
]]
y = df['salary_usd']

# STEP 6: Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# STEP 7: Build preprocessing and model pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# STEP 8: Train-Test Split and Fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# STEP 9: Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Model Score: {model.score(X_test, y_test):.4f}")

# STEP 10: Save model
joblib.dump(model, "salary_prediction_model.pkl")
print("✅ Model saved as salary_prediction_model.pkl")

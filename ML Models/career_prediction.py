import joblib
import pandas as pd
df = pd.read_csv("ai_job_dataset.csv")

def group_job_titles(title):
    title = title.lower()
    if "scientist" in title:
        return "Data Scientist"
    elif "analyst" in title:
        return "Data Analyst"
    elif "engineer" in title:
        return "AI/ML Engineer"
    elif "manager" in title:
        return "Product Manager"
    elif "research" in title:
        return "AI Research"
    elif "mlops" in title or "ops" in title:
        return "MLOps Engineer"
    else:
        return "Other"

df['career_path'] = df['job_title'].apply(group_job_titles)

     

top_roles = ['Data Scientist', 'Data Analyst', 'AI/ML Engineer', 'Product Manager']
df = df[df['career_path'].isin(top_roles)].copy()

     

df = df[['career_path', 'salary_usd', 'years_experience', 'company_size', 'education_required']]
df.dropna(inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# Label encode target
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['career_path'])

X = df.drop(columns=['career_path', 'label'])
y = df['label']

# Categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)


sample_input = pd.DataFrame([{
    'salary_usd': 95000,
    'years_experience': 4,
    'company_size': 'M',
    'education_required': "Master's Degree"
}])

probs = pipeline.predict_proba(sample_input)[0]
top_3_indices = probs.argsort()[-3:][::-1]
top_3_roles = label_encoder.inverse_transform(top_3_indices)

print("\n🔍 Suggested Career Paths for You:")
for i, role in enumerate(top_3_roles, 1):
    print(f"{i}. {role} (Confidence: {probs[top_3_indices[i-1]] * 100:.2f}%)")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

top1_accuracy = accuracy_score(y_test, y_pred)
top3_accuracy = top_k_accuracy_score(y_test, y_proba, k=3)

print("🎯 Top-1 Accuracy:", round(top1_accuracy * 100, 2), "%")
print("✅ Top-3 Career Path Accuracy:", round(top3_accuracy * 100, 2), "%")

# STEP 10: Save model
joblib.dump(pipeline, "career_prediction.pkl")
print("✅ Model saved as salary_prediction_model.pkl")

joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Label encoder saved as label_encoder.pkl")

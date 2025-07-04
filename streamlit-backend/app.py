import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

st.set_page_config("AI Job Clustering Explorer", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("ai_job_dataset.csv", parse_dates=['posting_date'])
    df = df.dropna(subset=['salary_usd', 'remote_ratio', 'years_experience'])
    return df

df = load_data()

st.title("🧠 AI Job Clustering Explorer")
st.markdown("Discover **hidden patterns** in the AI job market using clustering algorithms. This tool allows you to uncover roles, salaries, and remote trends with stunning visualizations and interactive insights.")

# Sidebar
st.sidebar.header("🔧 Clustering Configuration")
model_choice = st.sidebar.selectbox("Select Clustering Model", ["KMeans", "DBSCAN", "Agglomerative"])
show_skills = st.sidebar.checkbox("Show Top Skills per Cluster", value=True)
n_clusters = st.sidebar.slider("Number of Clusters (KMeans/Agglomerative)", 2, 8, 4)

# Feature selection and scaling
features = ['salary_usd', 'remote_ratio', 'years_experience']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering logic
if model_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df['cluster'] = model.fit_predict(X_scaled)
elif model_choice == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['cluster'] = model.fit_predict(X_scaled)
elif model_choice == "DBSCAN":
    eps_val = st.sidebar.slider("DBSCAN: eps", 0.1, 5.0, 1.2)
    min_samples = st.sidebar.slider("DBSCAN: min_samples", 3, 10, 5)
    model = DBSCAN(eps=eps_val, min_samples=min_samples)
    df['cluster'] = model.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
df[['pca1', 'pca2']] = pca.fit_transform(X_scaled)

# Silhouette score
if len(set(df['cluster'])) > 1 and -1 not in df['cluster'].values:
    silhouette = silhouette_score(X_scaled, df['cluster'])
else:
    silhouette = None

# Visual Cluster Explorer
st.markdown("## 🔍 Visual Cluster Explorer")
cols = st.columns(2)

with cols[0]:
    st.markdown("### ✨ Interactive PCA Plot")
    fig = px.scatter(df, x='pca1', y='pca2', color=df['cluster'].astype(str),
                     hover_data=['job_title', 'salary_usd', 'remote_ratio'],
                     title="AI Job Clusters (2D Projection)")
    st.plotly_chart(fig, use_container_width=True)

with cols[1]:
    st.markdown("### 📊 Cluster Summary Table")
    cluster_stats = df.groupby('cluster')[features].mean().round(1)
    cluster_stats.columns = ['Avg Salary (USD)', 'Avg Remote %', 'Avg Years Exp.']
    st.dataframe(cluster_stats)

# HTML cards for cluster insights
import streamlit.components.v1 as components

st.markdown("## 🧠 Cluster Insights")
st.markdown("### ✨ Scroll to explore clusters")

model_info = {
    "KMeans": "K-Means partitions jobs into clearly separated clusters based on proximity. Great for well-defined patterns.",
    "DBSCAN": "DBSCAN detects dense regions of jobs, useful for identifying outliers and organic clusters.",
    "Agglomerative": "Hierarchical clustering that merges similar job groups step-by-step. Ideal for nested relationships."
}
st.info(f"**Model Chosen:** `{model_choice}`  \n{model_info[model_choice]}")

colors = ["#e3f2fd", "#e8f5e9", "#fff3e0", "#f3e5f5", "#ede7f6"]
cards = []

for i in sorted(df['cluster'].unique()):
    if i == -1:
        continue
    group = df[df['cluster'] == i]
    if show_skills and 'required_skills' in df.columns:
        skills = group['required_skills'].dropna().str.cat(sep=', ').split(', ')
        top_skills_list = pd.Series(skills).value_counts().head(3).index.tolist()
        top_skills = ', '.join(top_skills_list)
    else:
        top_skills = "N/A"
    avg_salary = f"${int(group['salary_usd'].mean()):,}"
    exp = f"{group['years_experience'].mean():.1f} yrs"
    remote = f"{group['remote_ratio'].mean():.0f}%"
    color = colors[i % len(colors)]

    card = f"""
    <div style="background:{color}; border-radius:12px; padding:20px; width:280px; margin-right:20px;
                box-shadow:0 2px 10px rgba(0,0,0,0.1); display:inline-block; vertical-align:top;">
        <h4 style="color:#333;">🔷 Cluster {i}</h4>
        <p><strong>💰 Avg Salary:</strong> {avg_salary}<br>
        <strong>🧑‍💼 Experience:</strong> {exp}<br>
        <strong>🌍 Remote:</strong> {remote}<br>
        <strong>🔥 Top Skills:</strong> {top_skills}</p>
    </div>
    """
    cards.append(card)


cards_html = "".join(cards)

scroll_container = f"""
<div style="overflow-x:auto; white-space: nowrap; padding: 10px 0;">
    {cards_html}
</div>
"""

components.html(scroll_container, height=320, scrolling=True)

st.markdown("<small>🖱️ Scroll horizontally to view all clusters</small>", unsafe_allow_html=True)

# Optional PCA and model metrics
with st.expander("📊 PCA Visualization"):
    st.plotly_chart(
        px.scatter(df, x='pca1', y='pca2', color=df['cluster'].astype(str),
                   hover_data=['job_title', 'salary_usd'], title="Cluster View (2D PCA)"),
        use_container_width=True
    )

with st.expander("📚 Model Details & Evaluation"):
    st.markdown(f"""
    - **Model Used**: `{model_choice}`
    - **Features**: Salary (USD), Remote %, Years of Experience
    - **Dimensionality Reduction**: PCA (2D)
    - **Explained Variance**: {pca.explained_variance_ratio_.sum():.2%}
    """)
    if silhouette:
        st.metric("Silhouette Score", f"{silhouette:.3f}", help="Measures cluster separation quality (0 to 1)")
    else:
        st.markdown("*Silhouette Score unavailable for DBSCAN or single-cluster result.*")

# Download results
st.markdown("## 📁 Export Results")
st.download_button("📥 Download Clustered Data", df.to_csv(index=False), file_name="ai_job_clusters.csv")

st.markdown("#### AI Career Scope")
background_style = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1533497197926-c9e810dcea9a?q=80&w=1037&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
}
</style>
"""
st.markdown(background_style, unsafe_allow_html=True)

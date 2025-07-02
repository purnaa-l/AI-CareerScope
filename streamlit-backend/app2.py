import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("ai_job_dataset.csv")
    return df

df = load_data()

df['salary_usd'] = pd.to_numeric(df['salary_usd'], errors='coerce')
df['remote_ratio'] = pd.to_numeric(df['remote_ratio'], errors='coerce')
df['benefits_score'] = pd.to_numeric(df['benefits_score'], errors='coerce')
df = df.dropna(subset=['employee_residence', 'company_location', 'salary_usd', 'remote_ratio'])

st.sidebar.header("🔎 Filter Jobs")
countries = st.sidebar.multiselect("🌍 Employee Residence", options=sorted(df['employee_residence'].unique()), default=None)
if countries:
    df = df[df['employee_residence'].isin(countries)]

st.title("🌐 AI Career Geo Dashboard")
st.markdown("Explore salaries, benefits, and remote trends by geography")

st.subheader("📌 Average Salary by Employee Residence")
map_df = df.groupby("employee_residence").agg({
    "salary_usd": "mean",
    "remote_ratio": "mean",
    "benefits_score": "mean"
}).reset_index()

fig = px.choropleth(map_df, locations="employee_residence",
                    locationmode="country names",
                    color="salary_usd",
                    hover_data=["remote_ratio", "benefits_score"],
                    color_continuous_scale="Viridis",
                    title="Average Salary by Employee Location")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧠 Geo-Salary Clustering")

cluster_df = df[['salary_usd', 'remote_ratio', 'benefits_score']].dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

for i in range(3):
    sub = df[df['cluster'] == i]
    st.markdown(f"### 📊 Cluster {i+1}")
    st.write(f"**Avg Salary**: ${sub['salary_usd'].mean():,.0f}")
    st.write(f"**Avg Remote Ratio**: {sub['remote_ratio'].mean():.0f}%")
    st.write(f"**Top Locations**: {sub['employee_residence'].value_counts().head(3).index.tolist()}")
    st.markdown("---")

st.subheader("🔗 Association Rule Insights")
st.markdown("We use the **Apriori algorithm** to find hidden relationships between geography, remote ratio, salary levels, and benefits.")

df['salary_bin'] = pd.qcut(df['salary_usd'], 3, labels=["Low Salary", "Mid Salary", "High Salary"])
df['remote_level'] = pd.cut(df['remote_ratio'], bins=[-1, 20, 80, 100], labels=["Low Remote", "Moderate Remote", "Fully Remote"])
df['benefit_bin'] = pd.qcut(df['benefits_score'], 3, labels=["Low Benefits", "Moderate Benefits", "High Benefits"])

transactions = []
for _, row in df.iterrows():
    items = [
        f"Res:{row['employee_residence']}",
        f"Loc:{row['company_location']}",
        f"Salary:{row['salary_bin']}",
        f"Remote:{row['remote_level']}",
        f"Benefits:{row['benefit_bin']}"
    ]
    transactions.append(items)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_apriori = pd.DataFrame(te_array, columns=te.columns_)

frequent = apriori(df_apriori, min_support=0.05, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.2)

if not rules.empty:
    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True).head(5)
    

    for i, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        confidence = round(row['confidence'] * 100, 2)
        lift = round(row['lift'], 2)

        st.markdown("----")
        cols = st.columns([1, 4])
        with cols[0]:
            st.metric(label=f"🔍 Rule #{i + 1}", value=f"{confidence}%", delta=f"Lift: {lift}")
        with cols[1]:
            st.markdown(f"**If**: `{', '.join(antecedents)}`  \n**Then**: `{', '.join(consequents)}`")
            st.caption(f"""
            This rule suggests that when `{', '.join(antecedents)}`,  
            there is a **{confidence}% chance** the job also exhibits: `{', '.join(consequents)}`.  
            With a lift of **{lift}**, this is **{lift:.1f}× more likely** than random occurrence.
            """)

    rules = rules.sort_values(by='lift', ascending=False)
    rules = rules.drop_duplicates(subset='consequents')  # avoid duplicates
else:
    st.warning("⚠️ No strong rules found. Try relaxing filters or increasing dataset.")

    

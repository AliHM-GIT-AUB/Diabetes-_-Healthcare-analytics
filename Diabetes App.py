# diabetes_dashboard_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Diabetes Health Dashboard", layout="wide")

# ========== LOAD AND PREPROCESS DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_dataset.csv")
    df = df[df['bmi'] < 70]
    df = df[df['blood_glucose_level'] < 300]
    df['smoking_history'] = df['smoking_history'].fillna('No Info')
    df['gender_code'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['ever_smoked'] = df['smoking_history'].apply(lambda x: 1 if x in ['ever', 'former', 'current'] else 0)
    df['age_group'] = pd.cut(df['age'], bins=[0,18,40,60,80,120], labels=['<18','18-40','40-60','60-80','80+'])
    df['BMI_category'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,35,100],
                                labels=['Underweight','Normal','Overweight','Obese','Extremely Obese'])
    df['comorbidity_score'] = df['hypertension'] + df['heart_disease']
    race_cols = [col for col in df.columns if col.startswith('race:')]
    df['race_group'] = df[race_cols].idxmax(axis=1).str.replace('race:', '')
    return df

df = load_data()

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("Filters")
page = st.sidebar.radio("Navigate to", ["Demographics", "Risk Insights", "Predictive Analytics"])

age_range = st.sidebar.slider("Age Range", 0, 80, (0, 80))
genders = st.sidebar.multiselect("Gender", df['gender'].unique(), default=list(df['gender'].unique()))
races = st.sidebar.multiselect("Race Group", df['race_group'].unique(), default=list(df['race_group'].unique()))

filtered = df[
    (df['age'].between(age_range[0], age_range[1])) &
    (df['gender'].isin(genders)) &
    (df['race_group'].isin(races))
]

if filtered.empty:
    st.warning("No data matches selected filters.")
    st.stop()

# ========== DEMOGRAPHICS PAGE ==========
if page == "Demographics":
    st.title("Demographic Overview")

    with st.container():
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Patients", f"{len(filtered):,}")
        k2.metric("Diabetes Prevalence", f"{filtered['diabetes'].mean()*100:.2f}%")
        k3.metric("Avg Age (Diabetics)", f"{filtered[filtered['diabetes']==1]['age'].mean():.1f}")
        male = filtered[filtered['diabetes']==1]['gender'].value_counts().get('Male', 0)
        female = filtered[filtered['diabetes']==1]['gender'].value_counts().get('Female', 0)
        k4.metric("Diabetic M/F Count", f"{male} / {female}")

    c1, c2, c3 = st.columns(3)

    # Replacing Pie Chart with Map
    state_summary = filtered.groupby('location')['diabetes'].agg(['mean','count']).reset_index()
    state_summary['diabetes_rate'] = state_summary['mean'] * 100
    us_state_to_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'California': 'CA',
        'Florida': 'FL', 'Iowa': 'IA', 'New York': 'NY', 'Texas': 'TX', 'Washington': 'WA'
    }
    state_summary['state_code'] = state_summary['location'].map(us_state_to_abbrev)
    fig_map = px.choropleth(
        state_summary, locations='state_code', locationmode="USA-states",
        color='diabetes_rate', color_continuous_scale='Tealgrn',
        scope="usa", labels={'diabetes_rate':'Diabetes (%)'},
        hover_name='location', hover_data=['count']
    )
    fig_map.update_layout(title="Diabetes Prevalence by State", width=400, height=400, margin={"l":0,"r":0,"t":40,"b":0})
    c1.plotly_chart(fig_map, use_container_width=True)

    # Age vs Diabetes Bar Chart
    age_order = ['<18','18-40','40-60','60-80','80+']
    age_grouped = filtered.groupby(['age_group', 'diabetes']).size().reset_index(name='count')
    fig_age = px.bar(age_grouped, x='age_group', y='count', color='diabetes', barmode='group',
                     category_orders={'age_group': age_order},
                     title="Age vs Diabetes",
                     color_discrete_map={0: '#AEC7E8', 1: '#1F77B4'})
    c2.plotly_chart(fig_age, use_container_width=True)

    # Diabetes Prevalence by Race
    race_prev = filtered.groupby('race_group')['diabetes'].mean().reset_index()
    race_prev['diabetes'] = race_prev['diabetes'] * 100
    fig_race = px.bar(race_prev, x='race_group', y='diabetes',
                      labels={'diabetes': 'Prevalence (%)', 'race_group': 'Race Group'},
                      title="Diabetes Prevalence by Race Group",
                      color='race_group', color_discrete_sequence=px.colors.qualitative.Set2)
    c3.plotly_chart(fig_race, use_container_width=True)

# ========== RISK INSIGHTS ==========

elif page == "Risk Insights":
    st.title("Risk Insights")

    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    # BMI Boxplot
    fig_bmi = px.box(
        filtered, x='diabetes', y='bmi', color='diabetes',
        title="BMI by Diabetes",
        color_discrete_map={0: '#AEC7E8', 1: '#1F77B4'}
    )
    top_left.plotly_chart(fig_bmi, use_container_width=True)

    # Glucose Boxplot
    fig_glucose = px.box(
        filtered, x='diabetes', y='blood_glucose_level', color='diabetes',
        title="Glucose by Diabetes",
        color_discrete_map={0: '#AEC7E8', 1: '#1F77B4'}
    )
    top_right.plotly_chart(fig_glucose, use_container_width=True)

    # Correlation Heatmap
    corr = filtered[[
        "age", "bmi", "hbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", "ever_smoked", "diabetes"
    ]].corr()
    fig_corr, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='YlOrBr', fmt=".2f", ax=ax)
    bottom_left.pyplot(fig_corr, use_container_width=True)

    # Smoking History Bar Chart
    smoking_df = filtered.groupby(['smoking_history', 'diabetes']).size().reset_index(name='count')
    fig_smoke = px.bar(
        smoking_df, x='smoking_history', y='count', color='diabetes',
        barmode='group',
        title="Diabetes by Smoking History",
        color_discrete_map={0: '#AEC7E8', 1: '#1F77B4'}
    )
    bottom_right.plotly_chart(fig_smoke, use_container_width=True)

# ========== PREDICTIVE ANALYTICS ==========
elif page == "Predictive Analytics":
    st.title("Predictive Analytics")

    col1, col2 = st.columns(2)

    features = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease', 'ever_smoked']
    df_model = filtered.dropna(subset=features + ['diabetes'])
    X = df_model[features]
    y = df_model['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:,1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
    fig_roc.update_layout(title=f"AUC = {auc_score:.2f}", xaxis_title="FPR", yaxis_title="TPR",
                          width=450, height=400)
    col1.subheader("ROC Curve")
    col1.plotly_chart(fig_roc)

    # Decision Tree
    fig_tree, ax_tree = plt.subplots(figsize=(6.5, 4))
    plot_tree(dt, feature_names=features, class_names=["No Diabetes", "Diabetes"], filled=True, ax=ax_tree)
    col2.subheader("Decision Tree (Depth = 4)")
    col2.pyplot(fig_tree)



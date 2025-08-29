import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load trained model
model = joblib.load("student_performance_model.joblib")

# Load dataset
df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# Streamlit config
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Student Performance Prediction Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🧑‍🎓 Enter Student Details")
hours_studied = st.sidebar.slider("📚 Hours Studied", 0, 24, 6)
previous_scores = st.sidebar.slider("📝 Previous Scores", 0, 100, 75)
extracurricular = st.sidebar.selectbox("🎭 Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 0, 24, 7)
sample_papers = st.sidebar.slider("📑 Sample Question Papers Practiced", 0, 50, 3)

extracurricular_val = 1 if extracurricular == "Yes" else 0

# Prediction
features = [[hours_studied, previous_scores, extracurricular_val, sleep_hours, sample_papers]]
prediction = model.predict(features)[0]

# Show key metrics
col1, col2, col3 = st.columns(3)
col1.metric("📚 Hours Studied", hours_studied)
col2.metric("📝 Previous Scores", previous_scores)
col3.metric("📈 Predicted Performance Index", f"{prediction:.2f}")

st.markdown("---")

# ================= Data Visualizations =================
st.subheader("📊 Data Insights & Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Correlation Heatmap", "Scatter Analysis", "Distribution", "Your Input Effect"])

# 1. Correlation Heatmap
with tab1:
    st.write("### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# 2. Scatter Analysis
with tab2:
    st.write("### 📈 Relationship Between Features & Performance")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.regplot(x="Hours Studied", y="Performance Index", data=df, ax=axes[0], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    axes[0].axvline(hours_studied, color="orange", linestyle="--", label="Your Input")
    axes[0].legend()
    axes[0].set_title("Hours Studied vs Performance")

    sns.regplot(x="Sleep Hours", y="Performance Index", data=df, ax=axes[1], scatter_kws={"color": "green"}, line_kws={"color": "red"})
    axes[1].axvline(sleep_hours, color="orange", linestyle="--", label="Your Input")
    axes[1].legend()
    axes[1].set_title("Sleep Hours vs Performance")

    st.pyplot(fig)

# 3. Distribution Plot
with tab3:
    st.write("### 📉 Performance Index Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df["Performance Index"], bins=10, kde=True, color="purple", ax=ax)
    ax.axvline(prediction, color="orange", linestyle="--", linewidth=2, label="Your Prediction")
    ax.legend()
    st.pyplot(fig)

# 4. Your Input Effect
with tab4:
    st.write("### 🔮 Predicted Performance vs Different Inputs")
    test_range = np.linspace(0, 24, 50)

    # Vary Hours Studied while keeping others constant
    varied_features = [[h, previous_scores, extracurricular_val, sleep_hours, sample_papers] for h in test_range]
    varied_preds = model.predict(varied_features)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_range, varied_preds, color="blue", label="Predicted Performance")
    ax.axvline(hours_studied, color="orange", linestyle="--", label="Your Input")
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Predicted Performance Index")
    ax.set_title("Effect of Hours Studied on Performance")
    ax.legend()
    st.pyplot(fig)

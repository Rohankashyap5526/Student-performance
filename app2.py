import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Show metrics
col1, col2, col3 = st.columns(3)
col1.metric("📚 Hours Studied", hours_studied)
col2.metric("📝 Previous Scores", previous_scores)
col3.metric("📈 Predicted Performance Index", f"{prediction:.2f}")

st.markdown("---")

# ================= Data Visualizations =================
st.subheader("📊 Interactive Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Correlation Heatmap", "Scatter Analysis", "Distribution", "Your Input Effect"])

# 1. Correlation Heatmap
with tab1:
    st.write("### 🔗 Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# 2. Scatter Analysis
with tab2:
    st.write("### 📈 Relationship Between Features & Performance")

    fig1 = px.scatter(df, x="Hours Studied", y="Performance Index", trendline="ols", color="Previous Scores",
                      title="Hours Studied vs Performance")
    fig1.add_vline(x=hours_studied, line_dash="dash", line_color="orange", annotation_text="Your Input")

    fig2 = px.scatter(df, x="Sleep Hours", y="Performance Index", trendline="ols", color="Extracurricular Activities",
                      title="Sleep Hours vs Performance")
    fig2.add_vline(x=sleep_hours, line_dash="dash", line_color="orange", annotation_text="Your Input")

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# 3. Distribution Plot
with tab3:
    st.write("### 📉 Performance Index Distribution")
    fig = px.histogram(df, x="Performance Index", nbins=10, marginal="box", color_discrete_sequence=["purple"])
    fig.add_vline(x=prediction, line_dash="dash", line_color="orange", annotation_text="Your Prediction")
    st.plotly_chart(fig, use_container_width=True)

# 4. Input Effect Simulation
with tab4:
    st.write("### 🔮 Predicted Performance vs Different Hours Studied")
    test_range = np.linspace(0, 24, 50)
    varied_features = [[h, previous_scores, extracurricular_val, sleep_hours, sample_papers] for h in test_range]
    varied_preds = model.predict(varied_features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_range, y=varied_preds, mode="lines", name="Predicted Performance", line=dict(color="blue")))
    fig.add_vline(x=hours_studied, line_dash="dash", line_color="orange", annotation_text="Your Input")
    fig.update_layout(title="Effect of Hours Studied on Performance",
                      xaxis_title="Hours Studied", yaxis_title="Predicted Performance Index")
    st.plotly_chart(fig, use_container_width=True)

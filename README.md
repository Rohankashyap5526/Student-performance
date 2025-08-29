# Student Performance Prediction Dashboard

Access the live app here: **[Student Performance Predictor](https://student-performancegit-rwfxcx6njyounmckzkgjft.streamlit.app/)**

##  Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Usage Instructions](#usage)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

##  Overview
This project features a **Streamlit-powered dashboard** that predicts a student's **Performance Index** using a trained **Linear Regression** model. Users can input study-related and lifestyle data to generate instant predictions and insights through **interactive visualizations**.

---

##  Demo
Access the live app here: **[Student Performance Predictor](https://student-performancegit-rwfxcx6njyounmckzkgjft.streamlit.app/)**

---

##  Features
- **Prediction Interface**: Input features such as hours studied, previous scores, sleep hours, extracurricular participation, and practice questions to get a predicted Performance Index.
- **Interactive Visualizations** (powered by Plotly):
  - **Correlation Heatmap** — Understand relationships between all features.
  - **Scatter Plots with Trendlines** — Visualize how hours studied and sleep impact performance, with regression overlays.
  - **Performance Distribution** — See how your prediction compares with the dataset's performance distribution.
  - **Simulation Curve** — Dynamically assess how varying a feature (e.g., hours studied) influences predicted performance.
- **Dynamic Metrics Display**: Immediate visual feedback on user inputs and predicted output.

---

##  Tech Stack
- **Python**: Data manipulation and ML model
- **scikit-learn**: Linear Regression model training
- **joblib**: Model serialization
- **Streamlit**: Web app framework
- **Plotly**: Interactive charting
- **pandas, numpy**: Data preprocessing
- **statsmodels**: (Optional) Required for Plotly’s trendline calculations

---

##  Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pip (or equivalent package manager)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/student-performance-predictor.git
   cd student-performance-predictor

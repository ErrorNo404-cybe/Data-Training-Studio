import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Avash's Data Trainer", layout="wide")
st.title("🚀 Avash's Data Training Studio")
st.markdown("Upload your data and perform analysis & modeling.")

# Sidebar: Upload data
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to begin.")
    st.stop()

# Load data safely
try:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        st.error("Uploaded file is empty.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error reading the CSV file: {e}")
    st.stop()

# Store in session state and use immediately
st.session_state.df = df
df = st.session_state.df  # Optional, but ensures consistency

# Sidebar navigation
st.sidebar.header("🧭 Operations")
operation = st.sidebar.radio("Choose an operation:", [
    "📊 Data Overview",
    "🧹 Data Cleaning",
    "📈 Visualization",
    "🤖 Machine Learning"
])

# =============== 1. Data Overview ===============
if operation == "📊 Data Overview":
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    st.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Column Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.bar_chart(missing)
    else:
        st.success("✅ No missing values!")

# =============== 2. Data Cleaning ===============
elif operation == "🧹 Data Cleaning":
    st.subheader("Data Cleaning Options")
    df_clean = df.copy()

    if st.checkbox("Drop rows with any missing values"):
        df_clean = df_clean.dropna()
        st.success(f"Rows dropped. New shape: {df_clean.shape}")

    cols_to_drop = st.multiselect("Select columns to drop", df_clean.columns.tolist())
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        st.success(f"Dropped columns: {cols_to_drop}")

    st.session_state.df = df_clean
    df = df_clean

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# =============== 3. Visualization ===============
elif operation == "📈 Visualization":
    st.subheader("Data Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for plotting.")
        st.stop()

    plot_type = st.selectbox("Choose plot type", [
        "Scatter Plot",
        "Histogram",
        "Box Plot",
        "Correlation Heatmap"
    ])

    if plot_type == "Scatter Plot" and len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        y = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        color = st.selectbox("Color by (optional)", [None] + categorical_cols, key="scatter_color")
        fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")
        st.plotly_chart(fig)

    elif plot_type == "Histogram":
        col = st.selectbox("Select column", numeric_cols, key="hist_col")
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig)

    elif plot_type == "Box Plot":
        col = st.selectbox("Select column", numeric_cols, key="box_col")
        fig = px.box(df, y=col, title=f"Box Plot of {col}")
        st.plotly_chart(fig)

    elif plot_type == "Correlation Heatmap" and len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Not enough numeric columns for this visualization.")

# =============== 4. Machine Learning ===============
elif operation == "🤖 Machine Learning":
    st.subheader("Auto Machine Learning")

    if len(df) < 10:
        st.error("Need at least 10 rows for modeling.")
        st.stop()
    if len(df.columns) < 2:
        st.error("Need at least one feature and one target column.")
        st.stop()

    target = st.selectbox("Select target variable", df.columns)
    X = df.drop(columns=[target])
    y = df[target]

    if X.empty:
        st.error("No features left after removing target.")
        st.stop()

    # Encode categorical features in X
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include='object').columns:
        X_encoded[col] = X_encoded[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])

    # Determine task type
    if y.dtype == 'object' or y.nunique() <= 20:
        task = "classification"
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
    else:
        task = "regression"
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    if task == "classification":
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)
    else:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    st.subheader("Feature Importance")
    fig = px.bar(importance, x='importance', y='feature', orientation='h')
    st.plotly_chart(fig)
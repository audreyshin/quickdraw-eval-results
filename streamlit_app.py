import streamlit as st
import pandas as pd

st.title("🎨 QuickDraw GPT-4o Evaluation Results")

# Sidebar selector for CSV file versions
csv_file = st.sidebar.selectbox(
    "Choose CSV Version",
    [
        "image_v9_step200.csv",
        "image_v8_step100.csv",
        "stroke_v9_step200.csv"
    ],
    index=0
)

# Load CSV from GitHub directly
csv_url = f"https://raw.githubusercontent.com/audreyshin/quickdraw-eval-results/main/{csv_file}"
df = pd.read_csv(csv_url)

st.subheader(f"Showing results for `{csv_file}`")
st.write(df.head(20))  # Preview first 20 rows

# Basic Metrics
accuracy = df["is_correct"].mean()
st.metric("Overall Accuracy", f"{accuracy:.2%}")

# Category-level accuracy
st.subheader("Category-Level Accuracy")
category_acc = df.groupby("category")["is_correct"].mean().sort_values(ascending=False)
st.bar_chart(category_acc)

# Inspect Incorrect Predictions
st.subheader("Incorrect Predictions")
st.dataframe(df[df["is_correct"] == False].head(10))

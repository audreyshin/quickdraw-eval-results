import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image, ImageDraw

st.set_page_config(layout="wide")
st.title("QuickDraw GPT-4o Evaluation Results")

# Sidebar filters
st.sidebar.header("Filter Options")

# Sidebar selections (updated to match new filenames)
prompt_id = st.sidebar.selectbox("Prompt Variant", ["zero_simple", "zero_cot", "zero_cot_consistent", "fewshot_custom"], index=0)
version = st.sidebar.selectbox("Version", ["v9", "v8", "v7", "v6"], index=0)
input_mode = st.sidebar.selectbox("Input Mode", ["image", "stroke"], index=0)
step = st.sidebar.selectbox("Step", [50, 100, 200], index=2)

# New CSV filename structure (with prompt_id included)
csv_file = f"{prompt_id}_{input_mode}_{version}_step{step}.csv"
csv_url = f"https://raw.githubusercontent.com/audreyshin/quickdraw-eval-results/main/{csv_file}"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

# Load data
try:
    df = load_data(csv_url)
except Exception as e:
    st.error(f"Failed to load CSV file. Check selection: {csv_file}")
    st.stop()

st.subheader(f"Results for `{csv_file}`")
st.write(df.head(10))

# Basic accuracy metrics
accuracy = df["is_correct"].mean()
st.metric("Overall Accuracy", f"{accuracy:.2%}")

# Category-level accuracy with percentage
st.subheader("Category Level Accuracy")
category_acc = (df.groupby("category")["is_correct"].mean() * 100).sort_values(ascending=False)

# Filtering options for categories
top_n = st.slider("Show top/bottom N categories", min_value=5, max_value=20, value=10)
view_option = st.radio("View Categories", ["Top", "Bottom", "Custom"], index=0)

if view_option == "Top":
    category_acc_display = category_acc.head(top_n)
elif view_option == "Bottom":
    category_acc_display = category_acc.tail(top_n)
else:
    selected_categories = st.multiselect("Select categories to view", category_acc.index.tolist())
    category_acc_display = category_acc[category_acc.index.isin(selected_categories)]

fig, ax = plt.subplots(figsize=(12, 6))
category_acc_display.plot.bar(ax=ax, color='skyblue')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.set_xlabel('Category')
st.pyplot(fig)

# Incorrect predictions analysis
st.subheader("Incorrect Predictions")
incorrect_df = df[df["is_correct"] == False]

if incorrect_df.empty:
    st.info("No incorrect predictions found!")
else:
    selected_row = st.selectbox(
        "Select an incorrect prediction to inspect",
        incorrect_df.index,
        format_func=lambda idx: f"{incorrect_df.at[idx, 'category']} (Predicted: {incorrect_df.at[idx, 'prediction']})"
    )
    selected_data = incorrect_df.loc[selected_row]

    st.markdown("### Detailed Analysis")
    st.write(f"**True Label:** {selected_data['category']}")
    st.write(f"**Predicted Label:** {selected_data['prediction']}")
    st.write(f"**Match Reason:** {selected_data['match_reason']}")
    st.write(f"**Country:** {selected_data['countrycode']}")
    st.write(f"**Timestamp:** {selected_data['timestamp']}")

    # Render image from stroke data if available
    if "raw_stroke" in selected_data:
        stroke_data = json.loads(selected_data["raw_stroke"])

        def render_drawing_to_image(drawing, image_size=256, line_width=3):
            img = Image.new("L", (image_size, image_size), color=255)
            draw = ImageDraw.Draw(img)
            for stroke in drawing:
                x, y = stroke
                points = list(zip(x, y))
                draw.line(points, fill=0, width=line_width)
            return img

        img = render_drawing_to_image(stroke_data)
        st.image(img, caption=f"True: {selected_data['category']} | Predicted: {selected_data['prediction']}")
    else:
        st.info("No stroke data available for rendering image.")

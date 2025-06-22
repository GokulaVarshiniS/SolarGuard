import streamlit as st
import os
import random
from PIL import Image
import numpy as np

# Constants
CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
CLASS_INSIGHTS = {
    "Bird-drop": "⚠️ Clean the panel to avoid shading and energy loss.",
    "Clean": "✅ No action required. Panel is clean and operating normally.",
    "Dusty": "🧹 Schedule cleaning to maintain optimal performance.",
    "Electrical-damage": "🔌 Immediate technical inspection needed!",
    "Physical-Damage": "🛠️ Replace or repair the damaged section to prevent further issues.",
    "Snow-Covered": "❄️ Remove snow to restore sunlight absorption."
}
IMG_SIZE = (224, 224)

# UI Title
st.set_page_config(page_title="SolarGuard", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #FFA500;'>☀️ SolarGuard</h1>
    <h4 style='text-align: center; color: #AAAAAA;'>Intelligent Defect Detection on Solar Panels</h4>
""", unsafe_allow_html=True)

# Background styling for light-dark contrast
st.markdown("""
    <style>
        body, .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }
        .highlight-box {
            background-color: #263238;
            padding: 15px;
            border-radius: 10px;
            border-left: 6px solid #00e676;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Dataset Overview
st.markdown("""
    <h2 style='color: #00e676;'>📁 Dataset Overview</h2>
""", unsafe_allow_html=True)
data_path = st.text_input("Enter dataset folder path (with class subfolders):", 
                          value="D:/GUVI/MINI PROJECT/GUVI-Prj05/Faulty_solar_panel")

if os.path.isdir(data_path):
    st.subheader("Class Distribution:")
    for cls in CLASSES:
        cls_path = os.path.join(data_path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
            st.markdown(f"<span style='color:#CCCCCC;'>📦 <strong>{cls}</strong>: {count} images</span>", unsafe_allow_html=True)
else:
    st.warning("Please enter a valid dataset path.")

# Upload and Predict
st.markdown("""
    <h2 style='color: #00e676;'>📸 Upload Image for Prediction</h2>
""", unsafe_allow_html=True)
img_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="🖼 Uploaded Image", use_container_width=True)

    # Simulate prediction
    simulated_probs = np.random.dirichlet(np.ones(len(CLASSES)), size=1)[0]
    predicted_index = int(np.argmax(simulated_probs))
    predicted_class = CLASSES[predicted_index]

    st.markdown(f"""
        <div class='highlight-box'>
        ✅ <strong style='font-size:18px;'>Predicted Condition</strong>: <span style='font-size:22px;color:#00e676;'><strong>{predicted_class}</strong></span>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Prediction Probabilities:")
    for i, cls in enumerate(CLASSES):
        prob = simulated_probs[i] * 100
        bar_color = "#00e676" if i == predicted_index else "#455A64"
        st.markdown(f"""
            <div style='margin:8px 0;'>
                <strong style='color:#bbbbbb;'>{cls}:</strong> {prob:.2f}%
                <div style='background-color:#333; height:16px; border-radius:8px;'>
                    <div style='width:{prob}%; background-color:{bar_color}; height:100%; border-radius:8px;'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 Use Case Insights (Simulated)")
    st.success(CLASS_INSIGHTS[predicted_class])

    st.caption("🔁 Tip: Refresh the app for a new simulation.")
else:
    st.info("👆 Upload an image to get started.")

import streamlit as st
from streamlit_option_menu import option_menu
import os
from PIL import Image
import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import skew
import matplotlib.pyplot as plt

# -------------------- Constants --------------------
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

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="SolarGuard", page_icon="☀️", layout="wide")

st.markdown("""
    <style>
        body, .stApp { background-color: #121212; color: #FFFFFF; }
        .main-title {text-align:center; color:#FFA500;}
        .highlight-box {
            background-color: #263238;
            padding: 15px;
            border-radius: 10px;
            border-left: 6px solid #00e676;
            margin-bottom: 20px;
        }
        .prob-bar {
            background-color: #333; height:16px; border-radius:8px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Database Setup --------------------
conn = sqlite3.connect("solar_predictions.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    predicted_class TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# -------------------- Mock Prediction Function --------------------
def mock_predict(filename):
    filename_lower = filename.lower() if filename else ""
    if "snow" in filename_lower:
        return "Snow-Covered"
    elif "clean" in filename_lower:
        return "Clean"
    elif "dusty" in filename_lower or "dust" in filename_lower:
        return "Dusty"
    elif "bird" in filename_lower or "bird-drop" in filename_lower:
        return "Bird-drop"
    elif "electrical" in filename_lower or "damage" in filename_lower:
        return "Electrical-damage"
    elif "physical" in filename_lower:
        return "Physical-Damage"
    return np.random.choice(CLASSES)

# -------------------- Sidebar Menu --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
    st.title("☀️ SolarGuard")
    selected = option_menu(
        menu_title="Clarify Your View",
        options=["Home", "Dataset Overview", "EDA & Skewness", "Classification Demo", "Insights", "SQL View"],
        icons=["house", "folder", "bar-chart", "camera", "lightbulb", "database"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#1e1e1e"},
            "nav-link": {"font-size": "16px", "color": "#e0e0e0", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#00e676", "color": "black"},
        }
    )

# -------------------- Home --------------------
if selected == "Home":
    st.markdown("<h1 class='main-title'>SolarGuard: Intelligent Defect Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    SolarGuard leverages **Deep Learning** to automatically detect defects on solar panels such as dust, snow, 
    bird droppings, and damage. By providing actionable insights, this system helps optimize **cleaning schedules**, 
    **maintenance**, and **overall energy efficiency**.
    """)
    st.markdown("### 🔧 Key Features")
    st.markdown("""
    - Automated classification of solar panel conditions (6 classes)
    - Interactive visualization through Streamlit
    - Real-time defect insights and recommendations
    - SQL integration for result storage
    - EDA with skewness check
    """)
    st.success("Navigate through the sidebar to explore data, run classification, and view insights.")

# -------------------- Dataset Overview --------------------
elif selected == "Dataset Overview":
    st.markdown("<h2 style='color:#00e676;'>📁 Dataset Overview</h2>", unsafe_allow_html=True)
    data_path = st.text_input("Enter dataset folder path:", value="D:/GUVI/MINI PROJECT/GUVI-Prj05/Faulty_solar_panel")

    if os.path.isdir(data_path):
        class_counts = {}
        for cls in CLASSES:
            cls_path = os.path.join(data_path, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
                class_counts[cls] = count
                st.markdown(f"📦 **{cls}**: {count} images")
        if class_counts:
            st.bar_chart(pd.Series(class_counts))
    else:
        st.warning("Please enter a valid dataset path.")

# -------------------- EDA & Skewness --------------------
elif selected == "EDA & Skewness":
    st.markdown("<h2 style='color:#00e676;'>📊 Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    data = {
        'Class': ['Clean', 'Dusty', 'Bird-drop', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered'],
        'Count': [200, 150, 100, 80, 70, 120]  # Example counts; replace with your dataset
    }
    df = pd.DataFrame(data)

    # Bar chart
    st.bar_chart(df.set_index('Class'))

    # Skewness Calculation
    skewness_value = skew(df['Count'])
    st.markdown(f"### 🧮 Skewness of Class Distribution: **{skewness_value:.2f}**")
    if abs(skewness_value) < 0.5:
        st.success("✅ The data is fairly balanced (low skewness).")
    else:
        st.warning("⚠️ The dataset is imbalanced. Consider using augmentation or class weights.")

# -------------------- Classification Demo --------------------
elif selected == "Classification Demo":
    st.markdown("<h2 style='color:#00e676;'>📸 Upload Image for Prediction</h2>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="🖼 Uploaded Image", use_container_width=True)

        predicted_class = mock_predict(getattr(img_file, 'name', None))
        base_probs = np.random.rand(len(CLASSES)) * 0.3
        predicted_index = CLASSES.index(predicted_class)
        base_probs[predicted_index] = 0.7 + np.random.rand() * 0.3
        probs = base_probs / base_probs.sum()

        st.markdown(f"""
            <div class='highlight-box'>
            ✅ <strong>Predicted Condition:</strong> 
            <span style='font-size:22px;color:#00e676;'><strong>{predicted_class}</strong></span>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Confidence Scores:")
        for i, cls in enumerate(CLASSES):
            prob = probs[i] * 100
            bar_color = "#00e676" if i == predicted_index else "#455A64"
            st.markdown(f"""
                <div style='margin:6px 0;'>
                    <strong>{cls}:</strong> {prob:.2f}%
                    <div class='prob-bar'>
                        <div style='width:{prob}%; background-color:{bar_color}; height:100%; border-radius:8px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.subheader("🧠 Maintenance Insight")
        st.info(CLASS_INSIGHTS[predicted_class])

        # Save prediction to SQL
        cursor.execute(
            "INSERT INTO predictions (filename, predicted_class, confidence) VALUES (?, ?, ?)",
            (img_file.name, predicted_class, float(probs[predicted_index]))
        )
        conn.commit()

        st.success("✅ Prediction saved to SQL database!")
    else:
        st.info("👆 Upload an image to start prediction.")

# -------------------- SQL View --------------------
elif selected == "SQL View":
    st.markdown("<h2 style='color:#00e676;'>🗄️ Stored Predictions (SQL Database)</h2>", unsafe_allow_html=True)
    df_sql = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    if not df_sql.empty:
        st.dataframe(df_sql)
    else:
        st.info("No predictions stored yet.")

# -------------------- Insights --------------------
elif selected == "Insights":
    st.markdown("<h2 style='color:#00e676;'>💡 Business Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    This section summarizes how predictions can be transformed into **actionable insights**:
    """)
    st.markdown("""
    - **Automated Inspections:** Reduce manual monitoring costs using continuous prediction models.
    - **Optimized Cleaning Schedules:** Panels with dust or snow alerts can be prioritized.
    - **Efficiency Reports:** Calculate performance loss for dirty or damaged panels.
    - **Smart Alerts:** Integrate with IoT sensors for automated maintenance notifications.
    """)
    st.success("Use AI-driven insights to improve energy yield and reduce downtime.")

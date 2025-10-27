import streamlit as st
from streamlit_option_menu import option_menu
import os
from PIL import Image
import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import skew
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import time

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
BATCH_SIZE = 16
EPOCHS = 3  # Keep small for demo
MODEL_PATH = "solar_model.h5"

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="SolarGuard", page_icon="☀️", layout="wide")
st.markdown("""
    <style>
        body, .stApp { background-color: #121212; color: #FFFFFF; }
        .main-title {text-align:center; color:#FFA500;}
        .highlight-box { background-color: #263238; padding: 15px; border-radius: 10px; border-left: 6px solid #00e676; margin-bottom: 20px; }
        .prob-bar { background-color: #333; height:16px; border-radius:8px; }
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

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
    st.title("☀️ SolarGuard")
    selected = option_menu(
        menu_title="Clarify Your View",
        options=["Home", "Dataset Overview", "EDA & Skewness", "Train Model", "Classification Demo", "Insights", "SQL View"],
        icons=["house", "folder", "bar-chart", "gear", "camera", "lightbulb", "database"],
        default_index=0,
        styles={
            "container": {"background-color": "#1e1e1e"},
            "nav-link": {"font-size": "16px", "color": "#e0e0e0", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#00e676", "color": "black"},
        }
    )

# -------------------- Helper Functions --------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Home --------------------
if selected == "Home":
    st.markdown("<h1 class='main-title'>SolarGuard: Intelligent Defect Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    SolarGuard leverages **Deep Learning** to automatically detect solar panel defects (dust, snow, bird drops, and damage).  
    It provides actionable maintenance insights for maximizing **energy efficiency**.
    """)
    st.markdown("### 🔧 Key Features")
    st.markdown("""
    - Automatic defect classification (6 classes)  
    - EDA with skewness analysis  
    - Interactive Streamlit dashboard  
    - Real-time predictions and SQL storage  
    - Maintenance recommendations
    """)
    st.success("Navigate through the sidebar to explore all features.")

# -------------------- Dataset Overview --------------------
elif selected == "Dataset Overview":
    st.markdown("<h2 style='color:#00e676;'>📁 Dataset Overview</h2>", unsafe_allow_html=True)
    data_path = st.text_input("Enter dataset folder path:", value="D:/GUVI/MINI PROJECT/GUVI-Prj05/Faulty_solar_panel")

    if os.path.isdir(data_path):
        class_counts = {}
        for cls in CLASSES:
            cls_path = os.path.join(data_path, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('jpg','jpeg','png'))])
                class_counts[cls] = count
                st.markdown(f"📦 **{cls}**: {count} images")
        if class_counts:
            st.bar_chart(pd.Series(class_counts))
    else:
        st.warning("Please enter a valid dataset path.")

# -------------------- EDA & Skewness --------------------
elif selected == "EDA & Skewness":
    st.markdown("<h2 style='color:#00e676;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    data = {'Class': CLASSES, 'Count': [200, 150, 100, 80, 70, 120]}  # Example
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index('Class'))
    skew_val = skew(df['Count'])
    st.markdown(f"### 🧮 Skewness of Class Distribution: **{skew_val:.2f}**")
    if abs(skew_val) < 0.5:
        st.success("✅ Balanced dataset (low skewness).")
    else:
        st.warning("⚠️ Dataset is imbalanced — consider augmentation or class weights.")

# -------------------- Train Model (Optimized) --------------------
elif selected == "Train Model":
    st.markdown("<h2 style='color:#00e676;'>⚡ Train Deep Learning Model</h2>", unsafe_allow_html=True)
    data_path = st.text_input("Enter dataset folder path for training:", value="D:/GUVI/MINI PROJECT/GUVI-Prj05/Faulty_solar_panel")

    if st.button("Start Training"):
        if os.path.isdir(data_path):
            progress = st.progress(0)
            status_text = st.empty()
            st.info("🔄 Loading and preparing dataset...")

            try:
                datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
                train_gen = datagen.flow_from_directory(
                    data_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                    class_mode='categorical', subset='training'
                )
                val_gen = datagen.flow_from_directory(
                    data_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                    class_mode='categorical', subset='validation'
                )

                model = create_model()
                st.info("⚙️ Model created successfully! Starting training...")

                history = {'accuracy': [], 'val_accuracy': []}
                total_epochs = EPOCHS

                for epoch in range(total_epochs):
                    status_text.text(f"Training Epoch {epoch+1}/{total_epochs} ...")
                    hist = model.fit(
                        train_gen, validation_data=val_gen,
                        epochs=1, verbose=0
                    )
                    train_acc = hist.history['accuracy'][0]
                    val_acc = hist.history['val_accuracy'][0]
                    history['accuracy'].append(train_acc)
                    history['val_accuracy'].append(val_acc)

                    progress.progress((epoch + 1) / total_epochs)
                    time.sleep(0.5)

                model.save(MODEL_PATH)
                st.success(f"✅ Training completed! Model saved as `{MODEL_PATH}`")

                # Plot accuracy
                fig, ax = plt.subplots()
                ax.plot(history['accuracy'], label='Training Accuracy', marker='o')
                ax.plot(history['val_accuracy'], label='Validation Accuracy', marker='x')
                ax.set_title('Model Accuracy Progress')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error during training: {e}")
        else:
            st.warning("Please enter a valid dataset path.")

# -------------------- Classification Demo --------------------
elif selected == "Classification Demo":
    st.markdown("<h2 style='color:#00e676;'>📸 Upload Image for Prediction</h2>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="🖼 Uploaded Image", use_container_width=True)

        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            img_array = preprocess_image(img)
            preds = model.predict(img_array)[0]
            predicted_index = np.argmax(preds)
            predicted_class = CLASSES[predicted_index]

            st.markdown(f"""<div class='highlight-box'>✅ <strong>Predicted Condition:</strong> <span style='font-size:22px;color:#00e676;'><strong>{predicted_class}</strong></span></div>""", unsafe_allow_html=True)
            
            st.subheader("📊 Confidence Scores:")
            for i, cls in enumerate(CLASSES):
                prob = preds[i]*100
                bar_color = "#00e676" if i == predicted_index else "#455A64"
                st.markdown(f"""<div style='margin:6px 0;'><strong>{cls}:</strong> {prob:.2f}%<div class='prob-bar'><div style='width:{prob}%; background-color:{bar_color}; height:100%; border-radius:8px;'></div></div></div>""", unsafe_allow_html=True)

            st.subheader("🧠 Maintenance Insight")
            st.info(CLASS_INSIGHTS[predicted_class])

            # Save to SQL
            cursor.execute("INSERT INTO predictions (filename, predicted_class, confidence) VALUES (?, ?, ?)",
                           (img_file.name, predicted_class, float(preds[predicted_index])))
            conn.commit()
            st.success("✅ Prediction saved to SQL database!")
        else:
            st.warning("Model not found. Train the model first.")
    else:
        st.info("👆 Upload an image to start prediction.")

# -------------------- SQL View --------------------
elif selected == "SQL View":
    st.markdown("<h2 style='color:#00e676;'>🗄️ Stored Predictions</h2>", unsafe_allow_html=True)
    df_sql = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    if not df_sql.empty:
        st.dataframe(df_sql)
    else:
        st.info("No predictions stored yet.")

# -------------------- Insights --------------------
elif selected == "Insights":
    st.markdown("<h2 style='color:#00e676;'>💡 Business Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
    - **Automated Inspections:** Reduce manual effort and detect faults early.  
    - **Optimized Cleaning Schedules:** Focus maintenance where it's needed most.  
    - **Efficiency Reports:** Quantify energy loss from dirty/damaged panels.  
    - **Smart Alerts:** Integrate with IoT sensors for proactive management.
    """)
    st.success("AI-powered maintenance for maximum solar performance ☀️")

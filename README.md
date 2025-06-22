☀️ SolarGuard (Simulated Edition)
SolarGuard is a prototype application designed to simulate intelligent defect detection on solar panels. Built using Streamlit, it generates randomized predictions and actionable insights for different solar panel conditions. This version is ideal for demonstrations and interface testing without a trained model.

🖼️ Live Preview
Upload any solar panel image to simulate classification output and receive insights.

🚀 Features
📁 Dataset Overview: Displays class-wise image distribution from your dataset folder.

📸 Image Upload: Upload .jpg, .jpeg, or .png images via UI.

🔮 Simulated Prediction: Uses random probabilities to mimic real model output.

📊 Confidence Visualization: Clean, interactive bar chart for prediction probabilities.

💡 Class Insights: Provides specific maintenance tips for each detected condition.

🎯 Classes Simulated
Class Name	Insight
Clean	✅ No action required. Panel is clean and operating normally.
Dusty	🧹 Schedule cleaning to maintain optimal performance.
Bird-drop	⚠️ Clean the panel to avoid shading and energy loss.
Electrical-damage	🔌 Immediate technical inspection needed!
Physical-Damage	🛠️ Replace or repair the damaged section.
Snow-Covered	❄️ Remove snow to restore sunlight absorption.


bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
🧪 Folder Structure Example
sql
Copy
Edit
Faulty_solar_panel/
├── Clean/
├── Dusty/
├── Bird-drop/
├── Electrical-damage/
├── Physical-Damage/
└── Snow-Covered/
The app reads this structure only to display dataset stats, not for model training.

📸 Simulated Image Prediction
<!-- Optional image link -->

✅ Predicted Class: Dusty

📊 Confidence Score: e.g., 88.42%

💬 Insight: “🧹 Schedule cleaning to maintain optimal performance.”

⚙️ Tech Stack
Python 3.13.1

Streamlit

NumPy

PIL (Pillow)

📝 License
MIT License. See LICENSE for details.

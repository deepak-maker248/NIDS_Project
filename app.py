import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/intrusion_model.pkl"

st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #00c8ff;
}
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #00c8ff;
}
.result-normal {
    background-color: #064e3b;
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 24px;
    font-weight: bold;
}
.result-attack {
    background-color: #7f1d1d;
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ AI-Based Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.write("This Machine Learning project detects whether network traffic is **Normal** or an **Intrusion/Attack** using the NSL-KDD dataset.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load("models/label_encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Accuracy", "99.87%")
k2.metric("Algorithm", "Random Forest")
k3.metric("Dataset", "NSL-KDD")
k4.metric("Project Type", "Cybersecurity ML")

st.sidebar.header("Enter Network Traffic Details")

user_data = {
    "duration": st.sidebar.number_input("Duration", min_value=0, value=0),
    "protocol_type": st.sidebar.selectbox("Protocol Type", ["tcp", "udp", "icmp"]),
    "service": st.sidebar.selectbox("Service", ["http", "ftp_data", "smtp", "domain_u", "eco_i", "private"]),
    "flag": st.sidebar.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "RSTO"]),
    "src_bytes": st.sidebar.number_input("Source Bytes", min_value=0, value=100),
    "dst_bytes": st.sidebar.number_input("Destination Bytes", min_value=0, value=100),
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": st.sidebar.number_input("Count", min_value=0, value=10),
    "srv_count": st.sidebar.number_input("Service Count", min_value=0, value=10),
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 255,
    "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

input_df = pd.DataFrame([user_data])

for col, encoder in label_encoders.items():
    if input_df[col].iloc[0] in encoder.classes_:
        input_df[col] = encoder.transform(input_df[col])
    else:
        input_df[col] = 0

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Network Traffic Data")
    st.dataframe(pd.DataFrame([user_data]), use_container_width=True)

with col2:
    st.subheader("🔍 Prediction Result")

    if st.button("Detect Intrusion", use_container_width=True):
        prediction = model.predict(input_df)[0]
        result = target_encoder.inverse_transform([prediction])[0]

        if result == "normal":
            st.markdown('<div class="result-normal">✅ Normal Traffic Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-attack">🚨 Intrusion / Attack Detected</div>', unsafe_allow_html=True)

        st.info("Prediction completed using trained Random Forest ML model.")

st.markdown("---")
st.subheader("📈 Project Visualizations")

c1, c2 = st.columns(2)

with c1:
    if os.path.exists("images/accuracy_chart.png"):
        st.image("images/accuracy_chart.png", caption="Model Accuracy Graph")
    else:
        st.info("Run train_model.py to generate accuracy graph.")

with c2:
    if os.path.exists("images/confusion_matrix.png"):
        st.image("images/confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.info("Run train_model.py to generate confusion matrix.")

st.markdown("---")

st.subheader("🧠 Project Methodology")
st.write("""
1. Collected NSL-KDD network traffic dataset.
2. Preprocessed categorical and numerical data.
3. Encoded protocol, service, and flag features.
4. Trained Random Forest Classifier.
5. Evaluated model using accuracy and confusion matrix.
6. Built Streamlit dashboard for user interaction.
""")

st.subheader("✅ Conclusion")
st.write("""
This project shows how Machine Learning can be used to detect suspicious network traffic.
The system classifies traffic as normal or attack and provides a simple dashboard for demonstration.
""")

st.markdown("---")
st.write("Developed by **Deepak Panda**")
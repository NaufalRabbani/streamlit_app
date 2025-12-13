import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL & PREPROCESSOR
# =========================
@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return svm_model, scaler, pca

svm, scaler, pca = load_models()

# =========================
# UI STREAMLIT
# =========================
st.set_page_config(
    page_title="Electric Device Detection",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Electric Device Detection")
st.write(
    """
    Aplikasi ini mendeteksi **keberadaan perangkat listrik**
    berdasarkan **histogram intensitas X-ray (256 fitur)**  
    menggunakan **model Support Vector Machine (SVM)**.
    """
)

# =========================
# INPUT DATA
# =========================
st.subheader("📥 Input Histogram")

input_method = st.radio(
    "Pilih metode input:",
    ["Upload File (.csv)", "Input Manual"]
)

if input_method == "Upload File (.csv)":
    uploaded_file = st.file_uploader(
        "Upload file CSV berisi 256 kolom histogram",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = np.loadtxt(uploaded_file, delimiter=",")
        if data.shape[0] != 256:
            st.error("❌ File harus berisi tepat 256 nilai!")
            st.stop()
        input_data = data.reshape(1, -1)

elif input_method == "Input Manual":
    st.write("Masukkan 256 nilai histogram (dipisahkan dengan koma):")
    user_input = st.text_area("Contoh: 0, 1, 5, 10, ...")

    if user_input:
        try:
            values = np.array([float(x) for x in user_input.split(",")])
            if len(values) != 256:
                st.error("❌ Harus tepat 256 nilai!")
                st.stop()
            input_data = values.reshape(1, -1)
        except:
            st.error("❌ Format input tidak valid")
            st.stop()

# =========================
# PREDIKSI
# =========================
if "input_data" in locals():
    if st.button("🔍 Prediksi"):
        # Preprocessing
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)

        # Prediction
        prediction = svm.predict(input_pca)[0]
        probability = svm.predict_proba(input_pca)[0][1]

        # Output
        st.subheader("📊 Hasil Prediksi")

        if prediction == 1:
            st.error("⚠️ PERANGKAT LISTRIK TERDETEKSI")
        else:
            st.success("✅ TIDAK TERDETEKSI PERANGKAT LISTRIK")

        st.metric(
            label="Probabilitas Electric Device",
            value=f"{probability*100:.2f}%"
        )

        # Visualisasi histogram
        st.subheader("📈 Histogram Input")
        st.line_chart(input_data.flatten())

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Model: SVM | Dataset: ElectricDeviceDetection (X-ray 3D Histogram)"
)

import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
import shap
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Electric Device Detection (X-ray)",
    page_icon="⚡",
    layout="centered"
)

# =========================================================
# LOAD MODEL & PREPROCESSING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_models():
    xgb = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return xgb, scaler, pca

xgb, scaler, pca = load_models()

# =========================================================
# LOAD SHAP EXPLAINER
# =========================================================
explainer = shap.TreeExplainer(xgb)

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def is_likely_xray(image):
    img = np.array(image)

    if len(img.shape) == 3:
        std_r = np.std(img[:, :, 0])
        std_g = np.std(img[:, :, 1])
        std_b = np.std(img[:, :, 2])

        if abs(std_r - std_g) > 15 or abs(std_r - std_b) > 15:
            return False

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return np.std(img) <= 80


def image_to_histogram(image):
    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (256, 256))

    # === TAMBAHAN PENTING ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    return hist / hist.sum() if hist.sum() > 0 else hist


def preprocess_histogram(hist):
    hist = hist.reshape(1, -1)
    hist_scaled = scaler.transform(hist)
    hist_pca = pca.transform(hist_scaled)
    return hist_pca


def predict_histogram(hist):
    hist_pca = preprocess_histogram(hist)
    pred = xgb.predict(hist_pca)[0]
    prob = xgb.predict_proba(hist_pca)[0][1]
    return pred, prob, hist_pca


def get_alert_level(prob, threshold):
    if prob < threshold:
        return "LOW", "🟢", "Risiko rendah"
    elif prob < 0.85:
        return "MEDIUM", "🟡", "Perlu pemeriksaan tambahan"
    else:
        return "HIGH", "🔴", "PERINGATAN: Risiko tinggi"


# =========================================================
# UI
# =========================================================
st.title("⚡ Electric Device Detection System")

st.write(
    """
    Sistem ini mendeteksi **perangkat listrik** dari:
    - 📄 **CSV histogram**
    - 🩻 **Gambar X-ray**
    - ⌨️ **Input manual histogram**

    Pipeline:
    **Histogram → StandardScaler → PCA → XGBoost**
    """
)

st.markdown("---")

# =========================================================
# CONFIDENCE THRESHOLD
# =========================================================
st.subheader("⚙️ Confidence Threshold")

threshold = st.slider(
    "Threshold probabilitas deteksi",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05
)

st.markdown("---")

# =========================================================
# INPUT METHOD
# =========================================================
input_method = st.radio(
    "Pilih metode input:",
    [
        "📄 Upload CSV Histogram",
        "🩻 Upload Gambar X-ray",
        "⌨️ Input Manual Histogram"
    ]
)

histogram = None

# =========================================================
# CSV INPUT
# =========================================================
if input_method == "📄 Upload CSV Histogram":
    uploaded_csv = st.file_uploader("Upload CSV (256 nilai)", type=["csv"])

    if uploaded_csv is not None:
        data = np.loadtxt(uploaded_csv, delimiter=",")
        if data.shape[0] != 256:
            st.error("CSV harus berisi 256 nilai")
            st.stop()

        histogram = data / data.sum()
        st.success("Histogram CSV valid")

# =========================================================
# IMAGE INPUT
# =========================================================
elif input_method == "🩻 Upload Gambar X-ray":
    uploaded_image = st.file_uploader("Upload gambar X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).resize((512, 512))
        st.image(image, caption="X-ray Input", use_column_width=True)

        if not is_likely_xray(image):
            st.warning("⚠️ Gambar kemungkinan bukan X-ray")

        histogram = image_to_histogram(image)

# =========================================================
# MANUAL INPUT
# =========================================================
elif input_method == "⌨️ Input Manual Histogram":
    manual_text = st.text_area(
        "Masukkan 256 nilai histogram (dipisahkan koma / spasi)",
        height=200
    )

    if manual_text.strip():
        try:
            values = np.array([float(v) for v in manual_text.replace(",", " ").split()])
            if values.shape[0] != 256:
                st.error("Jumlah nilai harus 256")
                st.stop()

            histogram = values / values.sum()
            st.success("Histogram manual valid")

        except ValueError:
            st.error("Input harus berupa angka")

# =========================================================
# PREDICTION + SHAP
# =========================================================
if histogram is not None:
    if st.button("🔍 Jalankan Prediksi"):
        with st.spinner("Memproses prediksi..."):
            pred, prob, hist_pca = predict_histogram(histogram)
            alert, icon, desc = get_alert_level(prob, threshold)

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if pred == 1 and prob >= threshold:
            st.error(f"{icon} PERANGKAT LISTRIK TERDETEKSI")
        else:
            st.success("✅ TIDAK TERDETEKSI PERANGKAT LISTRIK")

        st.metric("Probabilitas Electric Device", f"{prob*100:.2f}%")
        st.metric("Alert Level", f"{icon} {alert}")
        st.caption(desc)

        st.subheader("📈 Histogram (Normalized)")
        st.line_chart(histogram)

        # =================================================
        # SHAP EXPLANATION
        # =================================================
        st.markdown("---")
        st.subheader("🧠 Penjelasan Model (SHAP)")

        with st.expander("Lihat penjelasan keputusan model"):
            shap_values = explainer.shap_values(hist_pca)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=hist_pca[0],
                    feature_names=[f"PC{i+1}" for i in range(hist_pca.shape[1])]
                ),
                max_display=10,
                show=False
            )
            st.pyplot(fig)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Model: XGBoost | Explainable AI (SHAP) | End-to-End System")
st.caption("Developed by NRSF")

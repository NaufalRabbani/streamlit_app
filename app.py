import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image

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

    if np.std(img) > 80:
        return False

    return True


def image_to_histogram(image):
    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (256, 256))

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def predict_histogram(hist):
    hist = hist.reshape(1, -1)
    hist_scaled = scaler.transform(hist)
    hist_pca = pca.transform(hist_scaled)

    pred = svm.predict(hist_pca)[0]
    prob = svm.predict_proba(hist_pca)[0][1]

    return pred, prob


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
    - 📄 **File CSV histogram (256 fitur)**, atau
    - 🩻 **Gambar X-ray**

    Menggunakan:
    **Histogram → StandardScaler → PCA → SVM**
    """
)

st.markdown("---")

# =========================================================
# CONFIDENCE THRESHOLD
# =========================================================
st.subheader("⚙️ Pengaturan Confidence Threshold")

threshold = st.slider(
    "Threshold probabilitas deteksi perangkat listrik",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05
)

st.caption("Semakin tinggi threshold → semakin ketat deteksi")

st.markdown("---")

# =========================================================
# INPUT METHOD
# =========================================================
input_method = st.radio(
    "Pilih metode input:",
    ["📄 Upload CSV Histogram", "🩻 Upload Gambar X-ray"]
)

histogram = None

# =========================================================
# CSV INPUT
# =========================================================
if input_method == "📄 Upload CSV Histogram":
    uploaded_csv = st.file_uploader(
        "Upload file CSV berisi 256 nilai histogram",
        type=["csv"]
    )

    if uploaded_csv is not None:
        data = np.loadtxt(uploaded_csv, delimiter=",")

        if data.shape[0] != 256:
            st.error("❌ CSV harus berisi tepat 256 nilai!")
            st.stop()

        histogram = data / data.sum() if data.sum() > 0 else data
        st.success("✅ Histogram CSV berhasil dimuat")

# =========================================================
# IMAGE INPUT
# =========================================================
elif input_method == "🩻 Upload Gambar X-ray":
    uploaded_image = st.file_uploader(
        "Upload gambar X-ray (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).resize((512, 512))
        st.image(image, caption="Gambar X-ray", use_column_width=True)

        if not is_likely_xray(image):
            st.warning(
                "⚠️ Gambar kemungkinan **BUKAN citra X-ray**. "
                "Hasil prediksi mungkin tidak valid."
            )

        histogram = image_to_histogram(image)

# =========================================================
# PREDICTION
# =========================================================
if histogram is not None:
    if st.button("🔍 Jalankan Prediksi"):
        with st.spinner("Melakukan prediksi..."):
            pred, prob = predict_histogram(histogram)
            alert, icon, desc = get_alert_level(prob, threshold)

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if pred == 1 and prob >= threshold:
            st.error(f"{icon} **PERANGKAT LISTRIK TERDETEKSI**")
        else:
            st.success("✅ **TIDAK TERDETEKSI PERANGKAT LISTRIK**")

        st.metric("Probabilitas Electric Device", f"{prob*100:.2f}%")
        st.metric("Alert Level", f"{icon} {alert}")

        st.caption(desc)

        st.subheader("📈 Histogram Intensitas (Normalized)")
        st.line_chart(histogram)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Model: SVM | Histogram Intensitas X-ray | End-to-End Detection System"
)
st.caption("Developed by NRSF")
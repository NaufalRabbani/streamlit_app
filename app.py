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
# LOAD MODEL & PREPROCESSING (CACHE)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_models():
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    return svm, scaler, pca

svm, scaler, pca = load_models()

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def is_likely_xray(image):
    """
    Heuristic ringan untuk memberi peringatan
    jika gambar kemungkinan bukan X-ray
    """
    img = np.array(image)

    # Jika RGB, cek variasi channel warna
    if len(img.shape) == 3:
        std_r = np.std(img[:, :, 0])
        std_g = np.std(img[:, :, 1])
        std_b = np.std(img[:, :, 2])

        # Perbedaan channel besar → kemungkinan foto biasa
        if abs(std_r - std_g) > 15 or abs(std_r - std_b) > 15:
            return False

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # X-ray biasanya kontras sedang
    if np.std(img) > 80:
        return False

    return True


def image_to_histogram(image):
    """
    Konversi gambar → histogram 256 bin + auto-normalization
    """
    img = np.array(image)

    # RGB → Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize agar konsisten & cepat
    img = cv2.resize(img, (256, 256))

    # Histogram 256 bin
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # AUTO-NORMALIZATION (L1)
    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist


def predict_histogram(hist):
    """
    Pipeline prediksi lengkap
    """
    hist = hist.reshape(1, -1)
    hist_scaled = scaler.transform(hist)
    hist_pca = pca.transform(hist_scaled)

    pred = svm.predict(hist_pca)[0]
    prob = svm.predict_proba(hist_pca)[0][1]

    return pred, prob


# =========================================================
# UI
# =========================================================
st.title("⚡ Electric Device Detection from X-ray Image")

st.write(
    """
    Aplikasi ini mendeteksi **keberadaan perangkat listrik**
    dari **citra X-ray** menggunakan:
    - Histogram intensitas (256 bin)
    - StandardScaler + PCA
    - **Support Vector Machine (SVM)**

    Sistem ini merupakan **simulasi end-to-end**
    dari proses deteksi perangkat listrik berbasis X-ray.
    """
)

st.markdown("---")

# =========================================================
# INPUT IMAGE
# =========================================================
uploaded_image = st.file_uploader(
    "📤 Upload gambar X-ray (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Resize awal agar ringan di Cloud
    image = image.resize((512, 512))

    st.image(image, caption="Gambar X-ray yang diunggah", use_column_width=True)

    # Validasi X-ray
    if not is_likely_xray(image):
        st.warning(
            "⚠️ Gambar yang diunggah kemungkinan **BUKAN citra X-ray**. "
            "Hasil prediksi mungkin tidak valid."
        )

    # Konversi ke histogram
    histogram = image_to_histogram(image)

    if st.button("🔍 Prediksi"):
        with st.spinner("Memproses dan melakukan prediksi..."):
            prediction, probability = predict_histogram(histogram)

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if prediction == 1:
            st.error("⚠️ **PERANGKAT LISTRIK TERDETEKSI**")
        else:
            st.success("✅ **TIDAK TERDETEKSI PERANGKAT LISTRIK**")

        st.metric(
            label="Probabilitas Electric Device",
            value=f"{probability * 100:.2f}%"
        )

        # Visualisasi histogram
        st.subheader("📈 Histogram Intensitas (Normalized)")
        st.line_chart(histogram)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Model: SVM | Representasi: Histogram Intensitas X-ray | Deployment: Streamlit"
)
st.caption("Developed by NRSF")
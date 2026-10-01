import streamlit as st
import joblib


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model/fake_news_model.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">📰 Fake News Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered news classification using Machine Learning</div>',
    unsafe_allow_html=True
)


st.divider()


# -----------------------------
# News input
# -----------------------------
news = st.text_area(
    "📝 Enter News Article",
    placeholder="Paste the news article here...",
    height=250
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check News", type="primary", use_container_width=True):

    if not news.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        # Prediction
        prediction = model.predict([news])[0]

        # Probability
        probabilities = model.predict_proba([news])[0]

        confidence = max(probabilities) * 100


        # -----------------------------
        # Result
        # -----------------------------
        st.subheader("Prediction Result")


        if prediction == 0:

            st.error("🚨 FAKE NEWS")

            st.write(
                "The model predicts that this news resembles "
                "patterns found in fake news articles."
            )

        else:

            st.success("✅ REAL NEWS")

            st.write(
                "The model predicts that this news resembles "
                "patterns found in real news articles."
            )


        # -----------------------------
        # Confidence
        # -----------------------------
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            confidence / 100,
            text=f"Confidence: {confidence:.2f}%"
        )


st.divider()


# -----------------------------
# About project
# -----------------------------
with st.expander("ℹ️ About this project"):

    st.write("""
    This project uses Natural Language Processing and Machine Learning
    to classify news articles as Fake or Real.

    **Technologies used:**

    • Python  
    • Pandas  
    • Scikit-learn  
    • TF-IDF  
    • Logistic Regression  
    • Streamlit  

    The model was trained on a dataset containing fake and real news
    articles.
    """)


st.caption(
    "⚠️ This tool is a machine-learning classifier and does not independently verify facts."
)
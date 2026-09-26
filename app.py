import joblib
import streamlit as st

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

st.title("🕵️ Fake News Detector")
st.write(
    "Enter a news headline or short news snippet and this app will predict "
    "whether it's **Real** or **Fake**, using a TF-IDF + Multinomial Naive "
    "Bayes model."
)


# ---------------------------------------------------------------
# Load the trained vectorizer and model (cached so it only loads once)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    return vectorizer, model


vectorizer, model = load_artifacts()

# ---------------------------------------------------------------
# User input
# ---------------------------------------------------------------
news_text = st.text_area(
    "News text",
    placeholder="e.g. Scientists discover a new planet similar to Earth",
    height=120,
)

if st.button("Check News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        # Transform input text using the saved TF-IDF vectorizer
        vector = vectorizer.transform([news_text])
        prediction = model.predict(vector)[0]

        if prediction == "Real":
            st.success(f"**Prediction:** {prediction} ✅")
        else:
            st.error(f"**Prediction:** {prediction} ❌")

        # Show prediction probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            st.subheader("Prediction Confidence")
            prob_data = {
                cls: float(p) for cls, p in zip(model.classes_, probs)
            }
            st.bar_chart(prob_data)

st.divider()
st.caption(
    "Model: TfidfVectorizer + MultinomialNB, trained on a small sample "
    "news-text dataset. For demonstration purposes only — not a substitute "
    "for professional fact-checking."
)

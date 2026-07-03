import streamlit as st
import pickle
import nltk
import string
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="MessageGuard",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------

nltk.download("punkt")
nltk.download("stopwords")

# -----------------------------
# LOAD MODEL
# -----------------------------

tfidf = pickle.load(open("vectorizer_new.pkl", "rb"))
model = pickle.load(open("model_new.pkl", "rb"))

ps = PorterStemmer()

# -----------------------------
# PREPROCESSING
# -----------------------------

def transform_text(text):

    text = text.lower()

    words = nltk.word_tokenize(text)

    words = [word for word in words if word.isalnum()]

    words = [
        word
        for word in words
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]

    words = [ps.stem(word) for word in words]

    return " ".join(words)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background:linear-gradient(135deg,#0f172a,#1e293b,#111827);
}

/* Hide Streamlit header */

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#111827;
}

/* Hero */

.hero{

padding:30px;

border-radius:18px;

background:rgba(255,255,255,.05);

border:1px solid rgba(255,255,255,.08);

text-align:center;

margin-bottom:25px;

}

.hero h1{

color:white;

font-size:46px;

margin-bottom:5px;

}

.hero p{

color:#cbd5e1;

font-size:18px;

}

/* Button */

div.stButton > button{

width:100%;

height:55px;

border-radius:12px;

border:none;

background:#6366F1;

color:white;

font-size:18px;

font-weight:600;

transition:.3s;

}

div.stButton > button:hover{

background:#4F46E5;

transform:translateY(-2px);

}

/* Result Card */

.result{

padding:20px;

border-radius:15px;

margin-top:20px;

}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("🛡️ MessageGuard")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
Detect spam SMS and Email messages using
Machine Learning and Natural Language Processing.
"""
    )

    st.markdown("---")

    st.subheader("Technology")

    st.write("🐍 Python")

    st.write("⚡ Streamlit")

    st.write("🧠 Scikit-learn")

    st.write("📚 NLTK")

    st.markdown("---")

    st.subheader("Model")

    st.success("Multinomial Naive Bayes")

    st.markdown("---")

    st.caption("Developed by Anshuman Parida")

# -----------------------------
# HERO SECTION
# -----------------------------

st.markdown("""

<div class="hero">

<h1>🛡️ MessageGuard</h1>

<p>
Smart SMS & Email Spam Detection
</p>

</div>

""", unsafe_allow_html=True)

st.markdown("### ✉️ Enter your message")

message = st.text_area(

    "",

    placeholder="Type or paste your SMS or Email here...",

    height=220

)

analyze = st.button("Analyze Message")
# -----------------------------
# PREDICTION
# -----------------------------

if analyze:

    if message.strip() == "":

        st.warning("⚠️ Please enter a message to analyze.")

    else:

        with st.spinner("Analyzing message..."):

            processed_message = transform_text(message)

            vector = tfidf.transform([processed_message])

            prediction = model.predict(vector)[0]

            confidence = None

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(vector).max()

        st.markdown("---")

        # -----------------------------
        # RESULT
        # -----------------------------

        if prediction == 1:

            st.error("🚨 Spam Detected")

            st.markdown("""
This message contains characteristics commonly associated with spam.

**Recommendation**

- Do not click unknown links.
- Avoid sharing personal information.
- Verify the sender before responding.
""")

        else:

            st.success("✅ Safe Message")

            st.markdown("""
This message appears to be genuine and does not show common spam characteristics.
""")

        # -----------------------------
        # CONFIDENCE
        # -----------------------------

        if confidence is not None:

            st.markdown("### 🎯 Prediction Confidence")

            st.progress(float(confidence))

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.markdown("---")

        # -----------------------------
        # MESSAGE STATISTICS
        # -----------------------------

        st.markdown("## 📊 Message Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Characters",
                len(message)
            )

        with col2:
            st.metric(
                "Words",
                len(message.split())
            )

        with col3:
            st.metric(
                "Tokens",
                len(processed_message.split())
            )

        # -----------------------------
        # DETECTED KEYWORDS
        # -----------------------------

        spam_keywords = [
            "free",
            "winner",
            "win",
            "claim",
            "cash",
            "offer",
            "urgent",
            "click",
            "bonus",
            "limited",
            "prize",
            "call",
            "gift"
        ]

        found = []

        for word in message.lower().split():

            cleaned = word.strip(".,!?()[]{}")

            if cleaned in spam_keywords:
                found.append(cleaned)

        if found:

            st.markdown("---")

            st.warning(
                "⚠️ Suspicious keywords detected: "
                + ", ".join(sorted(set(found)))
            )

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;font-size:14px;'>

🛡️ <b>MessageGuard</b><br>

Smart SMS & Email Spam Detection<br><br>

Built with ❤️ using Python • Streamlit • Scikit-learn • NLTK

</div>
""",
unsafe_allow_html=True
)
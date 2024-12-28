import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pickle

# Initialize IndoBERT model and tokenizer
@st.cache_resource
def load_indobert_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

# Function to load the uploaded model
def load_custom_model(uploaded_file):
    try:
        model = pickle.load(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Function to classify polysemy using IndoBERT
def classify_with_indobert(word, context_sentences, indobert_classifier):
    """
    Use IndoBERT to classify whether the word is polysemous.
    This is a simplified example using text classification.
    """
    context = " ".join(context_sentences)
    prediction = indobert_classifier(context)[0]  # Simplify to binary classification logic
    nilai = 1 - prediction['score']
    return f"'{word}' dikategorikan sebagai {'polysemous' if prediction['label'] == 'LABEL_1' else 'polysemi'} dengan nilai confidence {nilai:.2f}."

# Function to classify polysemy using a custom model
def classify_with_custom_model(word, context_sentences, model):
    """
    Use the uploaded custom model to classify polysemy.
    Adjust this based on the custom model's logic.
    """
    contexts = [sentence for sentence in context_sentences if word in sentence]
    if len(contexts) > 1:
        # Example: simulate prediction using the custom model
        score = model.predict([word])  # Adjust as per your custom model
        return f"'{word}' is polysemous with a confidence score of {score}."
    return f"'{word}' does not appear to be polysemous."

# Streamlit app
st.title("Sistem Klasifikasi Bahasa Polisemi - M.Yamin")

st.write("""
Aplikasi ini mengidentifikasi apakah suatu kata dalam bahasa Indonesia bersifat polisemi (yaitu, memiliki banyak arti dalam konteks yang berbeda). 
Anda dapat memilih untuk menggunakan model IndoBERT atau mengunggah model kustom Anda sendiri.
""")

# Model selection
model_option = st.radio(
    "Pilih Model yang akan digunakan:",
    ("IndoBERT", "Upload Model")
)

if model_option == "Upload Model":
    uploaded_file = st.file_uploader("Upload Model anda Sendiri (e.g., .pkl):", type=["pkl"])
    if uploaded_file:
        model = load_custom_model(uploaded_file)
        if model:
            st.success("Custom model Berhasil Diupload!")
        else:
            st.error("Gagal memuat custom model.")
else:
    st.info("Menggunakan IndoBERT model untuk klasifikasi.")
    indobert_classifier = load_indobert_model()

# Input section
word = st.text_input("Masukan kata:")
context_sentences = st.text_area("Masukkan Kalimat (Pisahkan kalimat dengan enter):")

if st.button("Analyze"):
    if not word or not context_sentences:
        st.error("Tolong berikan kata dan kalimat konteksnya.")
    else:
        sentences = [sentence.strip() for sentence in context_sentences.split('\n') if sentence.strip()]
        
        if model_option == "IndoBERT":
            result = classify_with_indobert(word, sentences, indobert_classifier)
        elif uploaded_file and model:
            result = classify_with_custom_model(word, sentences, model)
        else:
            st.error("Tidak ditemukan model yang valid. Harap unggah model khusus atau pilih IndoBERT.")
            result = None
        
        if result:
            st.success(result)

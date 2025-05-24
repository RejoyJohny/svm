
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Emotion labels 
label_map = {
    0: "Neutral",
    1: "Worried",
    2: "Romantic",
    3: "Happy",
    4: "Sad",
    5: "Confused"
}

st.title("Tweet Emotion Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        st.success(f"Predicted Emotion: **{label_map.get(prediction, 'Unknown')}**")

import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
  model = pipeline("zero-shot-classification")
  return model

cl = load_model()
image = Image.open('header.png')
st.image(image)
st.title("Zero Shot Classification to Support Emotional Intelligence in a Journaling Context")
sequences = st.text_area('Write a short letter to your future self here:', height=30)
button = st.button("Done")
with st.spinner("Empathy Engine Running..."):
    if button and sequences:
        mod = cl(sequences=sequences, candidate_labels=["encouraging", "cautious", "positive", "negative", "upbeat", "thankful", "forward-looking", "neutral", "sad"])
        first = mod["labels"][0]
        second = mod["labels"][1]
        output = f"Your letter seems quite **{first}** and **{second}**. What do you think?"
        st.markdown(output)
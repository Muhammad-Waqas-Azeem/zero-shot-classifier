import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Zero-Shot Classifier")
st.title("Muhammad Harmain Ansari | B04 0923 000006")
st.subheader("Zero-Shot Text Classification (facebook/bart-large-mnli)")

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

clf = load_model()

text = st.text_area("Enter text to classify:", height=120)

st.write("Enter exactly 3 candidate labels:")
c1, c2, c3 = st.columns(3)
l1 = c1.text_input("Label 1")
l2 = c2.text_input("Label 2")
l3 = c3.text_input("Label 3")

if st.button("Classify"):
    labels = [l.strip() for l in (l1, l2, l3) if l.strip()]
    if not text.strip():
        st.warning("Please enter some text.")
    elif len(labels) != 3:
        st.warning("Please enter all 3 candidate labels.")
    elif len(set(l.lower() for l in labels)) != 3:
        st.warning("Labels must be different from each other.")
    else:
        with st.spinner("Classifying..."):
            result = clf(text, candidate_labels=labels)
        st.success(f"Top label: {result['labels'][0]} ({result['scores'][0]:.2%})")
        if result["scores"][0] < 0.5:
            st.warning("Low confidence: the result may be unreliable.")
        for label, score in zip(result["labels"], result["scores"]):
            st.write(f"**{label}**: {score:.2%}")
            st.progress(float(score))

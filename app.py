import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load pre-trained model and tokenizer from Hugging Face
@st.cache_resource  # Cache the model loading to improve performance
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

tokenizer, model = load_model_and_tokenizer()

# Define labels (matches toxic-bert's output)
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def classify_comment(comment):
    # Tokenize input
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert logits to probabilities using sigmoid
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs > 0.7).astype(int)  # Same threshold as your original
    
    # Debug output
    print("Comment:", comment)
    print("Probabilities:", dict(zip(labels, probs)))
    print("Binary Predictions (threshold=0.7):", dict(zip(labels, preds)))
    
    return dict(zip(labels, probs))

# Streamlit UI (same design as your original)
st.title("🛡️ Toxic Comment Classifier")
st.write("Enter a comment below to check if it's toxic or not.")

user_input = st.text_area("💬 Enter your comment:")

if st.button("Classify 🔍"):
    if user_input.strip():
        try:
            result = classify_comment(user_input)
            
            st.subheader("Toxicity Breakdown:")
            for label, prob in result.items():
                st.write(f"{label.replace('_', ' ').title()}: {prob:.2%}")
            
            if any(prob > 0.7 for prob in result.values()):
                st.error("⚠️ This comment contains potentially toxic content!")
            else:
                st.success("✅ This comment appears to be non-toxic.")
        except Exception as e:
            st.error(f"Error processing comment: {str(e)}")
    else:
        st.warning("⚠️ Please enter a comment before classifying.")
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.set_page_config(page_title="SummarizeSense", page_icon="✨")

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

st.title("✨ SummarizeSense")
st.write("Convert long paragraphs into **clear & concise summaries.**")

text = st.text_area("Paste text here:", height=200)

if st.button("Summarize"):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.subheader("✅ Summary:")
    st.write(summary)

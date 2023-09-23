import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)


# Streamlit App
def main():
    st.title("Quick Text Summarization app")
    text = st.text_area("Enter Text Here", value="")

    if st.button("Summarize"):
        if text:
            st.subheader("Summary")
            # initialize the tokenizer model:
            inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=50512, truncation=True)

            # generate the summary by using the model.generate function on T5:
            summary_ids = model.generate(inputs, max_length=35550, min_length=580, length_penalty=5., num_beams=2)

            # decode the tokenized summary using the tokenizer.decode function:
            summary = tokenizer.decode(summary_ids[0])
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")


if __name__ == "__main__":
    main()

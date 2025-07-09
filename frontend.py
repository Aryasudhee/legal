from rag_pipeline import answer_query,retrieve_docs,llm


import streamlit as st

# Step 1: Setup Upload PDF functionality
st.set_page_config(page_title="PDF Uploader", layout="centered")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")


ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)

    # RAG Pipeline
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs, model=llm, query=user_query)

        st.chat_message("AI Lawyer").write(response)  # This sends the response to the UI

    else:
        st.error("Kindly upload a valid PDF file first!")


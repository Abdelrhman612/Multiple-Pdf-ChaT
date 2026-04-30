import streamlit as st
from service.chat_service import process_question, process_pdfs


def run_ui():

    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("Chat with Multiple PDFs 📚")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    user_question = st.text_input("Ask a question")

    # --- Ask question ---
    if user_question and st.session_state.vectorstore:
        response = process_question(
            st.session_state.vectorstore,
            user_question
        )
        st.write(response)

    # --- Sidebar ---
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                st.session_state.vectorstore = process_pdfs(pdf_docs)
                st.success("Ready!")
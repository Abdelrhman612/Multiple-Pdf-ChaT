import streamlit as st

from utils.pdf_loader import  get_pdf_text
from utils.text_splitter import get_text_chunks
from vectorstore.vectorstore import get_vector_store
from llm.llm import generate_answer
from rag.rag_pipeline import get_answer


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("Chat with Multiple PDFs 📚")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    

    user_question = st.text_input("Ask a question")

    if user_question and st.session_state.vectorstore:
        response = get_answer(
            st.session_state.vectorstore,
            user_question,
            generate_answer
        )
        st.write(response)

    

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)

                st.session_state.vectorstore = get_vector_store(chunks)

                st.success("Ready!")


if __name__ == "__main__":
    main()
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS


# --- PDF → Text ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# --- Text → Chunks ---
def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(raw_text)


# --- Vector Store ---
def get_vector_store(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_texts(text_chunks, embedding=embeddings)


# --- LLM ---
llm = OllamaLLM(model="tinyllama")


# --- App ---
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("Chat with Multiple PDFs 📚")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    user_question = st.text_input("Ask a question")

    # --- QA FLOW ---
    if user_question and st.session_state.vectorstore:

        docs = st.session_state.vectorstore.similarity_search(user_question, k=3)

        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.

Context:
{context}

Question:
{user_question}
"""

        response = llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(response)


    # --- Sidebar ---
    with st.sidebar:
        st.header("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):

                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)

                    st.session_state.vectorstore = get_vector_store(chunks)

                st.success("Ready 🚀")


if __name__ == "__main__":
    main()
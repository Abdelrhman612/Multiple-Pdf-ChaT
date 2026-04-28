import  streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(raw_text):
   text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
     length_function=len
   )
   chunks = text_splitter.split_text(raw_text)
   return chunks

def get_vector_store(text_chunk):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(text_chunk , embedding=embeddings)
    return vector_store

    


def main():
    st.set_page_config(page_title="Chat with Multiple PDFS", page_icon=":books:")
    st.header("Chat with Multiple PDFS  :books:")
    st.text_input("Ask a question about your document")
    with st.sidebar:
        st.sidebar.header("your document")
        pdf_docs = st.file_uploader("Upload your pdfs here and click on 'Process'" , accept_multiple_files=True)
        if st.button("Process"):
         if pdf_docs:
          with st.spinner("Processing..."):

           raw_text = get_pdf_text(pdf_docs)
           text_chunk = get_text_chunks(raw_text)
           vector_store =  get_vector_store(text_chunk)
           
           






if __name__ == '__main__':
    main()
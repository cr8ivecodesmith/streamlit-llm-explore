import pickle
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from utils import get_openai_api_key


OPENAI_API_KEY = get_openai_api_key()


def sidebar():
    # Sidebar contents
    with st.sidebar:
        st.title('🤖LLM Doc Chat')
        st.markdown("""
        ## About
        An exploration of an LLM-powered chatbot for your documents using:
        - [Streamlit](#)
        - [LangChain](#)
        - [OpenAI](#) LLM model
        """)
        add_vertical_space(5)
        st.write('Made with 🫶 by [cr8ivecodesmith](#)')


def get_document(uploaded_file):
    """Returns the text document of the uploaded file

    TODO:
    - Look into langchain.document_loaders

    """
    doc = ''
    doc_file = Path(uploaded_file.name)
    metadata = {
        'name': doc_file.stem,
        'extension': doc_file.suffix,
        'type': uploaded_file.type,
        'size': uploaded_file.size,
    }

    if 'pdf' in metadata.get('type', ''):
        reader = PdfReader(uploaded_file)
        doc = '\n'.join([i.extract_text() for i in reader.pages])
    else:
        doc = uploaded_file.read().decode()

    return doc, metadata


def get_chunks(doc, chunk_size=1024, chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text=doc)
    return chunks


def get_vectorstore(chunks, metadata):
    """Compute the embeddings using OpenAIEmbeddings stored as
    FAISS vectorstores

    NOTE:
    - This will incur OpenAI API Usage charges
    - We only want to compute the embeddings once on a given file

    TODO:
    - Use the whole metadata to create a unique filename for the
      file.
    - Store the pickle file in another directory

    """
    store_name = f'{metadata.get("name")}.pkl'
    if Path(store_name).exists():
        with open(store_name, 'rb') as fh:
            vstore = pickle.load(fh)
        st.write(f'Embeddings for "{store_name}" loaded from disk.')
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vstore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(store_name, 'wb') as fh:
            pickle.dump(vstore, fh)
        st.write(f'Embeddings for "{store_name}" saved to disk.')

    return vstore


def query_document(query, vectorstore):
    """Use OpenAI to query the document

    TODO:
    - Try using the load_qa_with_sources
    - Learn about the different chain_type

    NOTES:
    - Increase the `k` similarity_search parameter to provide more
      context to ChatGPT

    """

    docs = vectorstore.similarity_search(query=query, k=5)
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print(f'query_document query - {query}')
        print(f'query_document callback - {cb}')

    return response


def main():
    sidebar()
    st.header("Chat with Document💬")

    uploaded_file = st.file_uploader(
        'Upload your document',
        type=('txt', 'md', 'pdf'),
        accept_multiple_files=False,
    )
    if not uploaded_file:
        return

    doc, doc_meta = get_document(uploaded_file)
    chunks = get_chunks(doc)
    vectorstore = get_vectorstore(chunks, metadata=doc_meta)

    query = st.text_input(
        'Ask something about the document',
        placeholder='Can you highlight the keypoints in bullets?',
        disabled=not uploaded_file,
    )
    if query and vectorstore:
        query += '. Use markdown formatting.'
        res = query_document(query, vectorstore)
        st.markdown(res)


if __name__ == '__main__':
    main()

import pickle

import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from utils import (
    DB_PATH,
    get_openai_api_key,
    make_tempfile,
    compute_md5_checksum,
)


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

    temp = make_tempfile(uploaded_file)
    checksum = compute_md5_checksum(temp)
    temp.unlink()

    if 'pdf' in uploaded_file.type:
        reader = PdfReader(uploaded_file)
        doc = '\n'.join([i.extract_text() for i in reader.pages])
    else:
        doc = uploaded_file.read().decode()

    return doc, checksum


def get_documents(uploaded_files):

    docs, checksums = [], []

    for file_ in uploaded_files:
        doc, checksum = get_document(file_)
        docs.append(doc)
        checksums.append(checksum)

    checksums.sort()

    return docs, checksums


def get_chunks(doc, chunk_size=1024, chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text=doc)
    return chunks


def get_chunks_list(docs, chunk_size=1024, chunk_overlap=256):
    chunks_list = []
    for doc in docs:
        chunks_list.extend(get_chunks(doc, chunk_size, chunk_overlap))
    return chunks_list


def get_vectorstore(chunks, checksum):
    """Compute the embeddings using OpenAIEmbeddings stored as
    FAISS vectorstores

    NOTE:
    - This will incur OpenAI API Usage charges
    - We only want to compute the embeddings once on a given file or batch

    """
    if isinstance(checksum, list):
        # This is a multi-file batch
        store_name = '-'.join([i for i in checksum]) + '.pkl'
    else:
        store_name = f'{checksum}.pkl'

    store_name = DB_PATH.joinpath(store_name)
    if store_name.exists():
        with store_name.open(mode='rb') as fh:
            vstore = pickle.load(fh)
        st.write(f'Embeddings for "{store_name}" loaded from disk.')
    else:
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vstore = FAISS.from_texts(chunks, embedding=embeddings)
            with store_name.open(mode='wb') as fh:
                pickle.dump(vstore, fh)
            st.write(f'Embeddings for "{store_name}" saved to disk.')
            print(f'get_vectorstore embeddings callback - {cb}')

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
        accept_multiple_files=True,
    )
    if not uploaded_file:
        return

    docs, checksums = get_documents(uploaded_file)
    chunks = get_chunks_list(docs)
    vectorstore = get_vectorstore(chunks, checksum=checksums)

    query = st.text_input(
        'Ask something about the document',
        placeholder='Can you highlight the keypoints in bullets?',
        disabled=not uploaded_file,
    )
    if query and vectorstore:
        query += '. Format the answer in markdown.'
        res = query_document(query, vectorstore)
        st.markdown(res)


if __name__ == '__main__':
    main()

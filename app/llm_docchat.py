import json
import pickle

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from utils import (
    DB_PATH,
    HISTORY_PATH,
    get_openai_api_key,
    make_tempfile,
    compute_md5_checksum,
)


OPENAI_API_KEY = get_openai_api_key()

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the
follow up question to be a standalone question.
You can assume the question to be related to the documents.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:
""")

QA_PROMPT = PromptTemplate(template="""
You are an AI assistant for answering questions about a provided set of
documents.
You are given the following extracted parts of a long document and a question.
Provide a conversational answer. If you don't know the answer, just say
"I do not know the answer.". Don't try to make up an answer.

Question: {question}
==========
{context}
==========
Answer in Markdown:
""", input_variables=['question', 'context'])

DOC_PROMPT = PromptTemplate(
    template='Content: {page_content}\nSource: {source}',
    input_variables=['page_content', 'source'],
)


class ChainManager:

    def __init__(self, chain, store_name):
        self._memory_store_name = store_name
        self.chain = chain

    def save_memory(self):
        with self._memory_store_name.open(mode='wb') as fh:
            pickle.dump(self.chain.memory, fh)


def sidebar():
    # Sidebar contents
    with st.sidebar:
        st.title('🤖DocGPT Chat')
        st.write('Made with 🫶 by [cr8ivecodesmith](https://github.com/cr8ivecodesmith/streamlit-llm-explore)')


def get_document(uploaded_file):
    """Returns the text document of the uploaded file

    TODO:
    - Look into langchain.document_loaders

    """
    temp = make_tempfile(uploaded_file)
    checksum = compute_md5_checksum(temp)
    file_name = uploaded_file.name.strip()

    if 'pdf' in uploaded_file.type:
        loader = UnstructuredPDFLoader(temp)
    elif 'markdown' in uploaded_file.type:
        loader = UnstructuredMarkdownLoader(temp)
    else:
        loader = TextLoader(temp, encoding='utf-8')

    doc = loader.load()
    temp.unlink()

    return doc, file_name, checksum


def get_documents(uploaded_files):

    docs, checksums = [], []

    for file_ in uploaded_files:
        doc, file_name, checksum = get_document(file_)
        docs.append((doc, file_name))
        checksums.append(checksum)

    checksums.sort()

    return docs, checksums


def get_chunks(doc, file_name, chunk_size=1024, chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(doc)
    for i, chunk in enumerate(chunks):
        chunk.metadata['source'] = f'{file_name}: {i}-pl'
    return chunks


def get_chunks_list(docs, chunk_size=1024, chunk_overlap=256):
    chunks_list = []
    for (doc, file_name) in docs:
        chunks_list.extend(get_chunks(
            doc, file_name, chunk_size, chunk_overlap
        ))
    return chunks_list


def get_storage_name(checksum):
    if isinstance(checksum, list):
        # This is a multi-file batch
        store_name = '-'.join([i for i in checksum])
    else:
        store_name = f'{checksum}'
    return store_name


def get_vectorstore(chunks, checksum):
    """Compute the embeddings using OpenAIEmbeddings stored as
    FAISS vectorstores

    NOTE:
    - This will incur OpenAI API Usage charges
    - We only want to compute the embeddings once on a given file or batch

    """
    store_name = DB_PATH.joinpath(get_storage_name(checksum) + '.pkl')
    if store_name.exists():
        with store_name.open(mode='rb') as fh:
            vstore = pickle.load(fh)
        print(f'Embeddings for "{store_name}" loaded from disk.')
    else:
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vstore = FAISS.from_documents(chunks, embedding=embeddings)
            with store_name.open(mode='wb') as fh:
                pickle.dump(vstore, fh)
            print(f'Embeddings for "{store_name}" saved to disk.')
            print(f'get_vectorstore embeddings callback - {cb}')

    return vstore


def get_chain(vectorstore, checksum):
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo-16k',
        temperature=0.7,
    )
    qa_chain = create_qa_with_sources_chain(llm)
    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name='context',
        document_prompt=DOC_PROMPT,
    )
    condense_question_chain = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
    )
    store_name = HISTORY_PATH.joinpath(get_storage_name(checksum) + '.pkl')
    if store_name.exists():
        with store_name.open(mode='rb') as fh:
            memory = pickle.load(fh)
        print(f'Chat history "{store_name}" loaded from disk.')
    else:
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        with store_name.open(mode='wb') as fh:
            pickle.dump(memory, fh)
        print(f'Chat history "{store_name}" saved to disk.')

    chain = ConversationalRetrievalChain(
        question_generator=condense_question_chain,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain=final_qa_chain,
    )

    return ChainManager(chain, store_name)


def get_response(query, chain_mgr):
    with get_openai_callback() as cb:
        response = chain_mgr.chain({'question': query})
        chain_mgr.save_memory()
        print(f'get_response callback - {cb}')
    return response


def main():
    sidebar()
    st.header("Chat with Document💬")

    uploaded_file = st.file_uploader(
        'Upload your documents',
        type=('txt', 'md', 'pdf'),
        accept_multiple_files=True,
    )
    if not uploaded_file:
        return

    docs, checksums = get_documents(uploaded_file)
    chunks = get_chunks_list(docs)
    vectorstore = get_vectorstore(chunks, checksum=checksums)
    chain_mgr = get_chain(vectorstore, checksums)

    query = st.text_input(
        'Ask something about the document',
        placeholder='Can you highlight the keypoints in bullets?',
        disabled=not uploaded_file,
    )
    if query:
        response = get_response(query, chain_mgr)

        response_answer = json.loads(response['answer'])
        answer = response_answer['answer']
        sources = '\n'.join([f'- {i}' for i in response_answer['sources']])

        st.markdown((
            f'**Question:**<br>'
            f'\n{query}'
        ), unsafe_allow_html=True)
        st.markdown((
            f'**Answer:**<br>'
            f'\n{answer}'
        ), unsafe_allow_html=True)
        if sources:
            st.markdown((
                f'**Sources:**<br>'
                f'\n{sources}'
            ), unsafe_allow_html=True)


if __name__ == '__main__':
    main()

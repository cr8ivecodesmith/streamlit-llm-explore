import json

import streamlit as st

from managers import (
    ChainManager,
    DocumentManager,
    VectorStoreManager,
)


def sidebar():
    # Sidebar contents
    with st.sidebar:
        st.title('🤖DocGPT Chat')
        st.write((
            'Made with 🫶 by '
            '[cr8ivecodesmith]'
            '(https://github.com/cr8ivecodesmith/streamlit-llm-explore)'
        ))


def get_documents(uploaded_files):
    docs = []
    for file_ in uploaded_files:
        doc_mgr = DocumentManager()
        doc_mgr.get_from_uploaded_file(file_)
        docs.append(doc_mgr)

    docs.sort(key=lambda x: x.checksum)
    return docs


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

    documents = get_documents(uploaded_file)
    vectorstore = VectorStoreManager(documents)
    chain = ChainManager(vectorstore)

    query = st.text_input(
        'Ask something about the document',
        placeholder='Can you highlight the keypoints in bullets?',
        disabled=not uploaded_file,
    )
    if query:
        response = chain.query(query)

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

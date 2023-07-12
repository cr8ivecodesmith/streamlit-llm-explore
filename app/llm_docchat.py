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

    if 'messages' not in st.session_state:
        # Load history from chain.memory
        st.session_state.messages = chain.get_history()

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            sources = message.get('sources')
            if sources:
                with st.expander('Sources'):
                    sources = '\n'.join([f'- {i}' for i in sources])
                    st.markdown(f'{sources}')

    query = st.chat_input(
        'Ask something about the document',
        disabled=not uploaded_file,
    )
    if query:
        response = chain.query(query)

        response_answer = json.loads(response['answer'])
        answer = response_answer['answer']
        sources = response_answer['sources']

        with st.chat_message('user'):
            st.markdown(f'{query}')

        with st.chat_message('assistant'):
            st.markdown(f'{answer}')

            if sources:
                with st.expander('Sources'):
                    sources = '\n'.join([f'- {i}' for i in sources])
                    st.markdown(f'{sources}')


if __name__ == '__main__':
    main()

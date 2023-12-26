from time import sleep

import json

import streamlit as st

from managers import (
    ChainManager,
    DocumentManager,
    PromptManager,
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
        sleep(5)

    docs.sort(key=lambda x: x.checksum)
    return docs


def load_message_history(chain):
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


def change_qa_prompt(prompts, chain):
    print('change_prompt')
    value = st.session_state.qa_prompt_select
    breakpoint()


def add_qa_prompt(prompts):
    value = st.session_state.qa_prompt_new_text
    if value and value not in prompts.qa_prompts:
        prompts.add_qa_prompt(value)


def load_prompts(prompts, chain):

    with st.container():
        st.session_state.qa_prompt_select = st.selectbox(
            'Select Question Prompt',
            options=list(range(len(prompts.qa_prompts))),
            index=prompts.qa_prompt_index,
        )
        st.text(prompts.qa_prompt)

        st.session_state.qa_prompt_new_text = st.text_area(
            label='Create a new Question Prompt',
            value=prompts.qa_prompt.strip(),
        )


def get_answer_from_external_sources(query: str):
    """Get answer from external sources using OpenAI API."""
    from langchain import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from utils import OPENAI_API_KEY

    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(question=query)
    st.markdown(f"[Answer from external source]: \n{response}")


def main():
    sidebar()
    st.header("Chat with Document💬")

    # Initialize chat session based on given documents
    uploaded_file = st.file_uploader(
        'Upload your documents',
        type=('txt', 'csv', 'md', 'pdf', 'xls', 'xlsx'),
        accept_multiple_files=True,
    )
    if not uploaded_file:
        return

    documents = get_documents(uploaded_file)
    vectorstore = VectorStoreManager(documents)
    prompts = PromptManager(vectorstore)
    chain = ChainManager(vectorstore)
    chain.init_chain(
        qa_prompt=prompts.qa_prompt,
        condense_question_prompt=prompts.cq_prompt,
    )

    # TODO: Load and display prompts
    # load_prompts(prompts, chain)

    # TODO: Prompt selection

    st.divider()

    # Load chat history
    load_message_history(chain)

    # Start chat
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
            if "I do not know the answer" in answer:
                get_answer_from_external_sources(query)
            else:
                st.markdown(f'{answer}')

            if sources:
                with st.expander('Sources'):
                    sources = '\n'.join([f'- {i}' for i in sources])
                    st.markdown(f'{sources}')


if __name__ == '__main__':
    main()

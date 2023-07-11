Streamlit LLM Explorer
===

Exploring LLMs for your document with Streamlit for the UI
and OpenAI LLM model.

This a proof-of-concept code but I want it fully featured and useable
for personal use.


## What you need?

- Python3
- OpenAI API Key


## Setup

1.) Fill out the environment variables needed

```
cp example.env .env
```


2.) Create a virtualenv (use your preferred method)

```
conda create --name llm-ex python=3
conda activate llm-ex
pip install -r requirements.txt
```

3.) Test

```
python ex/openai_check.py
```

## Usage


```
streamlit run app/llm_docchat.py
```


## TODOs

- [x] Use multiple documents for a session
- [x] Add memory to the conversation
- [x] Utilize `chat_models.ChatOpenAI` to include results from OpenAI's model
- [x] Conversing with the same documents picks up from the same history.
- [ ] Show conversation history
- [ ] Ability to have chat sessions
- [ ] Use a Chroma DB for the VectorStore instead of pickle
- [ ] Use a better data store for the chat history instead of pickle
- [ ] Add option to select OpenAI model and temperature
- [ ] Add more supported document types (i.e. webpages, csv, excel, word doc, epub)

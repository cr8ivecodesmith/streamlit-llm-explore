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

- Add memory to the conversation
- Show conversation history
- Use multiple documents for a session
- Utilize `chat_models.ChatOpenAI` to include results from OpenAI's model
- Use a Chroma DB for the VectorStore instead of pickle

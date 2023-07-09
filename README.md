Streamlit LLM Explorer
===


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
streamlit run app/llm_docchat2.py
```


## Resources

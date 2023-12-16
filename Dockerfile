# using python 3.9-slim to avoid issues with faiss-cpu
FROM python:3.9-slim

# set the working directory in the container
WORKDIR /app

# install necessary build tools
RUN apt-get update && \
    apt-get install -y gcc

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . /app

# EXPOSE 8501
EXPOSE 8501

CMD ["streamlit", "run", "app/llm_docchat.py"]

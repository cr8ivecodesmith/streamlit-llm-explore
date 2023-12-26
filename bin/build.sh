#!/bin/bash

function stop_delete_container() {
    # stop container
    docker stop docchat
    # delete container
    docker rm docchat
}

function build_container() {
    # build container
    docker build -t llm-docchat .
}

# execute stop_delete_container
stop_delete_container

# execute build_container
build_container


docker run -e OPENAI_API_KEY=$1 -p 8501:8501 --name docchat llm-docchat

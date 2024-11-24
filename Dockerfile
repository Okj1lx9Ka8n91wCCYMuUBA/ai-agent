FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

##
## User.
##

RUN apt update && apt install -y sudo

RUN groupadd -r user
RUN useradd -r -g user -m -s /bin/bash user
RUN usermod -aG sudo user

RUN echo "user ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers

USER user

WORKDIR /home/user

ENV USER=user

##
## Time zone.
##

ENV TZ=Europe/Moscow

RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime 
RUN echo $TZ | sudo tee /etc/timezone

##
## RealTimeSTT.
##

RUN sudo apt update && sudo apt install -y python3
RUN sudo apt update && sudo apt install -y python3-pip
RUN sudo apt update && sudo apt install -y python3-venv

RUN python3 -m venv venv

RUN bash -c "source venv/bin/activate && pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121"

##
## LLM.
##

RUN bash -c "source venv/bin/activate && pip install llama-index==0.11.23"
RUN bash -c "source venv/bin/activate && pip install llama-index-llms-ollama==0.3.6"

RUN sudo apt update && sudo apt install -y curl # Pull Ollama models with the `curl` POST requests.

RUN curl http://ollama:11434/api/pull -d '{ "model": "gemma2:9b" }'

##
## Service.
##

RUN sudo apt update && sudo apt install -y git

RUN bash -c "source venv/bin/activate && pip install fastapi==0.115.5"
RUN bash -c "source venv/bin/activate && pip install uvicorn==0.32.0"

EXPOSE 8808
RUN bash -c "source venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port 8808"
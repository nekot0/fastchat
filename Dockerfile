FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl git
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

# torch==2.6.0+cu126, torchvision==0.21.0+cu126, torchaudio==2.6.0+cu126
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# transformers==4.51.2, accelerate==1.6.0, sentencepiece==0.2.0
RUN pip3 install transformers accelerate sentencepiece

RUN pip3 install git+https://github.com/lm-sys/FastChat.git
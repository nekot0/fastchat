# FastChat with SLMs (edited on 12 Apr)

------------------------------
## Goal of this document
- Make SLM inferences efficient and capable of accomodating a certain number of users
- Install FastChat with SLMs as its backend and provide API for http request (API should be made https later for secure use of SLMs)


------------------------------
## SLMs tested
- microsoft/phi-4
- tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3

### FYI
- Tested on EC2 instance of g5.12xlarge with 4 A10G (GPU mem 96GB in total). 
- gemma-2 series currently have an issue with FastChat and not yet resolved. (see https://github.com/lm-sys/FastChat/issues/3448)


------------------------------
## Processes

### 1. Download FastChat repository
```bash
$ git clone https://github.com/lm-sys/FastChat.git
```

### 2. Modify Dockerfile & docker-compose.yml
```bash
# Files exist in the directory
$ FastChat/docker/Dockerfile
$ FastChat/docker/docker-compose.yml
```

#### Modification

**Dockerfile**
- FastChat libraries installed using the original Dockerfile are outdated, and the streaming response is not to be effective. Therefore, the modified Dockerfile uses the latest git repository obtained from github. 
- However, the latest FastChat does not include the installation of torch and those required to use it. Then, the installation processes are added in the Dockerfile.

```bash
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# added to skip interaction in package installation
ENV DEBIAN_FRONTEND=noninteractive

# add "git" to fetch github repository
RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl git
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

# add installations of torch, FastChat and those required
# the versions that were confirmed to work: torch==2.6.0+cu126, torchvision==0.21.0+cu126, torchaudio==2.6.0+cu126, transformers==4.51.2, accelerate==1.6.0, sentencepiece==0.2.0
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip3 install transformers accelerate sentencepiece
RUN pip3 install git+https://github.com/lm-sys/FastChat.git
```

**docker-compose.yml**
- The original docker-compose.yml only makes 1 model-worker. The modified one makes 1 additional model-worker, each of which is attached with the GPUs designated in the deploy section. 
- Several ways to attach GPUs to containers: write in the environment section or in the deploy section. The former way did not work and the model parameters were assigned to GPUs in ascending order of their indices. Therefore, this modified version uses the latter way.
- The dispatch method is originally "shortest-queue". The modified version sets "lottery" to select the worker with the fastest response at the moment. (see https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/controller.py)

```bash
# This example works with microsoft/phi-4
services:
  fastchat-controller:
    image: fastchat:latest
    ports:
      - "21001:21001"
    entrypoint: [
      "python3.9",
      "-m", "fastchat.serve.controller",
      "--dispatch-method", "lottery",
      "--host", "0.0.0.0", "--port", "21001"
    ]
  fastchat-model-worker-1:
    image: fastchat:latest
    volumes:
      - /home/ubuntu/models/phi-4:/model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
    entrypoint: [
      "python3.9",
      "-m", "fastchat.serve.model_worker",
      "--model-names", "phi-4",
      "--model-path", "/model",
      "--device", "cuda",
      "--num-gpus", "2",
      "--max-gpu-memory", "22GiB",
      "--worker-address", "http://fastchat-model-worker-1:21002",
      "--controller-address", "http://fastchat-controller:21001",
      "--host", "0.0.0.0", "--port", "21002"
    ]
  fastchat-model-worker-2:
    image: fastchat:latest
    volumes:
      - /home/ubuntu/models/phi-4:/model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2", "3"]
              capabilities: [gpu]
    entrypoint: [
      "python3.9",
      "-m", "fastchat.serve.model_worker",
      "--model-names", "phi-4",
      "--model-path", "/model",
      "--device", "cuda",
      "--num-gpus", "2",
      "--max-gpu-memory", "22GiB",
      "--worker-address", "http://fastchat-model-worker-2:21003",
      "--controller-address", "http://fastchat-controller:21001",
      "--host", "0.0.0.0", "--port", "21003"
    ]
  fastchat-api-server:
    image: fastchat:latest
    ports:
      - "8000:8000"
    entrypoint: ["python3.9", "-m", "fastchat.serve.openai_api_server", "--controller-address", "http://fastchat-controller:21001", "--host", "0.0.0.0", "--port", "8000"]
```


### 3. Build docker image
```bash
# command to be executed in the same directory as Dockerfile
$ docker build -t fastchat -f Dockerfile .
```


### 4. Download models from huggingface
- Use the FastChat container just built in "3. Build docker image".

```bash
# Download should be made on the container with huggingface-cli.
# Start a fastchat container and download models on it.
$ mkdir <model_path>
$ docker run --rm -it -v <model_path>:/model fastchat:latest bash
# On the container
$$ huggingface-cli download <model_name> --local-dir /model --local-dir-use-symlinks False
$$ exit
# Model files are downloaded in <model_path>
```


### 5. Start FastChat
```bash
# command to be executed in the same directory as docker-compose.yml
$ docker compose up -d
```


------------------------------
## Inference
- Inference is executed in the same way as ChatGPT API. 
- Below is a sample code. Two methods are implemented: obtain static responses or streaming responses from SLMs.

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

while True:
    print("--------------------")
    input_text = input("ask anything...")

    if input_text.lower() == "exit":
        break
    elif input_text == "":
        continue


    # static response (commented out)
    """
    # ------------------------
    payload = {
        "model": "llama-3.1-swallow-8b-instruct-v0.3",
        #"model": "phi-4",
        "messages": [
            {"role": "user", "content": input_text}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    content = response.json()["choices"][0]["message"]["content"]
    print("swallow: ", content)
    print("")
    # ------------------------
    """


    # streaming response
    # ------------------------
    payload = {
        #"model": "llama-3.1-swallow-8b-instruct-v0.3",
        "model": "phi-4",
        "stream": True,
        "messages": [
            {"role": "user", "content": input_text}
        ]
    }

    response = requests.post(url, json=payload, stream=True)
    print("")
    print("swallow: ", end=" ", flush=True)

    for line in response.iter_lines(decode_unicode=True):
        if not line or line.strip()=="data: [DONE]":
            continue
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                delta = data["choices"][0]["delta"]
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True)
            except Exception as e:
                print(f"\n[Error decoding line] {e}")
    print()
```


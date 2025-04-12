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


    # static response
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

import requests
from service_rag.app.config.config import setting


def connect_baidu_llm(question:str, prompt:str=""):
    print(f" 传过来的问题是🔥😂😂 {question} 😂😂😂😂")
    url = setting.CHAT_URL_TEMPLATE
    payload = {
        "model": "@cf/meta/llama-3.1-8b-instruct",
        "messages": [{
        "role": "user",
        "content": question +" "+prompt
    }]}

    r = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {setting.TOKEN_URL}"})

    print(f"结果是🚀🚀🚀🚀🚀🚀{r.json()} 🚀🚀🚀🚀🚀")
    body = r.json()
    if 'error_code' in body:
        print("[ERNIE ERROR]", body)
        raise RuntimeError(f"ERNIE API:{body['error_code']} {body.get('error_msg', '')}")
    #
    return {
        "role": body.get('choices', [{}])[0].get('message', {}).get('role', ''),
        "content": body.get('choices', [{}])[0].get('message', {}).get('content', '')
    }


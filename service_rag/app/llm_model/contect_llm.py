import os, pathlib, requests
from service_rag.app.config.config import setting

TOKEN_DIR  = pathlib.Path(__file__).resolve().parents[3] / "token"
TOKEN_FILE = TOKEN_DIR / "llm_token.txt"
TOKEN_DIR.mkdir(exist_ok=True)

def load_token():
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    return None

def save_token(llmtoken: str):
    TOKEN_FILE.write_text(llmtoken)
    os.chmod(TOKEN_FILE, 0o600)

def get_access_token():
    access_token = load_token()
    if access_token:
        return access_token
    r = requests.post(setting.TOKEN_URL, headers={'Content-Type': 'application/json'})
    r.raise_for_status()
    access_token = r.json()['access_token']
    save_token(access_token)
    return access_token

def connect_baidu_llm(question:str, prompt:str=""):
    print(f" 传过来的问题是😂😂 {question}")
    url = setting.CHAT_URL_TEMPLATE.format(get_access_token())
    payload = {"messages": [{
        "role": "user",
        "content": question +" "+prompt
    }]}
    r = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    body = r.json()
    if 'error_code' in body:
        print("[ERNIE ERROR]", body)
        raise RuntimeError(f"ERNIE API:{body['error_code']} {body.get('error_msg', '')}")
    #
    return {
        "role": "assistant",
        "content": body['result']
    }


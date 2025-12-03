import os, json, pathlib, requests
from service_rag.app.config.config import setting
TOKEN_DIR  = pathlib.Path(__file__).resolve().parents[3] / "token"
TOKEN_FILE = TOKEN_DIR / "llm_token.txt"
TOKEN_DIR.mkdir(exist_ok=True)


class ConnectLLm:

    def __init__(self):
        pass

    def load_token(self):
        if TOKEN_FILE.exists():
            return TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
        return None

    def save_token(self, llmtoken: str):
        TOKEN_FILE.write_text(llmtoken)
        os.chmod(TOKEN_FILE, 0o600)

    def get_access_token(self):
        access_token = self.load_token()
        if access_token:
            return access_token
        r = requests.post(setting.TOKEN_URL, headers={'Content-Type': 'application/json'})
        r.raise_for_status()
        access_token = r.json()['access_token']
        self.save_token(access_token)
        return access_token


    def connect_baidu_llm(self, question:str, prompt:str=""):

        url = setting.CHAT_URL_TEMPLATE.format(self.get_access_token())
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


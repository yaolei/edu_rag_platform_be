class Config:
    AK = "z7Ftb94wYtzyjKBFjFfqAkmn"
    SK = "VIschOTtaiSAaFHJh72FxdMXZGhDaJre"
    TOKEN_URL = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={AK}&client_secret={SK}"
    CHAT_URL_TEMPLATE = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token={}"

setting = Config()
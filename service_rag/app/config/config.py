class Config:
    TOKEN_URL = f"KizrytjuiXK1wpmd7Z0YbNDQfK4Q18n5wGODR0GG"
    CHAT_URL_TEMPLATE="https://api.cloudflare.com/client/v4/accounts/59861947ace4956d903803045f80b0fa/ai/v1/chat/completions"
    CHAT_URL_OPENAI="https://api.cloudflare.com/client/v4/accounts/59861947ace4956d903803045f80b0fa/ai/v1/responses"
    CHAT_URL_LLAMA="https://api.cloudflare.com/client/v4/accounts/59861947ace4956d903803045f80b0fa/ai/run/@cf/meta/llama-4-scout-17b-16e-instruct"

    CHAT_URL_IMAGE_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/59861947ace4956d903803045f80b0fa/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
    CHAT_URL_UFROM_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/59861947ace4956d903803045f80b0fa/ai/run/@cf/unum/uform-gen2-qwen-500m"

setting = Config()
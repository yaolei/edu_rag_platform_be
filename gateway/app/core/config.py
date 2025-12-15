from pydantic.v1 import BaseSettings


class Settings(BaseSettings):

    origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://localhost:5000",
        "http://localhost:5100",
        "http://localhost:5002",
        "http://localhost:5003",
        "http://localhost:5004",
    ]

    SERVICE = {
        'gateway': f"http://localhost:8000",
        'edu_rag': f"http://localhost:8001",
        'edu_shell': f"http://localhost:8002",
        'edu_admin': f"http://localhost:8003",
    }

settings = Settings()
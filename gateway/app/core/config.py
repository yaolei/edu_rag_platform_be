from pydantic.v1 import BaseSettings
from typing import List

class Settings(BaseSettings):
    origins: List[str] = [
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
        'gateway': "http://localhost:8000",
        'edu_rag': "http://localhost:8001",
        'edu_shell': "http://localhost:8002",
        'edu_admin': "http://localhost:8003",
    }

    class Config:
        env_file = '.env.production'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        json_loads = lambda x: [i.strip() for i in x.split(',') if i.strip()]

settings = Settings()
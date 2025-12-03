from pydantic import BaseModel, Field, constr
from datetime import datetime
from typing import List, Optional
from service_rag.app.modules.dataset_module import KnowledgeItemORM
class ChatRequest(BaseModel):
    questions: Optional[str] = Field(...)

class ResponseORM(BaseModel):
    status: int
    message: str

class KnowledgeItemCreate(BaseModel):
    knowledgeName:str = Field(..., max_length=94)
    activate:bool = Field(...)
    corpus_id:str = Field(..., max_length=700)

class DelKnowledgeItem(BaseModel):
    corpus_id:List[str]
    id:int

class KnowledgeItems(BaseModel):
    id: int
    knowledgeName:str
    activate:bool
    corpus_id:str
    created_at: datetime

    model_config = {"from_attributes": True}

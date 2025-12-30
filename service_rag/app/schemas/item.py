from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
class ChatRequest(BaseModel):
    questions: Optional[str] = Field(...)

class ResponseORM(BaseModel):
    status: int
    message: str

class KnowledgeItemCreate(BaseModel):
    knowledgeName:str = Field(..., max_length=94)
    activate:bool = Field(...)
    doc_type:str
    corpus_id:str

class DelKnowledgeItem(BaseModel):
    corpus_id:List[str]
    id:int

class KnowledgeItems(BaseModel):
    id: int
    knowledgeName:str
    activate:bool
    corpus_id:str
    doc_type:str
    created_at: datetime

    model_config = {"from_attributes": True}

class DeleteKnowledgeItem(BaseModel):
    corpus_ids:List[str]
    id:int
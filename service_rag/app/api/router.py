from fastapi import APIRouter, Depends, File, UploadFile,Form
from conect_databse.database import get_db
from sqlalchemy.orm import Session
from service_rag.app.llm_model.contect_llm import ConnectLLm
from service_rag.app.schemas.item import ChatRequest, ResponseORM, KnowledgeItems
from typing import List

from service_rag.app.service import item as svc
router = APIRouter()

@router.get('/')
def test_api():
    return {'message': 'Evan work fine test'}

@router.post('/chat')
def chat_with_ai(body: ChatRequest):
    con = ConnectLLm()
    res = con.connect_baidu_llm(question=body.questions, prompt="")
    return {'status': 200, 'content': res}

@router.post('/upload_knowledge', response_model=ResponseORM)
async def create_knowledge_item(
        knowledgeName: str = Form(..., max_length=94),
        activate:bool = Form(...),
        file:UploadFile = File(...),
        db:Session = Depends(get_db)
     ):

    knowledge_data = {
        "knowledgeName": knowledgeName,
        "activate":activate,
    }
    return await svc.create_knowledge_item(db, knowledge_data, file)

@router.get('/knowledge_items', response_model=List[KnowledgeItems])
async def get_knowledge_items(db:Session = Depends(get_db)):
    return await svc.get_knowledge_items(db)

@router.get('/del_knowledge_items', response_model=ResponseORM)
async def get_knowledge_item(db:Session = Depends(get_db)):
    return await svc.delete_knowledge_item(db)
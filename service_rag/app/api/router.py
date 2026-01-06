from fastapi import APIRouter, Depends, File, UploadFile,Form
from fastapi import HTTPException
from conect_databse.database import get_db
from sqlalchemy.orm import Session
from service_rag.app.schemas.item import ChatRequest, ResponseORM, KnowledgeItems, DeleteKnowledgeItem
from typing import List, Optional

from service_rag.app.service import item as svc
router = APIRouter()

@router.get('/')
def test_api():
    return {'message': 'Evan work fine test'}


@router.post('/chat_with_knowledge_stream')
async def chat_stream(body: ChatRequest):
    try:
        return await svc.chat_with_knowledge_api_stream(questions=body.questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/chat_by_files_stream')
async def chat_by_file_knowledge_stream( questions: Optional[str] = Form(None),   # 普通文本
                                  files: List[UploadFile] = File([])):
    try:
        return await svc.chat_with_knowledge_file_stream(question=questions, files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/upload_knowledge', response_model=ResponseORM)
async def create_knowledge_item(
        knowledgeName: str = Form(..., max_length=94),
        activate:bool = Form(...),
        document_type:str = Form(...),
        file:UploadFile = File(...),
        db:Session = Depends(get_db)
     ):

    knowledge_data = {
        "knowledgeName": knowledgeName,
        "activate":activate,
        "doc_type":document_type,
    }
    return await svc.create_knowledge_item(db, knowledge_data, [file])


@router.post('/chat_by_files')
async def chat_by_file_knowledge( questions: Optional[str] = Form(None),   # 普通文本
                                  files: List[UploadFile] = File([])):

    res = await svc.chat_with_knowledge_by_files(question=questions, files=files)
    return {
        'status': 200,
        'content' : res
    }

@router.get('/knowledge_items', response_model=List[KnowledgeItems])
async def get_knowledge_items(db:Session = Depends(get_db)):
    return await svc.get_knowledge_items(db)

@router.get('/del_knowledge_items', response_model=ResponseORM)
async def get_knowledge_item(db:Session = Depends(get_db)):
    return await svc.delete_knowledge_item(db)

@router.post('/del_knowledge_items_by_id', response_model=ResponseORM)
async def get_knowledge_item(ids:DeleteKnowledgeItem, db:Session = Depends(get_db)):
    return await svc.delete_knowledge_item_by_ids(ids, db)

@router.get('/dev_test_api_vector')
async def dev_test_api_vector():
    return await svc.dev_env_test_api()
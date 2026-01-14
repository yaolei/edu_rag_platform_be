from fastapi import APIRouter, Depends, File, UploadFile,Form
from fastapi import HTTPException
from conect_databse.database import get_db
from sqlalchemy.orm import Session
from service_rag.app.schemas.item import ResponseORM, KnowledgeItems, DeleteKnowledgeItem
from typing import List, Optional
import  time, json
from service_rag.app.service import item as svc
router = APIRouter()

# 简单的内存存储（生产环境建议使用Redis）
conversation_storage = {}

@router.get('/')
def test_api():
    return {'message': 'Evan work fine test'}

@router.post('/chat_with_knowledge_stream')
async def chat_stream(
        conversation_id: Optional[str] = Form(None),
        intent_type: str = Form(...),
        messages_json: str = Form(...)
):
    try:
        messages = json.loads(messages_json)
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages_json必须是数组")

        return await svc.chat_with_knowledge_api_stream(
            conversation_id=conversation_id,
            intent_type=intent_type,
            messages=messages
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的JSON格式")
    except Exception as e:
        print(f"❌ 接口异常: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post('/chat_by_files_stream')
async def chat_by_file_knowledge_stream(
        files: List[UploadFile] = File([]),
        conversation_id: Optional[str] = Form(None),
        intent_type: str = Form(...),
        messages_json: Optional[str] = Form(None)
):
    try:
        messages = []
        if messages_json:
            try:
                messages = json.loads(messages_json)
                print(f"📷 OCR - 解析到 {len(messages)} 条历史消息")
            except json.JSONDecodeError as e:
                print(f"❌ OCR JSON解析失败: {e}")
                messages = []

        return await svc.chat_with_knowledge_file_stream(
            files=files,
            messages=messages,
            conversation_id=conversation_id,
            intent_type=intent_type,
        )
    except Exception as e:
        print(f"❌ OCR接口异常: {e}")
        import traceback
        traceback.print_exc()
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


@router.post('/cleanup_conversations')
async def cleanup_conversations(max_age_hours: int = 24):
    """
    清理过期的会话
    """
    current_time = time.time()
    expired = []
    for conv_id, context_manager in conversation_storage.items():
        if context_manager.history:
            last_time = context_manager.history[-1]["timestamp"]
            # 如果超过指定小时没有活动，清理
            if (current_time - last_time) > (max_age_hours * 3600):
                expired.append(conv_id)

    for conv_id in expired:
        del conversation_storage[conv_id]

    return {"cleaned": len(expired), "remaining": len(conversation_storage)}
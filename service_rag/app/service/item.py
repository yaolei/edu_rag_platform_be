import json

from sqlalchemy.orm import Session
from service_rag.app.schemas.item import KnowledgeItemCreate, KnowledgeItems
from service_rag.app.repositories import item as repo
from service_rag.app.run_rag import RagService


async def create_knowledge_item(db:Session, obj, file):
    try:

        rag = await RagService.create(upload_file=file, embedding_type="store")
        store_ids = rag.run_rag_engine()
        print(f"✅ {store_ids} is store successfully")

        req_data = KnowledgeItemCreate(
            knowledgeName=obj['knowledgeName'],
            activate=obj['activate'],
            corpus_id=json.dumps(['123123', '123123123']),
        )
        res = repo.create_knowledge_item(db, req_data)

        if res:
            return {
                "status": 200,
                "message": "success",
            }

    except Exception as e:
        return {
            "status": 500,
            "message": str(e),
        }
async def get_knowledge_items(db:Session):
    return  [KnowledgeItems.model_validate(i) for i in repo.get_knowledge_item(db)]

async def delete_knowledge_item(db:Session):
    pass
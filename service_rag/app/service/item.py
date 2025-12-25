import json

from sqlalchemy.orm import Session
from service_rag.app.llm_model.contect_llm import connect_text_llm
from service_rag.app.schemas.item import KnowledgeItemCreate, KnowledgeItems
from service_rag.app.repositories import item as repo
from service_rag.app.run_rag import RagService


async def create_knowledge_item(db:Session, obj, file):
    try:
        rag = await RagService.create(upload_file=file, embedding_type="store")
        store_ids = await rag.run_rag_engine()
        if len(store_ids) > 0:
            print(f" ✅ 保存知识库存储成功, 开始进行物理数据库索引索引保存.")
            req_data = KnowledgeItemCreate(
                knowledgeName=obj['knowledgeName'],
                activate=obj['activate'],
                corpus_id=json.dumps(store_ids),
            )
            res = repo.create_knowledge_item(db, req_data)
            print(f" ✅ 物理数据库索引数据保存成功.")
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

    try:
        rag = await RagService.create()
        rag.clear_all_documents()

        res = repo.del_knowledge_all_item(db)

        return {
            "status": 200,
            "message": f"ALl data is removed ! {res}",
        }
    except Exception as e:
        return {
            "status": 500,
            "message": str(e),
        }


def chat_with_none_knowledge(body):
    res =  connect_text_llm(question=body.questions)
    return {'status': 200, 'content': res}

async def chat_with_knowledge_infor(questions):
    rag = await RagService.create(embedding_type="questions", question=questions)
    rag_message = await rag.run_rag_engine()
    return {'status': 200, 'content': rag_message}


async def delete_knowledge_item_by_ids(ids, db:Session):
    try:
        rag = await RagService.create()
        del_doc = rag.del_knowledge_item(ids)

        if del_doc:
            res = repo.delete_knowledge_item(db, ids.id)
            if res:
                return {
                    "status": 200,
                    "message": f"Deleted knowledge {del_doc} success",
                }
        else:
            return {
                "status": 404,
                "message": "no knowledge item found",
            }
    except Exception as e:
        return {
            "status": 500,
            "message": str(e),
        }

async def chat_with_knowledge_by_files(files, question):
    try:
        rag = await RagService.create(embedding_type="questions", upload_file=files, question= question)
        image_result = await rag.analyse_image_information()
        return image_result
    except Exception as e:
        print(f"❌ 处理文件时出错: {str(e)}")
        raise

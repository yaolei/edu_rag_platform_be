import json
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse, JSONResponse
from service_rag.app.schemas.item import KnowledgeItemCreate, KnowledgeItems, ChatRequest
from service_rag.app.repositories import item as repo
from service_rag.app.run_rag import RagService


async def create_knowledge_item(db:Session, obj, file):
    try:
        rag = await RagService.create(upload_file=file, embedding_type="store", doc_type=obj['doc_type'])
        store_ids = await rag.upload_infor_to_vector()
        if len(store_ids) > 0:
            print(f" ✅ 保存知识库存储成功, 开始进行物理数据库索引索引保存.")
            req_data = KnowledgeItemCreate(
                knowledgeName=obj['knowledgeName'],
                activate=obj['activate'],
                doc_type=obj['doc_type'],
                corpus_id=json.dumps(store_ids),
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

async def dev_env_test_api():
    rag = await RagService.create(embedding_type="questions", question='')
    rag.dev_env_test_api()


async def chat_with_knowledge_file_stream(files, question):
    try:
        rag = await RagService.create(
            embedding_type="questions",
            upload_file=files,
            question=question
        )

        print(f"🎯 开始处理文件流式响应，文件数量: {len(files)}")

        async def generate():
            try:
                print("🔄 开始生成流式响应...")
                chunk_count = 0
                async for chunk in rag.analyse_image_information():
                    chunk_count += 1
                    if chunk_count % 5 == 0:  # 每5个chunk打印一次日志
                        print(f"📦 向客户端发送第 {chunk_count} 个 chunk")
                    if chunk:
                        yield chunk
                print(f"✅ 流式响应生成完成，共 {chunk_count} 个 chunk")
            except Exception as e:
                import json
                print(f"❌ 生成流时出错: {e}")
                error_msg = json.dumps({"error": f"生成流时出错: {str(e)}"})
                yield f"data: {error_msg}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


async def chat_with_knowledge_api_stream(questions):
    try:
        rag = await RagService.create(question=questions)
        res_doc = rag.question_query_from_vector()
        async def generate():
            try:
                async for chunk in rag.stream_context_from_docs(res_doc):
                    if chunk:
                        yield chunk
            except Exception as e:
                import json
                error_msg = json.dumps({"error": f"生成流时出错: {str(e)}"})
                yield f"data: {error_msg}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )
    except Exception as e:
        # 如果创建 StreamResponse 失败，返回错误响应
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
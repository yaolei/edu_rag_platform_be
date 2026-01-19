from typing import Optional, List, Dict
from fastapi import UploadFile
import asyncio
import time
import json
from pathlib import Path
from service_rag.app.embedding.embedding_data import EmbeddingData
from service_rag.app.prompt.prompt import prompt_setting
from langchain_core.documents import Document
from service_rag.app.document_operation.document_loader import DocumentLoader
from service_rag.app.text_splitter.text_split import TextSplitter
from service_rag.app.vector.vector_store import VectorStore
from service_rag.app.llm_model.contect_llm import analyze_with_image, stream_llm_response
from service_rag.app.text_splitter.advanced_text_cleaner import AdvancedTextCleaner
from service_rag.app.service.gen_util import build_simple_context, prue_image_chunks

class RagService:
    def __init__(self):
        self.embedding_type = None
        self.upload_file = None
        self.file_name = None
        self.target_file = None
        self.embeddings = None
        self.vector = None
        self.question = None
        self.file_type = None
        self.if_files = None
        self.doc_type = None
        self.mutil_files = []

        self.image_binary_data = None
        self.conversation_id = None
        self.messages = []
        self.last_doc_types = []
        self.intent_type = None

    @classmethod
    async def create(cls, upload_file: List[UploadFile]=None, embedding_type='questions', doc_type="document",
                                                             conversation_id: Optional[str] = None,
                                                             messages: Optional[List[Dict]] = None,
                                                             intent_type="chat", **kwargs):
        instance = cls()
        await instance.initialize(upload_file, embedding_type, doc_type, conversation_id=conversation_id,
                                                                                      messages=messages,
                                                                                      intent_type=intent_type,
                                                                                      **kwargs)
        return instance

    async def initialize(self, upload_file: List[UploadFile]=None, embedding_type='questions', doc_type="document",
                         conversation_id: Optional[str] = None,
                         messages: Optional[List[Dict]] = None,
                         intent_type="chat", **kwargs):
        self.embedding_type = embedding_type

        self.upload_file = upload_file
        self.doc_type = doc_type
        self.intent_type=intent_type
        self.conversation_id = conversation_id  # 新增
        self.messages = messages or []
        self.question = ""
        if self.messages:
            for msg in reversed(self.messages):
                if msg.get("role") == "user":
                    self.question = msg.get("content", "").strip()
                    break
            if self.question:
                print(f"🎯 从messages中提取的问题: {self.question}")

        self.embeddings = EmbeddingData(embedding_type=embedding_type)
        self.vector = VectorStore(embedding_function=self.embeddings)

        if self.messages:
            print(f"📚 接收到 {len(self.messages)} 条历史消息")

        if not upload_file:
            pass
        elif len(upload_file) == 1:
            self.if_files = False
            self.file_name = upload_file[0].filename or "unknown file"
            path_obj = Path(self.file_name)

            try:
                if upload_file[0].content_type and upload_file[0].content_type.startswith('image/'):
                    self.intent_type = 'image'
                    content = await upload_file[0].read()
                    self.image_binary_data = content
                    # 重置文件指针，以便 DocumentLoader 可以读取
                    await upload_file[0].seek(0)

                document_loader = DocumentLoader(upload_file[0])
                self.target_file = await document_loader.load()
                document_loader.cleanup_temp_resources()
                if self.target_file and self.target_file[0].page_content == '':
                    self.target_file = None
                else:
                    if 'document_loader' in locals():
                        document_loader.cleanup_temp_resources()
                    self.file_type = document_loader._detect_document_type()
                    return False
            except Exception as e:
                print(f"❌ embedding module error: {str(e)}")
                raise e
        else:
            self.if_files = True
            for f in upload_file:
                document_loader_muti_file = DocumentLoader(f)
                self.mutil_files.append(await document_loader_muti_file.load())
                document_loader_muti_file.cleanup_temp_resources()
            print(f"🐯 {self.mutil_files} 🐯")

    async def llava_get_content(self, prompt_sentence, image_bytes):
        """获取LLaVA分析结果"""
        prompt_sentence = prompt_sentence.strip()
        print(f"🌛 发送给LLaVA的提示词: {prompt_sentence}")
        print(f"🌛 发送给LLaVA的提示词长度: {len(prompt_sentence)}")

        final_answer = await analyze_with_image(
            image_bytes=image_bytes,
            question=prompt_sentence,
        )

        print(f"🌟 分析的结果: {final_answer}")

        if isinstance(final_answer, dict) and 'content' in final_answer:
            result_content = final_answer['content'].strip()
        else:
            result_content = str(final_answer).strip()

        print(f"🌛 LLaVA返回结果长度: {len(result_content)}")
        return result_content

    async def analyse_image_information(self):
        """
        分析图片信息 - 统一使用message数组模式
        """
        try:
            image_byte_content = self.image_binary_data
            print(f"✅ 使用缓存的图片二进制数据: {len(image_byte_content)} 字节")
            user_question = ""
            if self.messages:
                for msg in reversed(self.messages):
                    if msg.get("role") == "user":
                        user_question = msg.get("content", "").strip()
                        break
            # 纯图片，更倾向人物风景图
            is_pure_image = not self.target_file
            if is_pure_image:
                print(f"🎯 进入纯图片分析分支，用户的问题: {user_question}")
                tmp_question = user_question if user_question else "请描述下这张图片的内容"
                prue_image_prompt = prompt_setting.pure_image_qa_template.format(question=tmp_question)
                result_content = await self.llava_get_content(
                    prue_image_prompt,
                    image_byte_content,
                )
                # 将结果流式返回
                chunks = prue_image_chunks(result_content)
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    data = {"choices": [{"delta": {"content": chunk + " "}}]}
                    yield f"data: {json.dumps(data)}\n\n"

                yield "data: [DONE]\n\n"
                return
            else:
                # ========== 图文处理模式 ==========
                print(f"🦁 开始分析图文信息")

                # 提取OCR文本
                ocr_text = self.target_file[0].page_content if self.target_file else ""
                print(f"🌛 OCR文本长度: {len(ocr_text)}")
                knowledge_base_info = ""
                # 如果有用户提问，尝试检索知识库
                if user_question and user_question.strip():
                    # 如果是chat模式，不涉及知识库查询
                    if self.intent_type != 'chat':
                        relevant_docs = self.vector.query_by_question_vector_with_filter(
                            question_vector=user_question,
                            doc_types=self.intent_type,
                            top_k=3
                        )

                        if relevant_docs and len(relevant_docs) > 0:
                            # 构建知识库上下文
                            knowledge_context = build_simple_context(relevant_docs)
                            knowledge_base_info = knowledge_context
                            print(f"🎯 知识库检索到 {len(relevant_docs)} 条相关信息")
                    prompt_muti_model = prompt_setting.image_word_qa_template_ocr.format(
                        question=user_question,
                        ocr_text=ocr_text,
                        knowledge_base=knowledge_base_info
                    )

                else:
                    # 没有用户问题，使用纯图片分析提示词
                    prompt_muti_model = "请详细描述这张图片的内容。"

                # 获取图文分析结果
                result_content = await self.llava_get_content(
                    prompt_muti_model,
                    image_byte_content
                )

                # 构建system消息（包含OCR和知识库）
                system_message_parts = []

                if ocr_text:
                    # 限制OCR长度，避免过长
                    if len(ocr_text) > 2000:
                        ocr_preview = ocr_text[:2000] + "...[后面内容已省略]"
                    else:
                        ocr_preview = ocr_text
                    system_message_parts.append(f"<ocr>【图片OCR文本内容】\n{ocr_preview}")

                if knowledge_base_info:
                    system_message_parts.append(f"<ocr>【相关知识库信息】\n{knowledge_base_info}")

                # 创建包含system消息和LLaVA结果的消息数组
                response_messages = []

                # 如果有system消息内容，添加到response_messages
                if system_message_parts:
                    system_message = "\n\n".join(system_message_parts)
                    system_message += "\n\n<ocr>"
                    response_messages.append({"role": "system", "content": system_message})

                # 添加LLaVA的assistant消息
                response_messages.append({"role": "assistant", "content": result_content})

                # 流式返回所有消息
                for message in response_messages:
                    # 如果是system消息，添加一个标记让前端知道这是system
                    if message["role"] == "system":
                        # 可以添加一个特殊标记，比如"__system__": true
                        data = {
                            "choices": [{"delta": {"content": message["content"]}}],
                            "role": "system"
                        }
                    else:
                        data = {"choices": [{"delta": {"content": message["content"]}}]}

                # 流式返回
                    chunks = prue_image_chunks(message["content"])
                    for chunk in chunks:
                        if not chunk.strip():
                            continue
                        # 更新chunk内容
                        if message["role"] == "system":
                            chunk_data = {
                                "choices": [{"delta": {"content": chunk + " "}}],
                                "role": "system"
                            }
                        else:
                            chunk_data = {"choices": [{"delta": {"content": chunk + " "}}]}

                        yield f"data: {json.dumps(chunk_data)}\n\n"

                yield "data: [DONE]\n\n"
                return

        except Exception as e:
            print(f"❌ 图片分析异常: {e}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    def store_document_to_vector(self, chunks, doc_type):
        try:
            print(f"🚀 共有{len(chunks)} 进行保存，文档类型: {doc_type}")
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'metadata'):
                    chunk.metadata['doc_type'] = doc_type
                else:
                    chunk.metadata = {'doc_type': doc_type}
            ids = self.vector.add_document_to_vector(chunks)
            print(f" stored {self.file_name} documents successfully")
            return ids
        except Exception as e:
                print(f" stored {self.file_name} documents failed: {str(e)}")
                raise e

    def del_knowledge_item(self, ids):
        corpus_ids = self.collation_ids(ids)

        result = []
        try:
            for corpus_id in corpus_ids:
                res = self.vector.delete_document(corpus_id)
                result.append(res)

        except Exception as e:
            print(f"删除失败向量数据库数据: {str(e)}")
            raise e

        if None in result:
            return None
        else:
            return True

    def clear_all_documents(self):
        self.vector.clear_collection()

    def question_query_from_vector(self):
        """
        逻辑：直接基于用户的意图分析，后期可融入智能分析，但是复杂性和分析质量问题有困难
        """
        results = self.vector.query_by_question_vector_with_filter(
                question_vector=self.question,
                doc_types=self.intent_type,
                top_k=8
            )
        if results and len(results) > 0:
            return results
        else:
            print(f"⚠️ 过滤查询无结果，知识库中没有相关类型的内容")
            return []

    def get_chunk_doc(self, target_file, clear_chunks=False):
        try:
            print(f"🚀 start split {self.file_name}")
            splitter_chunks = TextSplitter().split_document(target_file)

            if clear_chunks:
                chunks = self.clear_data(splitter_chunks)
            else:
                chunks = splitter_chunks
            return chunks
        except Exception as e:
            print(f" split error: {e}")
            raise e

    async def stream_context_from_docs(self, documents):
        """流式生成上下文 - 正确的多轮对话处理"""
        llm_messages = []

        # 1. 如果有知识库信息，作为system消息
        if documents:
            context_str = build_simple_context(documents)
            system_content = prompt_setting.knowledge_history_template.format(context_str=context_str)
            if self.intent_type == 'resume':
                system_content = prompt_setting.knowledge_history_resume_template.format(context_str=context_str)

            llm_messages.append({
                "role": "system",
                "content": system_content
            })

        # 2. 检查是否是指代图片的问题
        is_image_reference = False
        image_reference_text = ""

        if self.messages and len(self.messages) >= 2:
            current_question = self.question or ""
            # 检查是否包含图片相关的指代词
            image_keywords = ["图", "图片", "照片", "截图", "画面", "图像", "photo", "image"]
            has_image_keyword = any(keyword in current_question for keyword in image_keywords)

            if has_image_keyword:
                # 查找最近的图片描述
                for i in range(len(self.messages) - 2, -1, -1):  # 从倒数第二条往前找
                    msg = self.messages[i]
                    if isinstance(msg, dict):
                        # 检查是否是assistant的回复且包含图片描述特征
                        content = msg.get("content", "")
                        if ("这是一张" in content or "照片" in content or
                                "场景" in content or "画面" in content):
                            is_image_reference = True
                            image_reference_text = content
                            break
        if self.messages:
            # 确保格式正确
            for msg in self.messages:
                normalized_msg = {}
                # 转换role
                if "type" in msg:
                    normalized_msg["role"] = "user" if msg["type"] == "user" else "assistant"
                elif "role" in msg:
                    normalized_msg["role"] = msg["role"]
                else:
                    normalized_msg["role"] = "user"  # 默认

                # 确保content存在
                if "content" in msg:
                    normalized_msg["content"] = msg["content"]
                elif "text" in msg:
                    normalized_msg["content"] = msg["text"]
                else:
                    normalized_msg["content"] = ""

                # 只添加有内容的message
                if normalized_msg["content"].strip():
                    llm_messages.append(normalized_msg)

        if is_image_reference and image_reference_text:
            # 在最后一条用户消息后添加系统提示
            for i in range(len(llm_messages) - 1, -1, -1):
                if llm_messages[i].get("role") == "user":
                    # 修改当前用户问题，明确引用图片描述
                    original_content = llm_messages[i]["content"]
                    enhanced_content = f"""
                            {original_content}
                        （提示：根据之前的对话，图片描述为：{image_reference_text[:200]}...请基于这个图片描述回答。）"""

                    llm_messages[i]["content"] = enhanced_content
                    break
        # 记录开始时间
        start_time = time.time()
        try:
            # 调用流式LLM，传递正确的messages数组
            async for chunk in stream_llm_response(llm_messages):
                if chunk:
                    yield chunk

            end_time = time.time()
            print(f"✅ 流式生成完成，耗时: {end_time - start_time:.2f}秒")

        except Exception as e:
            print(f"❌ 流式生成异常: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    async def upload_infor_to_vector(self):
        try:
            if self.file_type != 'image':
                print(f" ✅ 开始进行保存知识库操作, 上传的知识类型{self.doc_type}")
                chunks = self.get_chunk_doc(self.target_file)
                stored_ids = self.store_document_to_vector(chunks, self.doc_type)
                return stored_ids
            else:
                print(f"不能上传图片")
                pass
        except Exception as e:
            print(f"❌存储向量数据库失败 {str(e)}")

    def clear_data(self, chunks):
        all_rag_chunks = []

        for j, chunk in enumerate(chunks):
            cur_document = Document(
                page_content=chunk.page_content,
                metadata={
                    "source":self.file_name,
                    "chunk":j
                }
            )
            all_rag_chunks.append(cur_document)
        clearner = AdvancedTextCleaner()
        cleaned_chunks = clearner.clean_documents(all_rag_chunks)
        print(f" from {len(all_rag_chunks)} remove {len(cleaned_chunks)} ")
        return cleaned_chunks

    def collation_ids(self, ids):
        data_dict = dict(ids)
        corpus_ids = data_dict.get("corpus_ids", [])
        return corpus_ids

    def dev_env_test_api(self):
        self.vector.verify_doc_type_storage()
        # 验证特定类型
        # self.vector.verify_doc_type_storage("resume")
        # self.vector.verify_doc_type_storage("code")

    # async def run_by_web(self):
    #     print(f"🚀 Rag started at {datetime.datetime.now()} ")
    #     try:
    #         loader = DocumentLoader(urls=["https://tailwindcss.com/docs/installation/using-vite"])
    #         docs = await loader.load()
    #         text = "\n".join([doc.page_content for doc in docs])
    #         chunks = TextSplitter().split_text(text)
    #         print(f" 分割完成，共生成 {len(chunks)} 个文本块")
    #         return chunks
    #     except Exception as e:
    #         print(f" 执行失败: {e}")
    #         raise

    # async def main():
    #     app = RagService()
    #     res = await app.run_rag()
    #     for i, chunk in enumerate(res):
    #         print(f"--- 分块 {i + 1} ---\n{chunk[:200]}...")
    #
    # asyncio.run(main())


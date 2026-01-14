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
from service_rag.app.llm_model.contect_llm import  connect_text_llm, analyze_with_image, stream_llm_response
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

    async def llava_get_content(self, prompt_sentence, image_bytes, is_text_image, user_question=""):
        """获取LLaVA分析结果"""
        prompt_sentence = prompt_sentence.strip()
        print(f"🌛 is_text_image: {is_text_image}")
        print(f"🌛 用户问题: {user_question}")

        if not is_text_image:
            # 纯图片模式
            if user_question and user_question.strip():
                # 有用户提问，使用问答模板
                llava_prompt = prompt_setting.pure_image_qa_template.format(question=user_question)
                print(f"🦁 纯图片带问题提问模式")
            else:
                # 没有用户提问，使用描述模板
                llava_prompt = prompt_sentence
                print(f"🦁 纯图片描述模式")
        else:
            # 图文混合模式 - 直接使用传入的提示词
            llava_prompt = prompt_sentence
            print(f"🦁 图文混合分析模式")

        print(f"🌛 发送给LLaVA的提示词长度: {len(llava_prompt)}")

        final_answer = await analyze_with_image(
            image_bytes=image_bytes,
            question=llava_prompt,
        )

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

            # 获取最后一个用户消息（如果有）
            user_question = ""
            if self.messages:
                for msg in reversed(self.messages):
                    if msg.get("role") == "user":
                        user_question = msg.get("content", "").strip()
                        break

            # 纯图片
            is_pure_image = not self.target_file
            if is_pure_image:
                print("🎯 进入纯图片分析分支")

                # 情况1: 无用户提问 - 直接返回图片描述
                if not user_question or user_question.strip() == "":
                    print("🎯 纯图片无提问，直接返回描述")
                    result_content = await self.llava_get_content(
                        prompt_setting.prue_image_analysis_template,
                        image_byte_content,
                        False,  # 不是图文混合
                        ""  # 无用户提问
                    )

                    # 将结果流式返回
                    chunks = prue_image_chunks(result_content)
                    for chunk in chunks:
                        if not chunk.strip():
                            continue
                        data = {"choices": [{"delta": {"content": chunk + " "}}]}
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(min(0.15, max(0.05, len(chunk) / 300)))

                    yield "data: [DONE]\n\n"
                    return

                # 情况2: 有用户提问 - 使用message数组模式
                else:
                    print("🎯 纯图片有提问，使用message数组模式")
                    # 获取图片分析结果
                    image_description = await self.llava_get_content(
                        prompt_setting.prue_image_analysis_template,
                        image_byte_content,
                        False,  # 不是图文混合
                        user_question  # 传递用户提问
                    )

                    # 构建system消息
                    system_message = f"【图片分析结果】\n{image_description}\n\n请根据图片内容回答用户问题。"

                    # 构建完整的消息数组
                    llm_messages = [{"role": "system", "content": system_message}]

                    # 添加历史消息（前端已限制数量）
                    if self.messages:
                        for msg in self.messages:
                            normalized_msg = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                            if normalized_msg["content"].strip():
                                llm_messages.append(normalized_msg)

                    print(f"🔄 纯图片message模式: 消息总数 {len(llm_messages)}")

                    # 调用流式LLM
                    async for chunk in stream_llm_response(llm_messages):
                        yield chunk

                    yield "data: [DONE]\n\n"
                    return

            else:
                # ========== 图文处理模式 ==========
                print(f"🦁 开始分析图文信息")

                # 获取图文分析结果
                image_description = await self.llava_get_content(
                    prompt_setting.prue_image_analysis_template,
                    image_byte_content,
                    True,  # 图文混合
                    user_question if user_question else ""  # 传递用户提问（如果存在）
                )

                # 提取OCR文本
                ocr_text = self.target_file[0].page_content if self.target_file else ""
                print(f"🌛 OCR文本长度: {len(ocr_text)}")

                # 构建基础system消息
                system_message_parts = []
                if image_description:
                    system_message_parts.append(f"【图片分析结果】\n{image_description}")
                if ocr_text:
                    system_message_parts.append(f"【OCR文本内容】\n{ocr_text}")

                # 如果有用户提问，尝试检索知识库
                if user_question and user_question.strip():
                    print(f"🎯 有用户提问，进行意图分析和知识库查询")
                    ''''
                         如果是chat模式，不涉及知识库查询
                    '''

                    if self.intent_type != 'chat':
                        relevant_docs = self.vector.query_by_question_vector_with_filter(
                            question_vector=user_question,
                            doc_types=self.intent_type,
                            top_k=3
                        )

                        if relevant_docs and len(relevant_docs) > 0:
                            # 构建知识库上下文
                            knowledge_context = build_simple_context(relevant_docs)
                            system_message_parts.append(f"【相关知识库信息】\n{knowledge_context}")
                            print(f"🎯 知识库检索到 {len(relevant_docs)} 条相关信息")

                # 如果没有用户提问，直接返回分析结果
                if not user_question or user_question.strip() == "":
                    print("🎯 没有用户问题，直接返回图文分析结果")
                    combined_content = "\n\n".join(system_message_parts)

                    # 将结果流式返回
                    chunk_size = 50
                    for i in range(0, len(combined_content), chunk_size):
                        chunk = combined_content[i:i + chunk_size]
                        data = {"choices": [{"delta": {"content": chunk}}]}
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0.01)

                    yield "data: [DONE]\n\n"
                    return

                # 有用户提问，使用完整的message数组模式
                system_message = "\n\n".join(system_message_parts)
                system_message += "\n\n请根据图片内容、OCR文本和相关知识库信息回答用户问题。"

                # 构建完整的消息数组
                llm_messages = [{"role": "system", "content": system_message}]

                # 添加历史消息
                if self.messages:
                    for msg in self.messages:
                        normalized_msg = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                        if normalized_msg["content"].strip():
                            llm_messages.append(normalized_msg)

                print(f"🔄 图文message模式: system消息长度 {len(system_message)}, 消息总数 {len(llm_messages)}")

                # 调用流式LLM
                async for chunk in stream_llm_response(llm_messages):
                    yield chunk

                yield "data: [DONE]\n\n"

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
        chat模式不查询知识库，不提供知识库内部信息
        """
        results = self.vector.query_by_question_vector_with_filter(
                question_vector=self.question,
                doc_types=self.intent_type,
                top_k=5  # 只需要5个最优结果
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
        # 构建消息数组
        llm_messages = []

        # 1. 如果有知识库信息，作为system消息
        if documents:
            context_str = build_simple_context(documents)
            system_content =  prompt_setting.knowledge_history_template.format(context_str=context_str)
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

        # 2. 直接传递原始对话历史（前端已限制数量）
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


        print(f"🔄 文本模式:开始流式生成，消息总数: {len(llm_messages)}")

        # 记录开始时间
        start_time = time.time()
        try:
            # 调用流式LLM，传递正确的messages数组
            async for chunk in stream_llm_response(llm_messages):
                if chunk:
                    yield chunk

            yield "data: [DONE]\n\n"
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

    def analyze_intent_with_llm(self, question):
        """
        使用LLM分析问题意图，返回可能的doc_type数组
        """
        try:
            result = connect_text_llm(question)

            # 简化处理：直接提取content
            if isinstance(result, dict):
                content = result.get('content', '')
            else:
                content = str(result)

            # 尝试解析JSON
            import json
            import re

            # 清理content
            content = content.strip()

            # 提取JSON部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    content_dict = json.loads(json_str)
                    doc_types = content_dict.get('doc_types', [])
                    print(f"🎯 LLM意图分析结果: {doc_types}")
                    return doc_types
                except json.JSONDecodeError:
                    print(f"❌ JSON解析失败，内容: {content[:100]}...")

            print(f"⚠️ 未能解析doc_types，返回空数组")
            return []

        except Exception as e:
            print(f"❌ LLM意图分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

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


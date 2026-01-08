from typing import Optional, List, Dict
from fastapi import UploadFile
import asyncio
import time
import json
from pathlib import Path
from service_rag.app.embedding.embedding_data import EmbeddingData
from service_rag.app.prompt.prompt import prompt_setting
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from service_rag.app.document_operation.document_loader import DocumentLoader
from service_rag.app.text_splitter.text_split import TextSplitter
from service_rag.app.vector.vector_store import VectorStore
from service_rag.app.llm_model.contect_llm import  connect_text_llm, analyze_with_image, stream_llm_response
from service_rag.app.text_splitter.advanced_text_cleaner import AdvancedTextCleaner
from service_rag.app.service.gen_util import switch_correct_prompt, build_simple_context, prue_image_chunks

class RagService:
    def __init__(self):
        self.prompt = PromptTemplate(input_variables=['context', 'question'],
                                     template=prompt_setting.rag_template)
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

    @classmethod
    async def create(cls, upload_file: List[UploadFile]=None, embedding_type='questions', doc_type="document",
                                                             conversation_id: Optional[str] = None,
                                                             messages: Optional[List[Dict]] = None, **kwargs):
        instance = cls()
        await instance.initialize(upload_file, embedding_type, doc_type, conversation_id=conversation_id,
                                                                                      messages=messages,
                                                                                      **kwargs)
        return instance

    async def initialize(self, upload_file: List[UploadFile]=None, embedding_type='questions', doc_type="document",
                         conversation_id: Optional[str] = None,
                         messages: Optional[List[Dict]] = None, **kwargs):
        self.embedding_type = embedding_type

        self.upload_file = upload_file
        self.doc_type = doc_type

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

        if not upload_file:  # 无文件
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
                print(f"🚀🚀🚀🚀{ self.target_file} 🚀🚀")
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


    async def llava_get_content(self, prompt_sentence, image_bytes, is_text_image):
        prompt_sentence = prompt_sentence.strip()

        if self.messages and len(self.messages) > 0:
            history_text = ""
            for msg in self.messages[-5:]:  # 只取最近5条消息
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"

            enhanced_prompt = f"【对话历史】\n{history_text}\n【当前任务】\n"
        else:
            enhanced_prompt = ""

        if not is_text_image:
            if self.question:
                llava_prompt = prompt_setting.pure_image_qa_template.format(question=self.question)
                print(f"🦁 用户提问: {self.question}")
            else:
                llava_prompt = prompt_sentence
                print(f"🦁 用户未提问，自动生成图片描述")
        else:
            llava_prompt = prompt_sentence

        # 如果有历史记录，添加到提示词中
        if self.messages and len(self.messages) > 0:
            llava_prompt = enhanced_prompt + llava_prompt
            print(f"🎯 使用上下文增强图片分析")

        final_answer = await analyze_with_image(
            image_bytes=image_bytes,
            question=llava_prompt,
            messages=self.messages
        )

        if isinstance(final_answer, dict) and 'content' in final_answer:
            result_content = final_answer['content'].strip()
        else:
            result_content = str(final_answer).strip()

        return result_content

    async def analyse_image_information(self):
        """
        1. 使用专业提示词让LLaVA分析图片
        2. 分析用户问题意图
        3. 根据意图决定是否查询知识库
        4. 使用专业图片问答模板生成最终回答
        """
        try:
            print(f"🦁 处理文件: {self.file_name}")
            image_byte_content = self.image_binary_data
            print(f"✅ 使用缓存的图片二进制数据: {len(image_byte_content)} 字节")

            # 获取对话历史
            history_str = ""
            if self.messages:
                for msg in self.messages:
                    role = "用户" if msg.get("role") == "user" else "助手"
                    content = msg.get("content", "")
                    history_str += f"{role}: {content}\n"

            # 纯图片
            is_pure_image = not self.target_file
            if is_pure_image:
                print("🎯 进入纯图片分析分支")
                # 获取纯图片分析结果
                result_content = await self.llava_get_content(
                    prompt_setting.prue_image_analysis_template,
                    image_byte_content,
                    False
                )
                print(f"📊 获取到纯图片分析结果，长度: {len(result_content)}")

                if self.messages and len(self.messages) > 0:
                    # 构建带上下文的提示词
                    conversation_prompt = prompt_setting.image_conversation_template.replace(
                        '{history}', history_str
                    ).replace(
                        '{image_analysis}', result_content
                    ).replace(
                        '{question}', self.question if self.question else "请描述这张图片"
                    )

                    # 使用新的提示词重新分析
                    enhanced_result = await self.llava_get_content(
                        conversation_prompt,
                        image_byte_content,
                        False
                    )
                    result_content = enhanced_result
                    print(f"🎯 使用上下文增强分析，新长度: {len(result_content)}")

                chunks = prue_image_chunks(result_content)
                # 将字符串转换为流式返回 - 使用异步方式
                import json
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    data = {
                        "choices": [{"delta": {"content": chunk + " "}}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                    # 根据chunk长度动态调整延迟
                    delay = min(0.15, max(0.05, len(chunk) / 300))
                    await asyncio.sleep(delay)

                yield "data: [DONE]\n\n"
                return

            else:
                # ========== 情况1：图文处理 ==========
                print(f"🦁 开始分析图像信息，问题: {self.question} 🦁")

                # 检查用户是否输入提问信息
                analyse_text_image = await self.llava_get_content(
                    prompt_setting.rag_image_analysis_template,
                    image_byte_content,
                    True
                )

                if self.messages and len(self.messages) > 0:
                    conversation_prompt = prompt_setting.image_conversation_template.replace(
                        '{history}', history_str
                    ).replace(
                        '{image_analysis}', analyse_text_image
                    ).replace(
                        '{question}', self.question if self.question else "请分析图片内容"
                    )

                    enhanced_result = await self.llava_get_content(
                        conversation_prompt,
                        image_byte_content,
                        True
                    )
                    analyse_text_image = enhanced_result

                if not self.question or self.question.strip() == "":
                    print("🎯 没有用户问题，直接返回图片分析结果")
                    # 将字符串转换为流式返回
                    import json
                    chunk_size = 50
                    total_chunks = (len(analyse_text_image) + chunk_size - 1) // chunk_size

                    for i in range(0, len(analyse_text_image), chunk_size):
                        chunk = analyse_text_image[i:i + chunk_size]
                        data = {
                            "choices": [{"delta": {"content": chunk}}]
                        }
                        print(f"📤 发送第 {i // chunk_size + 1}/{total_chunks} 个 chunk，长度: {len(chunk)}")
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0.01)

                    yield "data: [DONE]\n\n"

                else:
                    print(f"🎯 有用户问题，进行意图分析和知识库查询")
                    image_description = analyse_text_image
                    ocr_text = self.target_file[0].page_content
                    intent_analysis_prompt = prompt_setting.image_intent_prompt.format(
                        image_description=image_description,
                        ocr_text=ocr_text
                    )
                    doc_types = self.analyze_intent_with_llm(intent_analysis_prompt)
                    print(f"🈶 问题的图文类型结果是: {doc_types}")

                    if len(doc_types) > 0:
                        print(f"🈶 知识库包含问题类型，开始进行知识库查询")
                        relevant_docs = self.vector.query_by_question_vector_with_filter(
                            question_vector=self.question,
                            doc_types=doc_types,
                            top_k=5
                        )

                        if len(relevant_docs) > 0:
                            print(f"🎯 知识库有相关信息，开始智能融合知识库信息和用户问题")
                            final_prompt_for_text_model = switch_correct_prompt(
                                self.question,
                                doc_types[0],
                                image_description,
                                relevant_docs,
                                ocr_text
                            )

                            # 记录开始时间
                            start_time = time.time()
                            print(f"🔄 开始流式生成，prompt长度: {len(final_prompt_for_text_model)}")

                            # 调用流式LLM
                            chunk_count = 0
                            llm_messages = self.messages.copy() if self.messages else []
                            llm_messages.append({"role": "user", "content": final_prompt_for_text_model})
                            async for chunk in stream_llm_response(llm_messages):
                                if chunk:
                                    chunk_count += 1
                                    if chunk_count % 10 == 0:  # 每10个chunk打印一次
                                        print(f"📤 流式LLM第 {chunk_count} 个 chunk")
                                    yield chunk

                            # 发送结束信号
                            yield "data: [DONE]\n\n"
                            end_time = time.time()
                            print(f"✅ 流式生成完成，共 {chunk_count} 个 chunk，耗时: {end_time - start_time:.2f}秒")

                        else:
                            print(f"🎯 知识库没有相关信息，直接返回图片分析结果")
                            # 将字符串转换为流式返回
                            import json
                            chunk_size = 50
                            total_chunks = (len(analyse_text_image) + chunk_size - 1) // chunk_size

                            for i in range(0, len(analyse_text_image), chunk_size):
                                chunk = analyse_text_image[i:i + chunk_size]
                                data = {
                                    "choices": [{"delta": {"content": chunk}}]
                                }
                                print(f"📤 发送第 {i // chunk_size + 1}/{total_chunks} 个 chunk，长度: {len(chunk)}")
                                yield f"data: {json.dumps(data)}\n\n"
                                await asyncio.sleep(0.01)

                            yield "data: [DONE]\n\n"

                    else:
                        print(f"🎯 无匹配文档类型，返回图片分析结果")
                        # 将字符串转换为流式返回
                        import json
                        chunk_size = 50
                        total_chunks = (len(analyse_text_image) + chunk_size - 1) // chunk_size

                        for i in range(0, len(analyse_text_image), chunk_size):
                            chunk = analyse_text_image[i:i + chunk_size]
                            data = {
                                "choices": [{"delta": {"content": chunk}}]
                            }
                            print(f"📤 发送第 {i // chunk_size + 1}/{total_chunks} 个 chunk，长度: {len(chunk)}")
                            yield f"data: {json.dumps(data)}\n\n"
                            await asyncio.sleep(0.01)

                        yield "data: [DONE]\n\n"

        except Exception as e:
            import json
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
        新逻辑：使用LLM分析意图，然后进行过滤查询
        """
        print(f"🔍 执行向量查询，问题: '{self.question}'")

        # 1. 使用LLM分析意图
        intent_prompt = prompt_setting.intent_analysis_template.replace('{question}', self.question)
        doc_types = self.analyze_intent_with_llm(intent_prompt)

        # 2. 如果有匹配的doc_type，进行过滤查询
        if doc_types and len(doc_types) > 0:
            print(f"🎯 使用过滤查询 (目标分区: {doc_types})")

            # 使用过滤查询
            results = self.vector.query_by_question_vector_with_filter(
                question_vector=self.question,
                doc_types=doc_types,
                top_k=5  # 只需要5个最优结果
            )

            if results and len(results) > 0:
                return results
            else:
                print(f"⚠️ 过滤查询无结果，知识库中没有相关类型的内容")
                return []
        else:
            print(f"🎯 无匹配的文档类型，知识库没有相关信息")
            return []

    def get_chunk_doc(self, target_file, clear_chunks=False):
        try:
            print(f"🚀 start split {self.file_name}")
            splitter_chunks = TextSplitter().split_document(target_file)

            if clear_chunks:
                chunks = self.clear_data(splitter_chunks)
                print(f"🚀 🚀 🚀  {chunks}")
            else:
                chunks = splitter_chunks
            return chunks
        except Exception as e:
            print(f" split error: {e}")
            raise e

    async def stream_context_from_docs(self, documents):
        """流式生成上下文"""
        history_str = ""
        if self.messages:
            for msg in self.messages:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                history_str += f"{role}: {content}\n"

        if not documents:
            # 没有知识库信息，使用 no_knowledge_template
            formatter_prompt = prompt_setting.no_knowledge_template.replace(
                '{question}', self.question if self.question else ""
            )
        else:
            # 有知识库信息
            context_str = build_simple_context(documents)

            # 如果有历史记录，使用 rag_template_pro
            if history_str:
                formatter_prompt = prompt_setting.rag_template_pro.replace(
                    '{history}', history_str
                ).replace(
                    '{context}', context_str
                ).replace(
                    '{question}', self.question if self.question else ""
                )
            else:
                # 没有历史记录，使用 rag_template
                formatter_prompt = prompt_setting.rag_template.replace(
                    '{context}', context_str
                ).replace(
                    '{question}', self.question if self.question else ""
                )

        print(f"🔄 开始流式生成，prompt长度: {len(formatter_prompt)}")
        # 记录开始时间
        start_time = time.time()
        try:
            llm_messages = [
                {"role": "user", "content": formatter_prompt}
            ]
            # 调用流式LLM
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
            print(f"🎯🎯🎯🎯🎯🎯🎯传过来的问题是: \n{question} 🎯🎯🎯🎯🎯🎯")
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


from typing import Optional, List
from fastapi import UploadFile
from service_rag.app.embedding.embedding_data import EmbeddingData
from service_rag.app.prompt.prompt import prompt_setting
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from service_rag.app.document_operation.document_loader import DocumentLoader
from service_rag.app.text_splitter.text_split import TextSplitter
from service_rag.app.vector.vector_store import VectorStore
from service_rag.app.llm_model.contect_llm import  connect_text_llm, analyze_with_image
from service_rag.app.text_splitter.advanced_text_cleaner import AdvancedTextCleaner
from pathlib import Path
import base64

class RagService:
    def __init__(self):
        self.prompt = PromptTemplate(input_variables=['context', 'question'],
                                     template=prompt_setting.rag_template)
        self.embedding_type = None
        self.upload_file = None
        self.file_name = None
        self.file_name_without_extension = None
        self.target_file = None
        self.embeddings = None
        self.vector = None
        self.question = None
        self.file_type = None
        self.if_files = None
        self.doc_type = None
        self.mutil_files = []

    @classmethod
    async def create(cls, upload_file: List[UploadFile]=None, embedding_type='questions', doc_type="document",
                     question:Optional[str] = None, **kwargs):
        instance = cls()
        await instance.initialize(upload_file, embedding_type, doc_type, question, **kwargs)
        return instance

    async def initialize(self, upload_file: List[UploadFile]=None, embedding_type='questions', doc_type="document",
                         question:Optional[str] = None, **kwargs):
        self.embedding_type = embedding_type
        self.question = question
        self.upload_file = upload_file
        self.doc_type = doc_type
        self.embeddings = EmbeddingData(embedding_type=embedding_type)
        self.vector = VectorStore(embedding_function=self.embeddings)
        if not upload_file:  # 无文件
            pass
        elif len(upload_file) == 1:
            self.if_files = False
            self.file_name = upload_file[0].filename or "unknown file"
            path_obj = Path(self.file_name)
            self.file_name_without_extension = path_obj.stem

            try:
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

    async def analyse_image_information(self):
        """
        重构后的图片分析流程：
        1. 先用极简指令让LLaVA分析图片，得到客观描述。
        2. 用描述中的关键词去查询向量数据库。
        3. 最后结合描述和知识库，用文本模型生成最终答案。
        """
        print(f"🦁 开始分析图像信息，问题: {self.question} 🦁")

        # 0. 直接读取图片文件 (无论有无OCR文本，都需要分析图片)
        upload_file = self.upload_file[0]
        content = await upload_file.read()
        base64_str = base64.b64encode(content).decode("utf-8")
        image_data_url = f"data:{upload_file.content_type};base64,{base64_str}"
        print(f"🦁 处理文件: {upload_file.filename}")

        # ========== 第一步：让图片模型进行基础分析 ==========
        print(f"🦁 步骤1: 调用LLaVA进行基础图片分析...")
        # 使用一个极简、聚焦的提示词，只要求描述
        image_analysis_prompt = "请详细描述这张图片的场景、主要内容、物体、颜色和氛围。"
        image_analysis_result = analyze_with_image(
            image_base64_data_url=image_data_url,
            question=image_analysis_prompt  # 传入简短的、只关于图片本身的问题
        )

        # 提取图片描述文本
        if isinstance(image_analysis_result, dict) and 'content' in image_analysis_result:
            image_description = image_analysis_result['content'].strip()
        else:
            image_description = str(image_analysis_result).strip()

        # 处理LLaVA输出乱码的情况：如果描述异常简短或包含大量重复字符，视为失败
        if len(image_description) < 50 or "幅幅幅" in image_description:
            print(f"❌ LLaVA分析失败或输出异常，直接使用备用提示。")
            image_description = f"用户上传了一张图片，文件名为：{upload_file.filename}。"

        print(f"🦁 获得的图片描述摘要: {image_description[:150]}...")

        # ========== 第二步：基于图片描述查询知识库 ==========
        print(f"🦁 步骤2: 基于图片描述查询知识库...")
        # 使用图片描述（而不是OCR文本）作为查询依据
        query_for_vector = f"根据以下图片描述，查找相关知识：{image_description[:500]}"  # 限制长度
        relevant_docs = self.vector.query_by_question_vector(query_for_vector)

        knowledge_context = ""
        if relevant_docs and relevant_docs != "False" and len(str(relevant_docs).strip()) > 10:
                try:
                    if isinstance(relevant_docs, list):
                        # 提取每个doc的text字段
                        text_list = []
                        for doc in relevant_docs:
                            if isinstance(doc, dict) and 'text' in doc:
                                text_list.append(doc['text'])
                            elif hasattr(doc, 'page_content'):  # 如果是Document对象
                                text_list.append(doc.page_content)
                        knowledge_context = "\n\n".join(text_list)
                    else:
                        knowledge_context = str(relevant_docs)
                    print(f"🦁 找到相关知识点，长度: {len(knowledge_context)}")
                except Exception as e:
                    print(f"❌ 提取text字段时出错: {str(e)}")
        else:
            knowledge_context = "知识库中未找到与图片直接相关的信息。"
            print(f"🦁 未在知识库中找到相关信息")

        # ========== 第三步：综合信息，生成最终回答 ==========
        print(f"🦁 步骤3: 综合图片描述与知识库信息，生成最终回答...")
        # 构建给文本模型的提示词
        final_prompt_for_text_model = f"""
                请根据以下信息回答用户的问题。
            
                【图片分析结果】
                {image_description}
            
                【相关背景知识】
                {knowledge_context}
            
                【用户提出的问题】
                {self.question if self.question else '请分析这张图片。'}
            
                请将图片分析结果和相关背景知识有机结合，生成一个完整、流畅的回答。如果背景知识显示“未找到相关信息”，则主要依据图片分析结果回答。
                回答的开头请加上：“Evan 让您久等了。”
                """
        # 调用你的文本聊天函数
        final_answer = connect_text_llm(
            question=final_prompt_for_text_model  # 这里传入整合了所有信息的提示
        )

        # 处理最终结果
        if isinstance(final_answer, dict) and 'content' in final_answer:
            result_content = final_answer['content']
        else:
            result_content = str(final_answer)

        print(f"🦁 最终回答生成完毕，长度: {len(result_content)}")
        return result_content
    def store_document_to_vector(self, chunks, doc_type):
        try:
            print(f"🚀 共有{len(chunks)} 进行保存，文档类型: {doc_type}")
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'metadata'):
                    chunk.metadata['doc_type'] = doc_type
                else:
                    chunk.metadata = {'doc_type': doc_type}
            ids = self.vector.add_document_to_vector(chunks)
            print(f" stored {self.file_name_without_extension} documents successfully")
            return ids
        except Exception as e:
                print(f" stored {self.file_name_without_extension} documents failed: {str(e)}")
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
        doc_types = self.analyze_intent_with_llm(self.question)

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
                print(f"✅ 过滤查询完成: {len(results)} 个结果")
                return results
            else:
                print(f"⚠️ 过滤查询无结果，知识库中没有相关类型的内容")
                return []
        else:
            # 3. 如果没有匹配的doc_type，知识库没有相关信息
            print(f"🎯 无匹配的文档类型，知识库没有相关信息")
            return []

    def get_chunk_doc(self, target_file, clear_chunks=False):
        try:
            print(f"🚀 start split {self.file_name_without_extension}")
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

    def get_context_from_docs(self, documents):
        """构建上下文"""
        if not documents:
            formatter_prompt = prompt_setting.no_knowledge_template.replace(
                '{question}', self.question
            )
        else:
            context_str = self._build_simple_context(documents)
            formatter_prompt = prompt_setting.rag_template_pro.replace(
                '{context}', context_str
            ).replace(
                '{question}', self.question
            )

        print(f"✅ 最终Prompt长度: {len(formatter_prompt)} 字符")
        return connect_text_llm(formatter_prompt)

    def _build_simple_context(self, documents):
        """构建纯净的上下文，去掉内部标记和元数据"""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents[:5]):  # 最多5个
            content = ""

            if isinstance(doc, dict):
                content = doc.get('text', '')
                if not content:
                    content = doc.get('page_content', '')
                    if not content and hasattr(doc, 'get'):
                        # 尝试获取第一个字符串值
                        for key, value in doc.items():
                            if isinstance(value, str) and len(value.strip()) > 0:
                                content = value
                                break
            elif hasattr(doc, 'page_content'):
                # Document对象
                content = doc.page_content

            if content:
                content = content.strip()
                import re
                content = re.sub(r'\s+', ' ', content)

                # 只添加非空内容
                if content:
                    context_parts.append(content)

        if not context_parts:
            return ""

        return "\n\n---\n\n".join(context_parts)

    async def run_rag_engine(self):
        if self.embedding_type == 'questions':
            print(f"✅进入问答场景....")
            res_doc = self.question_query_from_vector()
            try:
                print(f"🚀 start query answer by LLM...")
                return self.get_context_from_docs(res_doc)
            except Exception as e:
                print(f"❌🔥 {str(e)}")
                raise e
        else:
            if self.file_type !='image':
                print(f" ✅ 开始进行保存知识库操作, 上传的知识类型{self.doc_type}")
                print(f" 上传的文件名称: {self.file_name_without_extension}")
                chunks = self.get_chunk_doc(self.target_file)
                stored_ids = self.store_document_to_vector(chunks, self.doc_type)
                return stored_ids
            else:
                print(f"不能上传图片")
                pass


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
            # 使用prompt.py中的意图分析模板
            intent_prompt = prompt_setting.intent_analysis_template.replace('{question}', question)

            print(f"🎯 发送给LLM的意图分析请求: {intent_prompt[:200]}...")

            # 直接传递字符串参数
            result = connect_text_llm(intent_prompt)

            # 调试：打印result的类型和内容
            print(f"🎯 connect_text_llm返回类型: {type(result)}")
            print(f"🎯 connect_text_llm返回值: {result}")

            # 处理返回结果
            content_dict = {}
            if isinstance(result, dict):
                print(f"🎯 result是字典，keys: {result.keys()}")
                content = result.get('content', '')

                # 重要：content可能是字典，也可能是字符串
                if isinstance(content, dict):
                    content_dict = content
                elif isinstance(content, str):
                    # 尝试解析字符串为字典
                    import json
                    try:
                        content_dict = json.loads(content)
                    except json.JSONDecodeError:
                        # 如果不是JSON，尝试提取JSON
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                content_dict = json.loads(json_match.group())
                            except:
                                pass

            # 从content_dict中提取doc_types
            if isinstance(content_dict, dict) and 'doc_types' in content_dict:
                doc_types = content_dict['doc_types']
                print(f"🎯 LLM意图分析结果: {doc_types}")
                return doc_types

            # 如果以上都失败，使用简单的关键词匹配
            return self._fallback_intent_analysis(question)

        except Exception as e:
            print(f"❌ LLM意图分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回默认值
            return ['document']

    def _fallback_intent_analysis(self, question):
        """备用意图分析方法：基于关键词匹配"""
        question_lower = question.lower()
        doc_types = []

        # 简历相关关键词
        if any(word in question_lower for word in
               ['简历', '求职', '候选人', '开发者', '经验', '招聘', '推荐', '工作经历', '项目经验']):
            doc_types.append('resume')
        # 代码相关关键词
        if any(word in question_lower for word in ['代码', '编程', '技术栈', '开发', '程序', 'bug']):
            doc_types.append('code')
        # 图片相关关键词
        if any(word in question_lower for word in ['图片', '图像', '照片', '图']):
            doc_types.append('image_desc')
        # 文档相关关键词
        if any(word in question_lower for word in ['文档', '文件', '资料']):
            doc_types.append('document')

        if not doc_types:
            doc_types.append('document')  # 默认

        print(f"🎯 关键词匹配意图分析结果: {doc_types}")
        return doc_types

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


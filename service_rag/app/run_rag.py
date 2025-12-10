from typing import Optional
from fastapi import UploadFile
import datetime, asyncio

from service_rag.app.embedding.embedding_data import EmbeddingData
from service_rag.app.prompt.prompt import prompt_setting
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from service_rag.app.document_operation.document_loader import DocumentLoader
from service_rag.app.text_splitter.text_split import TextSplitter
from service_rag.app.vector.vector_store import VectorStore
from service_rag.app.llm_model.contect_llm import connect_baidu_llm
from service_rag.app.text_splitter.advanced_text_cleaner import AdvancedTextCleaner
from pathlib import Path

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

    @classmethod
    async def create(cls, upload_file: UploadFile=None, embedding_type='questions', question:Optional[str] = None, **kwargs):
        instance = cls()
        await instance.initialize(upload_file, embedding_type, question, **kwargs)
        return instance

    async def initialize(self, upload_file: UploadFile=None, embedding_type='questions', question:Optional[str] = None, **kwargs):
        self.embedding_type = embedding_type
        self.question = question
        self.upload_file = upload_file

        self.embeddings = EmbeddingData(embedding_type=embedding_type)
        print(f"✅ embedding module include success")

        self.vector = VectorStore(embedding_function=self.embeddings)

        if upload_file:
            self.file_name = upload_file.filename or "unknown file"
            path_obj = Path(self.file_name)
            self.file_name_without_extension = path_obj.stem

            try:
                document_loader = DocumentLoader(upload_file)
                self.target_file = await document_loader.load()
            except Exception as e:
                print(f"❌ embedding module error: {str(e)}")
                raise e


    def store_document_to_vector(self, chunks):
        try:
            print(f"🚀 共有{len(chunks)} 进行保存")
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
        document = self.vector.query_by_question_vector(self.question)
        return document

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
        if not documents:
            context_str = "(上下文知识库未检索到相关内容)"
        else:
            context_str = "\n".join(d["text"] for d in documents)
        formatter_prompt = self.prompt.format(
            context=context_str,
            question=self.question,
        )

        return connect_baidu_llm(formatter_prompt)

    async def run_rag_engine(self):
        print(f"🚀 RAG engine start and current embedding type: 🌟{self.embedding_type}🌟")
        if self.embedding_type == 'questions':
            print(f" Flow the question process....")
            res_doc = self.question_query_from_vector()
            try:
                print(f"🚀 start query answer by LLM...")
                return self.get_context_from_docs(res_doc)
            except Exception as e:
                print(f"❌🔥 {str(e)}")
                raise e

        else:
            print(f" Query all  the documents from vector process....")
            print(f" load file {self.file_name_without_extension}")

            chunks = self.get_chunk_doc(self.target_file)

            stored_ids = self.store_document_to_vector(chunks)
            return stored_ids


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


import os
from uuid import uuid4
from chromadb.config import Settings
from langchain_chroma import Chroma
from numpy.distutils.from_template import list_re
from langchain_community.vectorstores.utils import filter_complex_metadata


class VectorStore:
    def __init__(self, embedding_function, collection_name='example_collection',
                 persist_directory='./chroma_langchain_db',
                 enable_rerank=True,
                 similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.enable_rerank = enable_rerank
        self.example_collection = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectors = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            client_settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
    # clear all the data
    def clear_collection(self):
        try:
            collections = self.vectors._collection
            all_ids = collections.get()['ids']
            if all_ids:
                collections.delete(ids=all_ids)
                print(f"✅ All collections deleted and removed: {len(all_ids)}")
            else:
                print(f"collections is empty: {len(all_ids)}")
        except Exception as e:
            print(str(e))
        return True

    def add_document_to_vector(self, document):
        try:
            uuid = [str(uuid4()) for _ in range(len(document))]
            docs = filter_complex_metadata(document)
            self.vectors.add_documents(documents=docs, ids=uuid)

        except Exception as e:
            print(str(e))
            raise e
        print(f"✅ successfully added {len(document)} documents to vector store")
        return uuid

    def query_by_question_vector(self, question_vector):
        # embedding_query = self.embedding_function.embed_query(question_vector)
        results_with_score = self.vectors.similarity_search_with_score(question_vector,
                                                                     k=10 if self.enable_rerank else 30)
        filtered_results = []
        for doc, score in results_with_score:
            if score >= self.similarity_threshold:
                filtered_results.append((doc, score))

        if not filtered_results:
            print(f"* 📊 无符合阈值({self.similarity_threshold})的结果")
            return []

        print(f"* ✅ 初筛后剩余 {len(filtered_results)} 个文档")

        if self.enable_rerank and len(filtered_results) > 1:
            print(f"* 🔄 启动重排序...")
            docs_only = [doc for doc, _ in filtered_results]

            max_rerank = min(10, len(docs_only))
            final_doc = self.embedding_function.rerank_with_encoder(
                question_vector,
                docs_only[:max_rerank]
            )
            return final_doc[:5]  # 返回Top 5
        else:
            return [
                {
                    'corpus_id': i,
                    'score': score,
                    'text': doc.page_content[:500]  # 截断长文本
                }
                for i, (doc, score) in enumerate(filtered_results[:10])
            ]


    def delete_document(self, ids):
        self.vectors.delete(ids=ids)
        res = self.query_single_document(ids)
        if not res or len(res) == 0:
            print(f"✅id {ids} been deleted")
            return True
        else:
            print(f"✅id {ids} not been deleted")
            return False

    def list_documents_items(self):
        try:
            collections = self.vectors._collection
            include_fields = ['documents']
            results = collections.get(include=include_fields)
            items =[]
            for i, ids in enumerate(results.get('ids', [])):
                item = {
                    'id': ids,
                    'content': results.get('documents', []),
                }
                items.append(item)
            return items
        except Exception as e:
            print(str(e))
            raise e

    def query_single_document(self, uids):
            collections = self.vectors._collection
            items = []
            include_fields = ['documents']
            results = collections.get(ids=uids, include=include_fields)
            for i, ids in enumerate(results.get('ids', [])):
                item = {
                    'id': ids,
                    'content': results.get('documents', []),
                }
                items.append(item)
            return items







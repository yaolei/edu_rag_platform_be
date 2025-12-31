import os
from uuid import uuid4
from chromadb.config import Settings
from langchain_chroma import Chroma
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

    def query_by_question_vector_with_filter(self, question_vector, doc_types=None, top_k=5):
        if not doc_types:
            return self.query_by_question_vector(question_vector)

        try:
            collection = self.vectors._collection
            query_embedding = self.embedding_function.embed_query(question_vector)

            # 构建过滤条件
            if len(doc_types) == 1:
                where_filter = {"doc_type": doc_types[0]}
            else:
                where_filter = {"doc_type": {"$in": doc_types}}

            # 执行查询
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # 格式化结果
            formatted_results = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    doc_content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0

                    # 计算分数
                    score = 1.0 / (1.0 + distance)

                    formatted_results.append({
                        'corpus_id': i,
                        'score': score,
                        'text': doc_content,
                        'metadata': metadata
                    })

            return formatted_results

        except Exception as e:
            print(f"❌ 过滤查询失败: {str(e)}")
            return []

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
                    'text': doc.page_content  # 截断长文本
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

    # 在你的向量库管理类中添加一个验证方法
    def verify_doc_type_storage(self, doc_type_to_check=None):
        """
        验证doc_type是否正确存储在向量库中
        """
        collections = self.vectors._collection
        try:
            # 方法1：直接查询向量库的所有文档
            results = collections.get(include=["metadatas"])

            if results and results['metadatas']:
                print("🔍 验证向量库中的文档metadata:")
                print(f"总文档数: {len(results['metadatas'])}")

                # 统计各doc_type的数量
                doc_type_counts = {}
                for meta in results['metadatas']:
                    doc_type = meta.get('doc_type', 'unknown')
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

                print("📊 文档类型统计:")
                for doc_type, count in doc_type_counts.items():
                    print(f"  - {doc_type}: {count}个")

                # 如果需要检查特定类型
                if doc_type_to_check:
                    specific_docs = []
                    for i, meta in enumerate(results['metadatas']):
                        if meta.get('doc_type') == doc_type_to_check:
                            specific_docs.append(i)

                    print(f"\n📑 类型为 '{doc_type_to_check}' 的文档:")
                    print(f"  数量: {len(specific_docs)}")
                    if specific_docs:
                        print(f"  索引位置: {specific_docs[:5]}{'...' if len(specific_docs) > 5 else ''}")

            return True

        except Exception as e:
            print(f"❌ 验证失败: {str(e)}")
            return False






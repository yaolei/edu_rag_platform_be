import os
from uuid import uuid4
from chromadb.config import Settings
from langchain_chroma import Chroma
from numpy.distutils.from_template import list_re
from langchain_community.vectorstores.utils import filter_complex_metadata


class VectorStore:
    def __init__(self, embedding_function, collection_name='example_collection',
                 persist_directory='./chroma_langchain_db'):
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
        print(f"* 🙋 Question is {question_vector}")
        print(f"* 🚀 start the first similarity search by 10 items: ")
        result = self.vectors.similarity_search(question_vector, k=20)

        if len(result) == 0:
            print(f"* 📊 no data from the vector store")
            return False

        print(f"🚀 start the second similarity search by 10 items... ")
        final_doc = self.embedding_function.rerank_with_encoder(question_vector, result)
        # threshold = 7.5
        # filtered = [doc for doc in final_doc if doc["score"] >= threshold]
        # for rank in final_doc[:10]:
        #     print(f"-分数: ({rank['score']:.4f}): {rank['text'][:50]}, current score id is : {rank['corpus_id']}")

        return final_doc[:10]


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







import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Optional

BASE_MODEL_DIR = os.path.abspath("./models")
class EmbeddingData:
    def __init__(self, embedding_type='store'):
        self.model = SentenceTransformer(os.path.join(BASE_MODEL_DIR, "multi-qa-mpnet-base-cos-v1"), token=False)
        if embedding_type != 'store':
            self.model = CrossEncoder(os.path.join(BASE_MODEL_DIR, "ms-marco-MiniLM-L6-v2"), token=False)

    def embed_documents(self, text:List[str]) -> List[List[float]]:
        embeddings = self.model.encode(text, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_query(self, question) -> List[float]:
        embedding_model = SentenceTransformer(os.path.join(BASE_MODEL_DIR, "multi-qa-mpnet-base-cos-v1"), token=False)
        embeddings = embedding_model.encode([question], convert_to_tensor=False)
        return embeddings[0].tolist()

    def rerank_with_encoder(self, question, document):
        query_str = str(question)
        passage_texts = [doc.page_content for doc in document]

        ranks = self.model.rank(query_str, passage_texts, return_documents=True)

        print("Query:", question)
        for rank in ranks:
            print(f"{rank['score']:.2f}\t{passage_texts[rank['corpus_id']]}")

        return ranks






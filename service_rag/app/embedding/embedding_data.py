import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Optional
from langchain_core.embeddings import Embeddings

class EmbeddingData:
    def __init__(self, embedding_type='store'):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        # self.model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

    def test(self):
        query_embedding = self.model.encode("How big is London")
        passage_embeddings = self.model.encode([
            "London is known for its financial district",
            "London has 9,787,426 inhabitants at the 2011 census",
            "The United Kingdom is the fourth largest exporter of goods in the world",
        ])
        similarities = self.model.similarity(query_embedding, passage_embeddings)

        return similarities

    def test1(self):
        scores = self.model.predict([
            ("How many people live in Berlin?",
             "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
            ("How many people live in Berlin?", "Berlin is well known for its museums."),
        ])
        query = "How many people live in Berlin?"
        passages = [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin is well known for its museums.",
            "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
            "The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
            "The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
            "An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
            "Berlin is subdivided into 12 boroughs or districts (Bezirke).",
            "In 2015, the total labour force in Berlin was 1.85 million.",
            "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
            "Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
        ]
        ranks = self.model.rank(query, passages)
        print("Query:", query)
        for rank in ranks:
            print(f"{rank['score']:.2f}\t{passages[rank['corpus_id']]}")
if __name__ == '__main__':
    emb = EmbeddingData()
    emb.test1()





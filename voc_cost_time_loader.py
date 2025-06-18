import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

MODEL_NAME = "jhgan/ko-sroberta-multitask"

class VocCostTimeRAG:
    def __init__(self, csv_path="voc_cost_time.csv", model_name=MODEL_NAME):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer(model_name)
        self.keywords = self.df["불평키워드"].tolist()
        self.embeddings = self.model.encode(self.keywords, show_progress_bar=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def save(self, path="voc_cost_time_faiss.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "faiss_index": self.index,
                "embeddings": self.embeddings,
                "keywords": self.keywords,
                "df": self.df
            }, f)

    @staticmethod
    def load(path="voc_cost_time_faiss.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = VocCostTimeRAG.__new__(VocCostTimeRAG)
        obj.index = data["faiss_index"]
        obj.embeddings = data["embeddings"]
        obj.keywords = data["keywords"]
        obj.df = data["df"]
        obj.model = SentenceTransformer(MODEL_NAME)
        return obj

    def search(self, query, top_k=1):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            kw = self.keywords[idx]
            row = self.df.iloc[idx]
            results.append({
                "keyword": kw,
                "cost": row["예상 비용(원)"],
                "time": row["예상 기간(일)"],
                "distance": dist
            })
        return results

if __name__ == "__main__":
    rag = VocCostTimeRAG()
    rag.save()

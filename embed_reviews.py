import os
import json
import pandas as pd
from openai import OpenAI
import ast
import time

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # for OpenAI text-embedding-3-small


def compute_review_embeddings(csv_path="total_brand_reviews_df.csv", out_path=None, embedding_col="embedding"):
    """
    상품별 리뷰(content)에 대해 OpenAI 임베딩을 생성하여 embedding 컬럼으로 저장.
    이미 embedding 컬럼이 있으면 skip, 없으면 새로 생성.
    out_path를 지정하지 않으면 원본 CSV를 overwrite.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(csv_path)
    if embedding_col not in df.columns:
        df[embedding_col] = ""
    updated = False
    for i, row in df.iterrows():
        if pd.notnull(row.get(embedding_col, "")) and str(row[embedding_col]).strip() != "":
            continue  # 이미 임베딩 있음
        content = str(row["content"]).strip()
        if not content:
            continue
        try:
            resp = client.embeddings.create(
                input=content,
                model=EMBEDDING_MODEL
            )
            emb = resp.data[0].embedding
            df.at[i, embedding_col] = json.dumps(emb)
            updated = True
        except Exception as e:
            print(f"임베딩 실패 (index={i}): {e}")
            time.sleep(1)
    if updated:
        save_path = out_path or csv_path
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"임베딩 컬럼 저장 완료: {save_path}")
    else:
        print("새로 추가된 임베딩 없음.")

if __name__ == "__main__":
    compute_review_embeddings()

import os
import json
import pandas as pd
from openai import OpenAI
import ast
import time
import csv

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
        # CSV 저장 시 quoting 옵션을 명확히 지정하여 embedding 컬럼이 셀을 침범하지 않게 함
        df.to_csv(save_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"임베딩 컬럼 저장 완료: {save_path}")
        # 저장 후 행 수 검증 (디버깅용)
        try:
            df_check = pd.read_csv(save_path)
            if len(df) != len(df_check):
                print(f"[경고] 저장 전후 행 수 불일치! 저장 전: {len(df)}, 저장 후: {len(df_check)}")
            else:
                print(f"[검증] 저장 전후 행 수 일치: {len(df)}")
        except Exception as e:
            print(f"[오류] 저장 후 CSV 재로딩 실패: {e}")
    else:
        print("새로 추가된 임베딩 없음.")

if __name__ == "__main__":
    compute_review_embeddings()

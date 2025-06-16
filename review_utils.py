import os
import json
from openai import OpenAI

# 기본 키워드 추출 함수 (백업)
def extract_keywords(text):
    # 아주 단순한 예시: 명사만 추출 (실제 구현은 필요에 따라 교체)
    # 여기서는 키워드 추출 실패 시 빈 리스트 반환
    return []

# 상위 5개 키워드 추출 함수 (공통)
def get_top5_keywords_from_list(keyword_list):
    """
    리스트에서 빈도수 기준 상위 5개 키워드 반환 (키워드만 리스트로)
    """
    from collections import Counter
    if not keyword_list:
        return []
    return [kw for kw, _ in Counter(keyword_list).most_common(5)]

# OpenAI API 기반 키워드 추출 함수
def extract_keywords_with_openai(review_text):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return extract_keywords(review_text)

    client = OpenAI(api_key=api_key)

    few_shot_messages = [
        {"role": "user",
         "content": "리뷰: 직원이 친절했지만 음식이 너무 늦게 나왔어요."},
        {"role": "assistant",
         "content": '{"keywords":["음식 지연"]}'},
        {"role": "user",
         "content": "리뷰: 가격이 비싸고 품질이 기대 이하여서 실망했습니다."},
        {"role": "assistant",
         "content": '{"keywords":["가격","품질"]}'},
    ]

    system_keyword = (
        "너는 비즈니스 의사결정을 돕는 분석가다. "
        "아래 리뷰에서 고객 컴플레인 핵심 키워드 3~6개를 추출한다. "
        "각 키워드는 한국어 명사구여야 한다. "
        "출력은 다음 JSON 스키마를 반드시 따른다.\n\n"
        '{\n  "keywords": ["키워드1", "키워드2", ...]\n}\n'
        "다른 텍스트(접두어·주석·마침표 등)를 절대 포함하지 마라."
    )
    user_keyword = f"리뷰: {review_text}"

    messages = (
        [{"role": "system", "content": system_keyword}]
        + few_shot_messages
        + [{"role": "user", "content": user_keyword}]
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=128,
        )
        data = json.loads(resp.choices[0].message.content)
        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            raise ValueError("`keywords` 필드 형식 오류")
        return keywords
    except Exception:
        return extract_keywords(review_text)


def generate_response_with_openai(review_text: str) -> str:
    """
    불평·불만 리뷰를 입력하면
    1) 먼저 진심 어린 사과
    2) 고객이 언급한 문제점을 인지했다는 표현
    3) 동일 문제가 반복되지 않도록 제품/서비스 개선에 힘쓰겠다는 약속
    만을 한국어로 작성해 돌려준다. (보상·쿠폰·환불·교환 금지)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # API 키가 없으면 아주 간단한 고정 답변
        return "불편을 드려 정말 죄송합니다. 말씀해 주신 부분을 면밀히 살펴 개선하겠습니다."

    client = OpenAI(api_key=api_key)

    # 1단계: 리뷰에서 핵심 키워드 추출
    keywords = extract_keywords_with_openai(review_text)

    # 2단계: 사과·대응 답변 생성
    system_prompt = (
        "너는 고객센터 매니저다. 반드시:\n"
        "1) 진심으로 사과한다. (첫 문장)\n"
        "2) 고객이 지적한 구체적인 문제점을 1~2문장 안에서 언급한다."
        "3) 쿠폰, 환불, 교환, 금전 보상 약속은 절대 하지 않는다.\n"
        "4) 해당 문제를 해결하고 서비스·상품 품질을 개선하겠다고 약속하며 마친다.\n"
        "5) 3~5문장, 丁寧체(존댓말)를 사용한다."
    )

    user_prompt = (
        f"[리뷰 전문]\n{review_text}\n\n"
        f"[추출 키워드]\n{', '.join(keywords) if keywords else '(추출 실패)'}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    return completion.choices[0].message.content.strip()

# ────────────────────────────────────────────────────────────────
# 상품별 키워드 의미(RAG) 및 키워드/비용/시간 업데이트 함수
import pandas as pd
import ast
import numpy as np
import json

def get_keyword_meaning_rag(goodsNo, keyword, csv_path='total_brand_reviews_df.csv', rag_limit=10):
    """
    해당 상품(goodsNo)의 리뷰 중 keyword와 의미적으로 유사한 리뷰(top-N)를 벡터 유사도 기반으로 추출하여 RAG Q&A.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI API 키가 없습니다."
    client = OpenAI(api_key=api_key)
    EMBEDDING_MODEL = "text-embedding-3-small"
    # 1. 상품별 리뷰와 임베딩 추출
    df = pd.read_csv(csv_path)
    df_prod = df[df['goodsNo'].astype(str) == str(goodsNo)].copy()
    # 2. 리뷰 임베딩 준비
    contents = df_prod['content'].tolist()
    embeddings = []
    for emb_str in df_prod.get('embedding', []):
        try:
            if pd.isnull(emb_str) or str(emb_str).strip() == '':
                embeddings.append(None)
            else:
                embeddings.append(np.array(json.loads(emb_str)))
        except Exception:
            embeddings.append(None)
    # 3. 키워드 임베딩 생성
    try:
        resp = client.embeddings.create(
            input=keyword,
            model=EMBEDDING_MODEL
        )
        keyword_emb = np.array(resp.data[0].embedding)
    except Exception as e:
        return f"키워드 임베딩 실패: {e}"
    # 4. 코사인 유사도 계산 및 top-N 리뷰 추출
    sims = []
    for emb in embeddings:
        if emb is None:
            sims.append(-1)
        else:
            sim = np.dot(keyword_emb, emb) / (np.linalg.norm(keyword_emb) * np.linalg.norm(emb) + 1e-8)
            sims.append(sim)
    top_idx = np.argsort(sims)[::-1][:rag_limit]
    keyword_reviews = [contents[i] for i in top_idx if sims[i] > 0]
    if not keyword_reviews:
        return "해당 키워드와 의미적으로 유사한 리뷰가 없습니다. (임베딩 기반)"
    # 5. RAG prompt 구성 및 GPT 호출
    prompt = f"다음은 상품 리뷰입니다. 키워드 '{keyword}'가 어떤 의미로 사용되는지 설명해 주세요.\n\n"
    for i, review in enumerate(keyword_reviews, 1):
        prompt += f"[{i}] {review}\n"
    prompt += f"\n이 키워드가 이 상품에서 어떤 맥락으로 쓰였는지 간단히 Q&A 방식으로 설명해 주세요. (한국어로, 2~3문장)"
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "너는 상품 리뷰 분석 전문가다."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def update_keyword_info(goodsNo, old_keyword, new_keyword, cost, time, csv_path='total_brand_reviews_df.csv'):
    """
    해당 상품(goodsNo)의 real_keywords_all, real_keywords_dmg_dict에서 old_keyword를 new_keyword로 완전 교체,
    비용(cost, 원), 시간(time, 시간)은 keyword_plan_info 컬럼의 dict로 저장.
    """
    import ast
    import json
    import pandas as pd
    df = pd.read_csv(csv_path)

    # ── 1. keyword_plan_info 컬럼이 없으면 생성 및 마이그레이션 ──
    if 'keyword_plan_info' not in df.columns:
        # 기존 cost/processing_time 컬럼이 있으면 마이그레이션
        plan_info_col = []
        for i, row in df.iterrows():
            try:
                kw_list = ast.literal_eval(row['real_keywords_all']) if pd.notnull(row['real_keywords_all']) else []
            except Exception:
                kw_list = []
            plan_dict = {}
            for kw in kw_list:
                c = row['cost'] if 'cost' in df.columns and pd.notnull(row['cost']) else ''
                t = row['processing_time'] if 'processing_time' in df.columns and pd.notnull(row['processing_time']) else ''
                plan_dict[kw] = {"cost": c, "time": t}
            plan_info_col.append(json.dumps(plan_dict, ensure_ascii=False))
        df['keyword_plan_info'] = plan_info_col

    # 상품별 행 인덱스
    idxs = df[df['goodsNo'].astype(str) == str(goodsNo)].index
    for idx in idxs:
        # real_keywords_all: 문자열 -> 리스트
        kw_list = ast.literal_eval(df.at[idx, 'real_keywords_all']) if pd.notnull(df.at[idx, 'real_keywords_all']) else []
        # 완전 교체
        kw_list = [new_keyword if kw == old_keyword else kw for kw in kw_list]
        df.at[idx, 'real_keywords_all'] = str(kw_list)
        # real_keywords_dmg_dict: 문자열 -> dict
        dmg_dict = ast.literal_eval(df.at[idx, 'real_keywords_dmg_dict']) if pd.notnull(df.at[idx, 'real_keywords_dmg_dict']) else {}
        # 키워드 교체 (값은 old_keyword의 값 유지)
        if old_keyword in dmg_dict:
            dmg_dict[new_keyword] = dmg_dict.pop(old_keyword)
        df.at[idx, 'real_keywords_dmg_dict'] = str(dmg_dict)
        # keyword_plan_info: 문자열 -> dict
        plan_info = {}
        if pd.notnull(df.at[idx, 'keyword_plan_info']):
            try:
                plan_info = json.loads(df.at[idx, 'keyword_plan_info'])
            except Exception:
                plan_info = {}
        # 기존 값 옮기기 (old_keyword -> new_keyword)
        if old_keyword in plan_info:
            plan_info[new_keyword] = plan_info.pop(old_keyword)
        # 값 저장
        plan_info[new_keyword] = {"cost": cost, "time": time}
        df.at[idx, 'keyword_plan_info'] = json.dumps(plan_info, ensure_ascii=False)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return True
import os
import json
from openai import OpenAI

# 기본 키워드 추출 함수 (백업)
def extract_keywords(text):
    # 아주 단순한 예시: 명사만 추출 (실제 구현은 필요에 따라 교체)
    # 여기서는 키워드 추출 실패 시 빈 리스트 반환
    return []

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

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
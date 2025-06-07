import json
import requests
from typing import Optional

URL     = "https://www.oliveyoung.co.kr/store/search/NewCateSearchApi.do"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.oliveyoung.co.kr/store/main/main.do",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
}


def safe_json(resp: requests.Response) -> dict:
    """
    getSearchMain.do 응답은 JSON 안에 JSON 문자열을 한 번 더
    감싸는(이중 인코딩) 경우가 있어요. 두 번 디코딩해 dict로 돌려줍니다.
    """
    try:
        data = resp.json()
    except ValueError:
        return {}

    if isinstance(data, str):          # 이중 인코딩 ⇒ 한 번 더
        try:
            data = json.loads(data)
        except ValueError:
            return {}
    return data if isinstance(data, dict) else {}


def get_brand_code(brand_name: str) -> Optional[str]:
    """
    Olive Young 검색 메인 API에서 'cateBrand' 배열을 읽어
    브랜드 코드를(id) 반환한다. 못 찾으면 None.
    """
    resp = requests.get(URL, params={"query": brand_name},
                        headers=HEADERS, timeout=5)
    resp.raise_for_status()
    data = safe_json(resp)
    for category_item  in data.get("category", []):
        if category_item.get("name") == "cateBrand":
            cate_brand_list = category_item.get("cateBrand", [])
            if cate_brand_list:
                brand_code = cate_brand_list[0].get("id")
                return brand_code
    return None


if __name__ == "__main__":
    for bn in ["토리든", "록시땅", "닥터지"]:
        print(f"{bn} → {get_brand_code(bn)}")

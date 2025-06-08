import re, math, time, random, requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import json
import requests
from typing import Optional


BASE_URL = "https://www.oliveyoung.co.kr/store/display/getBrandShopDetailGoodsPagingAjax.do"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.oliveyoung.co.kr/",
    "X-Requested-With": "XMLHttpRequest",
}

BRANDCODE_URL     = "https://www.oliveyoung.co.kr/store/search/NewCateSearchApi.do"
BRANDCODE_HEADERS = {
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


def olive_get_brand_code(brand_name: str) -> Optional[str]:
    """
    Olive Young 검색 메인 API에서 'cateBrand' 배열을 읽어
    브랜드 코드를(id) 반환한다. 못 찾으면 None.
    """
    resp = requests.get(BRANDCODE_URL, params={"query": brand_name},
                        headers=BRANDCODE_HEADERS, timeout=5)
    resp.raise_for_status()
    data = safe_json(resp)
    for category_item  in data.get("category", []):
        if category_item.get("name") == "cateBrand":
            cate_brand_list = category_item.get("cateBrand", [])
            if cate_brand_list:
                brand_code = cate_brand_list[0].get("id")
                return brand_code
    return None



def fetch_html(brand, page, rows=24,
               disp_cat="", flt_disp_cat=""):          # ← 기본값을 "" 로!
    """브랜드 전체상품 탭 HTML 조각 반환"""
    params = {
        "onlBrndCd": brand,
        "pageIdx": page,
        "dispCatNo": disp_cat,
        "fltDispCatNo": flt_disp_cat,
        "prdSort": "01",          # 정렬 방식(01=인기순), 전체/베스트 구분과 무관
        "rowsPerPage": rows,
        "trackingCd": "",
    }
    r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text

def parse_products(html):
    """HTML → [{goodsNo, name, rating}, …]"""
    soup = BeautifulSoup(html, "lxml")
    items = []
    for li in soup.select("ul.prod-list li[data-goods-idx]"):
        # 상품번호: 상세 링크 href 속 goodsNo= 파라미터에서 추출
        a = li.select_one("a[href*='goodsNo=']")
        m = re.search(r'goodsNo=([A-Z0-9]+)', a["href"])
        goods_no = m.group(1) if m else None

        # 상품명
        name = li.select_one(".prod-name").get_text(strip=True)

        # 별점(없으면 None)
        rt = li.select_one(".rating .point")
        rating = float(rt.text) if rt else None

        items.append({"goodsNo": goods_no, "name": name, "rating": rating, "link": a["href"]})
    return items

def olive_products_crawl(brand, rows=24,
                    disp_cat="", flt_disp_cat="",
                    delay=(0.3, 0.8), limit="all"):
    """브랜드 전체상품(모든 페이지) DataFrame 반환. limit: int 또는 'all'"""
    first_html = fetch_html(brand, 1, rows, disp_cat, flt_disp_cat)
    soup = BeautifulSoup(first_html, "lxml")

    # 총 상품 수(id="totCntFmt" input)에 기댈 수 있으면 이용
    total_tag = soup.select_one("#totCntFmt")
    if total_tag and total_tag.has_attr("value"):
        total_cnt = int(total_tag["value"])
        total_pages = math.ceil(total_cnt / rows)
    else:
        # 안전책: 토탈을 못 찾으면 '빈 페이지 나올 때까지' 방식으로
        total_pages = 1000    # 실질적으로 충분히 큰 값

    data = parse_products(first_html)
    if limit != "all":
        try:
            limit = int(limit)
        except Exception:
            limit = "all"
        if isinstance(limit, int) and len(data) >= limit:
            return pd.DataFrame(data[:limit])

    # 2page~Npage
    for page in range(2, total_pages + 1):
        html = fetch_html(brand, page, rows, disp_cat, flt_disp_cat)
        page_items = parse_products(html)
        if not page_items:             # 더 이상 상품이 없으면 종료
            break
        data.extend(page_items)
        if limit != "all" and isinstance(limit, int) and len(data) >= limit:
            return pd.DataFrame(data[:limit])
        time.sleep(random.uniform(*delay))

    return pd.DataFrame(data if limit == "all" else data[:limit])

# ───────────────────────── 리뷰 크롤러 함수 ─────────────────────────

def olive_reviews_crawl(goods_nos, limit="all"):
    import pandas as pd
    # 여러 상품 번호 처리
    if isinstance(goods_nos, (list, tuple, set)):
        all_reviews = []
        for goods_no in goods_nos:
            print(f"크롤링: {goods_no}")
            olive_reviews_df = olive_reviews_crawl(goods_no, limit=limit)
            print(f"{goods_no} 리뷰 수집 결과: {len(olive_reviews_df)}개")
            all_reviews.append(olive_reviews_df)
        if all_reviews:
            total_reviews_df = pd.concat(all_reviews, ignore_index=True)
            return total_reviews_df
        else:
            return pd.DataFrame(columns=['goodsNo', 'nickname', 'profile_present', 'top_rank_present', 'badges_present', 'score', 'date', 'content', 'help_cnt', 'photo_present'])
    # 단일 상품 번호 처리(기존 로직)
    goods_no = goods_nos
    url = f"https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo={goods_no}"
    options = EdgeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()), options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    actions = ActionChains(driver)
    results = []
    try:
        collected = 0
        # 1. 리뷰 탭 클릭
        review_tab = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "goods_reputation")))
        if "리뷰" in review_tab.text:
            review_tab.click()
            time.sleep(1)
        # 2. 리뷰 검색 필터 클릭
        filter_btn = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "btnOption")))
        if "리뷰 검색 필터" in filter_btn.text:
            filter_btn.click()
            time.sleep(1)
        # 3. 1점, 2점, 3점 체크박스 클릭 (searchPoint 내부에서 텍스트로 찾기)
        point_area = wait.until(EC.presence_of_element_located((By.ID, "searchPoint")))
        for score in ["1점", "2점", "3점"]:
            found = False
            for elem in point_area.find_elements(By.XPATH, ".//*"):
                if elem.text.strip() == score:
                    elem.click()
                    time.sleep(0.5)
                    found = True
                    break
            if not found:
                print(f"{score} 체크박스 없음")
        # 4. 적용 버튼 클릭
        apply_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnFilterConfirm")))
        if "적용" in apply_btn.text:
            apply_btn.click()
            time.sleep(2)
        # 5~6. 리뷰 리스트 및 페이지네이션 반복
        page = 1
        while True:
            # 5. 리뷰 정보 추출
            try:
                # robust하게 리뷰 리스트 ul을 XPath로 찾음
                review_ul = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="gdasList"]')))
                soup = BeautifulSoup(review_ul.get_attribute('innerHTML'), 'lxml')

                reviews = []

                for li in soup.body.find_all('li', recursive=False):
                    # 1. 작성자 프로필 유무 ─ 기본 이미지가 아니라면 True
                    profile_img = li.select_one('div.user .thum img')
                    profile_present = bool(profile_img) and 'my_picture_base.jpg' not in (profile_img.get('src') or '')

                    # 2-a. 닉네임
                    nickname = li.select_one('p.info_user a.id')
                    nickname = nickname.get_text(strip=True) if nickname else ''

                    # 2-b. 추가 사용자 정보(예: TOP 순위, 배지 등)
                    top_rank_present = li.select_one('p.info_user a.top') is not None
                    badges_present = bool(li.select('div.badge a'))

                    # 3. 리뷰 별점(1‒5점)
                    score_span = li.select_one('span.review_point span.point')
                    score = None
                    if score_span:
                        txt = score_span.get_text(strip=True)
                        m   = re.search(r'(\d)점$', txt)          # '5점만점에 3점' → 3
                        if m:
                            score = int(m.group(1))
                        else:                                     # 혹시 텍스트가 없으면 style width % 로 계산
                            width = score_span.get('style', '')
                            mw    = re.search(r'width\s*:\s*(\d+)', width)
                            if mw: score = round(int(mw.group(1)) / 20)

                    # 4. 리뷰 작성일(YYYY.MM.DD)
                    date = li.select_one('span.date')
                    date = date.get_text(strip=True) if date else ''

                    # 5. 리뷰 내용
                    content = li.select_one('div.txt_inner')
                    content = content.get_text("\n", strip=True) if content else ''

                    # 6. "도움이 돼요" 숫자
                    help_num = li.select_one('button.btn_recom span.num')
                    help_cnt = int(help_num.get_text(strip=True)) if help_num else 0

                    # 7. 사진 유무
                    photo_present = li.select_one('div.review_thum') is not None

                    reviews.append(
                            {
                                'goodsNo'         : goods_no,
                                'nickname'        : nickname,
                                'profile_present' : profile_present,
                                'top_rank_present' : top_rank_present,  # True/False
                                'badges_present'   : badges_present,    # True/False
                                'score'           : score,
                                'date'            : date,
                                'content'         : content,
                                'help_cnt'        : help_cnt,
                                'photo_present'   : photo_present,
                            })
                    collected += 1
                    if limit != "all" and isinstance(limit, int) and collected >= limit:
                        break
                if limit != "all" and isinstance(limit, int) and collected >= limit:
                    results.extend(reviews)
                    break

                results.extend(reviews)

            except Exception as e:
                results.extend(reviews)
                print(f"리뷰 추출 실패: {e}")
                break
            # 6. robust 페이지네이션 처리
            paging_area = driver.find_element(By.XPATH, '//*[@id="gdasContentsArea"]//div[contains(@class, "pageing")]')
            # 현재 페이지 번호 추출
            current_page = int(paging_area.find_element(By.XPATH, './/strong[@title="현재 페이지"]').text)
            next_page_no = current_page + 1
            next_btn = None
            try:
                # 다음 페이지 번호가 있으면 클릭
                next_btn = paging_area.find_element(By.XPATH, f'.//a[@data-page-no="{next_page_no}"]')
            except Exception:
                # 다음 페이지 번호가 없으면 "다음 10 페이지" 버튼이 있는지 확인
                try:
                    next_btn = paging_area.find_element(By.XPATH, './/a[contains(@class, "next")]')
                except Exception:
                    next_btn = None
            if next_btn:
                next_btn.click()
                time.sleep(2)
                page += 1
            else:
                break # 더 이상 페이지 없음
    finally:
        driver.quit()
    # 중복 nickname 제거 (뒤에 있는 것 제거, 앞에 있는 것만 남김)
    seen_nicknames = set()
    unique_results = []
    for review in reversed(results):
        if review['nickname'] not in seen_nicknames:
            unique_results.append(review)
            seen_nicknames.add(review['nickname'])
    results = list(reversed(unique_results))
    if limit != "all" and isinstance(limit, int):
        results = results[:limit]
    # DataFrame으로 변환, goodsNo가 첫 번째 열이 되도록
    if results:
        columns = ['goodsNo'] + [k for k in results[0].keys() if k != 'goodsNo']
        return pd.DataFrame(results, columns=columns)
    else:
        return pd.DataFrame(columns=['goodsNo', 'nickname', 'profile_present', 'top_rank_present', 'badges_present', 'score', 'date', 'content', 'help_cnt', 'photo_present'])

# ───────────────────────── 실행 예시 ─────────────────────────
if __name__ == "__main__":
    brand_code = olive_get_brand_code("가그린")
    prod_limit = 3  # 원하는 상품 개수, 'all'이면 전체
    review_limit = 5 # 원하는 리뷰 개수, 'all'이면 전체
    olive_products_df = olive_products_crawl(brand_code, limit=prod_limit)
    print(f"총 {len(olive_products_df)}개")
    print(olive_products_df)
    # 리뷰 크롤링 예시 (한 번에 처리)
    goods_nos = olive_products_df['goodsNo'].tolist()
    olive_total_reviews_df = olive_reviews_crawl(goods_nos, limit=review_limit)
    print(f"총 리뷰 수집 결과: {len(olive_total_reviews_df)}개")
    print(olive_total_reviews_df)

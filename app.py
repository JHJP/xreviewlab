import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, session, jsonify
import numpy as np
# ───────────────────────────────────────────────────────────────
# 📊  Brand-reputation research → ₩ per DAMAGE-UNIT
# ----------------------------------------------------------------
# Why 12 500 ₩ ?  (sources ⬇️)
# • Harvard Business School (Luca 2016): every ★ on Yelp shifts revenue
#   by ≈ 5-9 %. 
# • Gominga “Online Review Stats 2024”: one bad review can scare away
#   ~30 shoppers & 40 % of prospects. 
# • Statistics Korea (2024): avg e-commerce basket ≈ ₩51 800.
#
# Worst-case loss for a *high-damage* review:
#     30 customers × ₩51 800  ≈ ₩1 554 000
# The same review’s `compute_damage()` score peaks ≈ 125 units
# (explicit warning + long text + aged).                ⟶  1 unit
#     1 554 000 / 125  ≈ ₩12 400  → rounded → **₩12 500**
# ───────────────────────────────────────────────────────────────
WON_PER_DAMAGE_UNIT: float = 12_500

app = Flask(__name__)
app.secret_key = 'replace_this_with_a_random_secret_1234'

# 진행상황 초기화
def init_progress(total):
    session['progress_current'] = 0
    session['progress_total'] = total
    session.modified = True

def increment_progress():
    session['progress_current'] = session.get('progress_current', 0) + 1
    session.modified = True

def get_progress():
    return session.get('progress_current', 0), session.get('progress_total', 1)

@app.route('/progress')
def progress():
    current, total = get_progress()
    logging.info('[progress API] %s %s', current, total)  # 서버 콘솔에 직접 찍기
    return jsonify({'current': current, 'total': total})

@app.route("/", methods=["GET", "POST"])
def index():
    # ── 1. show welcome/tutorial overlay only the first time ───────────────
    if request.method == "POST":
        session["tutorial_seen"] = True 
    tutorial_mode = not session.get("tutorial_seen", False)

    # ── 2. POST: 기존 크롤링/저장 로직 유지 ──
    import time
    if request.method == "POST":
        t_post_start = time.time()
        brand_name = request.form.get("brand_name")
        brand_url = request.form.get("brand_url")
        products = []
        olive_products_df = None
        if brand_name:
            try:
                logging.info("[TIMER] 크롤러 import 및 olive_get_brand_code 시작")
                t_crawler_start = time.time()
                from crawlers.olive_crawler import olive_get_brand_code, olive_products_crawl
                brand_code = olive_get_brand_code(brand_name)
                logging.info(f"[TIMER] olive_get_brand_code 완료: {time.time() - t_crawler_start:.2f}s")
                t_crawl_all_start = time.time()
                # 브랜드의 모든 상품 가져오기
                all_products = []
                olive_products_df = olive_products_crawl(brand_code, limit=3)
                olive_products_df["platform"] = "olive"
                all_products.append(olive_products_df)
                if all_products:
                    import pandas as pd
                    total_products_df = pd.concat(all_products, ignore_index=True)
                logging.info(f"[TIMER] olive_products_crawl 완료: {time.time() - t_crawl_all_start:.2f}s")
                t_loop_start = time.time()
                # 전체 상품 정보 저장
                # 상품별 damage 총합 계산 (CSV가 있으면 활용)
                import os
                import pandas as pd
                csv_path = os.path.join(os.path.dirname(__file__), 'total_brand_reviews_df.csv')
                damage_sum_map = {}
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if 'prd_name' in df.columns and 'damage' in df.columns:
                        for prd_name, group in df.groupby('prd_name'):
                            damage_sum_map[prd_name] = group['damage'].sum()
                for _, row in total_products_df.iterrows():
                    damage_sum = damage_sum_map.get(row['name'], None)
                    products.append({
                        'goodsNo': row['goodsNo'],
                        'name': row['name'],
                        'rating': row['rating'],
                        'link': row['link'] if 'link' in row else '',
                        'platform': row['platform'],
                        'damage_sum': damage_sum
                    })
                logging.info(f"[TIMER] 상품 반복문 완료: {time.time() - t_loop_start:.2f}s")
            except Exception as e:
                logging.error(f"[ERROR] 브랜드 코드 추출 실패: {e}")
        else:
            # 브랜드명 없이 전체 상품 fallback 크롤링
            try:
                logging.info("[TIMER] fallback: olive_products_crawl(None) 전체 상품 크롤링 시작")
                from crawlers.olive_crawler import olive_products_crawl
                olive_products_df = olive_products_crawl(None, limit=10)
                olive_products_df["platform"] = "olive"
                total_products_df = olive_products_df
                logging.info(f"[TIMER] fallback: olive_products_crawl(None) 완료")
                t_loop_start = time.time()
                products = []
                for _, row in total_products_df.iterrows():
                    products.append({
                        'goodsNo': row['goodsNo'],
                        'name': row['name'],
                        'rating': row['rating'],
                        'link': row['link'] if 'link' in row else '',
                        'platform': row['platform'],
                        'damage_sum': None
                    })
                logging.info(f"[TIMER] fallback 상품 반복문 완료: {time.time() - t_loop_start:.2f}s")
            except Exception as e:
                logging.error(f"[ERROR] fallback 전체상품 크롤링 실패: {e}")
        session['total_products'] = products
        session['olive_products_df'] = olive_products_df.to_dict() if olive_products_df is not None else None
        # ---- 전체 상품 리뷰 수집 및 합치기 ----
        import pandas as pd
        from crawlers.olive_crawler import olive_reviews_crawl
        from review_utils import extract_keywords_with_openai
        all_reviews = []
        for prod in products:
            goodsNo = prod['goodsNo']
            platform = prod.get('platform', '')
            try:
                if platform == 'olive':
                    reviews_df = olive_reviews_crawl([goodsNo], limit=5)
                    if reviews_df is not None and not reviews_df.empty:
                        reviews_df["word_count"] = reviews_df["content"].str.split().apply(len)
                        from routes.damage_calc import calculate_damage
                        reviews_df = calculate_damage(reviews_df)
                        reviews_df["keywords"] = reviews_df["content"].apply(extract_keywords_with_openai)
                        reviews_df["prd_link"] = prod.get('link', '')
                        reviews_df["prd_name"] = prod.get('name', '')
                        reviews_df["rating"] = prod.get('rating', '')
                        reviews_df["prd_platform"] = platform
                        all_reviews.append(reviews_df)
            except Exception as e:
                logging.error(f"[ERROR] 리뷰 수집 실패: goodsNo={goodsNo}, error={e}")
        if all_reviews:
            total_brand_reviews_df = pd.concat(all_reviews, ignore_index=True)
            total_brand_reviews_df.to_csv('total_brand_reviews_df.csv', index=False, encoding='utf-8-sig')
            logging.info(f"[INFO] total_brand_reviews_df.csv 저장 완료: {len(total_brand_reviews_df)}건")
            # --- 키워드 클러스터링 및 damage 집계 실행 ---
            from keyword_clustering_and_damage import run_keyword_clustering_and_damage
            run_keyword_clustering_and_damage('total_brand_reviews_df.csv')
        else:
            logging.info("[INFO] 수집된 리뷰 없음. total_brand_reviews_df.csv 미생성")
        logging.info(f"[TIMER] 전체 POST 처리 완료: {time.time() - t_post_start:.2f}s")
        return render_template(
            "index.html",
            products=products,
            tutorial_mode=tutorial_mode
        )

    # ── 3. GET: CSV에서 상품별 테이블 직접 집계 ──
    import os
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), 'total_brand_reviews_df.csv')
    products = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        grouped = df.groupby('prd_name').agg({
            'rating': 'mean',
            'damage': 'sum',
            'goodsNo': 'first',
            'prd_link': 'first',
            'real_keywords_all': 'first'
        }).reset_index()
        from review_utils import get_top5_keywords_from_list
        import ast
        for _, row in grouped.iterrows():
            real_keywords_all = []
            val = row.get('real_keywords_all', None)
            if pd.notnull(val):
                if isinstance(val, str):
                    try:
                        real_keywords_all = ast.literal_eval(val)
                    except Exception:
                        real_keywords_all = []
                elif isinstance(val, list):
                    real_keywords_all = val
            top5_keywords = get_top5_keywords_from_list(real_keywords_all)
            products.append({
                'goodsNo': row['goodsNo'],
                'name': row['prd_name'],
                'rating': round(row['rating'], 2) if pd.notnull(row['rating']) else '',
                'damage_sum': round(row['damage'], 2) if pd.notnull(row['damage']) else 0,
                'link': row['prd_link'],
                'top5_keywords': top5_keywords
            })
    return render_template(
        "index.html",
        products=products,
        tutorial_mode=tutorial_mode
    )

# VoC 듣기: 강제 새로고침용 라우트 (프론트에서 버튼으로 호출)
@app.route("/voc_listen", methods=["POST"])
def voc_listen():
    from flask import request
    brand_name = request.form.get("brand_name")
    brand_url = request.form.get("brand_url")
    # Run the main index logic (crawling, saving, etc)
    with app.test_request_context('/', method='POST', data={'brand_name': brand_name or '', 'brand_url': brand_url or ''}):
        response = index()
    # After crawling, run embedding computation
    from embed_reviews import compute_review_embeddings
    try:
        compute_review_embeddings()
    except Exception as e:
        import logging
        logging.error(f"[임베딩 오류] {e}")
    return response


# Blueprint 등록
from routes.auto_reply import auto_reply_bp
from routes.product import product_bp
app.register_blueprint(auto_reply_bp)
app.register_blueprint(product_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

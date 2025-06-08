import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, session, jsonify

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
    print('[progress API]', current, total)  # 서버 콘솔에 직접 찍기
    return jsonify({'current': current, 'total': total})

@app.route("/", methods=["GET", "POST"])
def index():
    # ── 1. show welcome/tutorial overlay only the first time ───────────────
    if request.method == "POST":
        session["tutorial_seen"] = True 
    tutorial_mode = not session.get("tutorial_seen", False)

    # ── 2. skeletons for the template (stay None on plain GET)
    products = None
    olive_products_df = None

    # ── 3. Handle the POST: run the optimiser & build the dashboard
    import time
    if request.method == "POST":
        t_post_start = time.time()
        brand_name = request.form.get("brand_name")
        brand_url = request.form.get("brand_url")
        products = []
        olive_products_df = None
        if brand_name:
            try:
                print("[TIMER] 크롤러 import 및 olive_get_brand_code 시작")
                t_crawler_start = time.time()
                from crawlers.olive_crawler import olive_get_brand_code, olive_products_crawl
                brand_code = olive_get_brand_code(brand_name)
                print(f"[TIMER] olive_get_brand_code 완료: {time.time() - t_crawler_start:.2f}s")
                t_crawl_all_start = time.time()
                olive_products_df = olive_products_crawl(brand_code, limit=3)
                print(f"[TIMER] olive_products_crawl 완료: {time.time() - t_crawl_all_start:.2f}s")
                t_loop_start = time.time()
                # 전체 상품 정보 저장
                for _, row in olive_products_df.iterrows():
                    # 리뷰 없는 상품은 나중에 처리 위해 goodsNo도 저장
                    products.append({
                        'goodsNo': row['goodsNo'],
                        'name': row['name'],
                        'rating': row['rating'],
                        'link': row['link'] if 'link' in row else '',
                    })
                print(f"[TIMER] 상품 반복문 완료: {time.time() - t_loop_start:.2f}s")
            except Exception as e:
                print(f"[ERROR] 브랜드 코드 추출 실패: {e}")
        # 세션에 상품/리뷰 데이터 임시 저장 (리뷰 없는 상품 비활성화 위해)
        session['total_products'] = products
        session['olive_products_df'] = olive_products_df.to_dict() if olive_products_df is not None else None
        print(f"[TIMER] 전체 POST 처리 완료: {time.time() - t_post_start:.2f}s")
        return render_template(
            "index.html",
            products=products,
            tutorial_mode=tutorial_mode
        )
    # GET 요청 시에도 세션 저장된 상품 전달
    products = session.get('total_products')
    return render_template(
        "index.html",
        products=products,
        tutorial_mode=tutorial_mode
    )


# Blueprint 등록
from routes.auto_reply import auto_reply_bp
from routes.product import product_bp
app.register_blueprint(auto_reply_bp)
app.register_blueprint(product_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

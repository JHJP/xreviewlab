import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, session, send_file, redirect, url_for, jsonify
import pandas as pd

# 추가: 워드클라우드 관련
from collections import Counter
import io, base64
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import openai

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
    wordcloud_url = None
    total_products_df = None

    # ── 3. Handle the POST: run the optimiser & build the dashboard
    import time
    if request.method == "POST":
        t_post_start = time.time()
        brand_name = request.form.get("brand_name")
        brand_url = request.form.get("brand_url")
        products = []
        total_products_df = None
        if brand_name:
            try:
                print("[TIMER] 크롤러 import 및 get_brand_code 시작")
                t_crawler_start = time.time()
                from crawlers.olive_crawler import get_brand_code, crawl_brand_all
                brand_code = get_brand_code(brand_name)
                print(f"[TIMER] get_brand_code 완료: {time.time() - t_crawler_start:.2f}s")
                t_crawl_all_start = time.time()
                total_products_df = crawl_brand_all(brand_code, limit=3)
                print(f"[TIMER] crawl_brand_all 완료: {time.time() - t_crawl_all_start:.2f}s")
                t_loop_start = time.time()
                for _, row in total_products_df.iterrows():
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
        session['total_products_df'] = total_products_df.to_dict() if total_products_df is not None else None
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

# 상품별 리뷰/키워드 라우트
@app.route("/product/<goodsNo>/reviews")
def product_reviews(goodsNo):
    from crawlers.olive_crawler import crawl_reviews_for_goods
    import pandas as pd
    # 상품 데이터프레임 불러오기
    total_products = session.get('total_products')
    total_products_df_dict = session.get('total_products_df')
    if not total_products or not total_products_df_dict:
        return redirect(url_for('index'))
    total_products_df = pd.DataFrame(total_products_df_dict)
    # 상품 정보 찾기
    product_row = None
    for p in total_products:
        if str(p['goodsNo']) == str(goodsNo):
            product_row = p
            break
    if not product_row:
        return redirect(url_for('index'))
    product_name = product_row['name']
    # product_link를 total_products_df에서 goodsNo로 찾아서 할당
    product_link = '#'
    if not total_products_df.empty:
        match = total_products_df[total_products_df['goodsNo'].astype(str) == str(goodsNo)]
        if not match.empty and 'link' in match.columns:
            product_link = match.iloc[0]['link']
    # 리뷰 데이터 가져오기
    reviews_df = crawl_reviews_for_goods([goodsNo], limit=5)
    reviews = []
    keywords_all = []
    if not reviews_df.empty:
        for _, row in reviews_df.iterrows():
            review = row.to_dict()
            # 키워드 추출
            try:
                from review_utils import extract_keywords_with_openai
                keywords = extract_keywords_with_openai(review['content'])
                keywords_all.extend(keywords)
            except Exception:
                keywords = []
            review['keywords'] = keywords
            reviews.append(review)
    # 워드클라우드 생성
    wordcloud_url = None
    if keywords_all:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')  # 서버 환경에서 안전하게 워드클라우드 생성
        import matplotlib.pyplot as plt
        import io, base64
        # 한글 폰트 경로를 프로젝트 내 fonts/NanumGothic.ttf로 지정
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
        wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color="white").generate(' '.join(keywords_all))
        img_io = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_io.seek(0)
        wordcloud_url = 'data:image/png;base64,' + base64.b64encode(img_io.getvalue()).decode('utf8')
    # 각 리뷰에 대해 keywords_str 추가 (쉼표로 연결)
    for review in reviews:
        review['keywords_str'] = ', '.join(review.get('keywords', [])) if review.get('keywords') else ''

    return render_template(
        "product_reviews.html",
        product_name=product_name,
        product_link=product_link,
        reviews=reviews,
        wordcloud_url=wordcloud_url
    )


import tempfile
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

@app.route("/download_report", methods=["GET"])
def download_report():
    report_data = session.get('report_data')
    if not report_data:
        return redirect(url_for('index'))
    recs = report_data['recs']
    metrics = report_data['metrics']
    insights = report_data['insights']
    wordcloud_url = report_data['wordcloud_url']
    # 1. 추천조치 테이블 DataFrame 생성
    df = pd.DataFrame(recs)
    # 2. 엑셀 생성
    wb = Workbook()
    ws = wb.active
    ws.title = "추천조치"
    # 헤더
    ws.append(list(df.columns))
    # 데이터
    for row in df.itertuples(index=False):
        ws.append(list(row))
    # 3. 워드클라우드 이미지 삽입 (있을 때만)
    if wordcloud_url and wordcloud_url.startswith('data:image'):
        img_data = base64.b64decode(wordcloud_url.split(',')[1])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
            tmp_img.write(img_data)
            tmp_img.flush()
            img = XLImage(tmp_img.name)
            img.width = 400
            img.height = 180
            ws.add_image(img, 'H2')
    # 4. metrics, insights 시트 추가
    if metrics:
        ws2 = wb.create_sheet("효과요약")
        for k, v in metrics.items():
            ws2.append([k, v])
    if insights:
        ws3 = wb.create_sheet("인사이트")
        for k, v in insights.items():
            ws3.append([k, v])
    # 5. 파일로 저장 후 다운로드
    with BytesIO() as output:
        wb.save(output)
        output.seek(0)
        return send_file(output, as_attachment=True, download_name="review_report.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

import smtplib
from email.message import EmailMessage

@app.route("/send_report_email", methods=["POST"])
def send_report_email():
    report_data = session.get('report_data')
    if not report_data:
        return redirect(url_for('index'))
    recs = report_data['recs']
    metrics = report_data['metrics']
    insights = report_data['insights']
    wordcloud_url = report_data['wordcloud_url']
    # 1. 엑셀 리포트 생성 (메모리)
    df = pd.DataFrame(recs)
    wb = Workbook()
    ws = wb.active
    ws.title = "추천조치"
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))
    if wordcloud_url and wordcloud_url.startswith('data:image'):
        img_data = base64.b64decode(wordcloud_url.split(',')[1])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
            tmp_img.write(img_data)
            tmp_img.flush()
            img = XLImage(tmp_img.name)
            img.width = 400
            img.height = 180
            ws.add_image(img, 'H2')
    if metrics:
        ws2 = wb.create_sheet("효과요약")
        for k, v in metrics.items():
            ws2.append([k, v])
    if insights:
        ws3 = wb.create_sheet("인사이트")
        for k, v in insights.items():
            ws3.append([k, v])
    with BytesIO() as output:
        wb.save(output)
        output.seek(0)
        xlsx_bytes = output.read()
    # 2. 이메일 발송
    email_to = request.form.get('email')
    email_user = os.getenv('EMAIL_USER')
    email_password = os.getenv('EMAIL_PASSWORD')
    if not (email_to and email_user and email_password):
        return "이메일 발송 정보가 부족합니다. 관리자에게 문의하세요.", 400
    msg = EmailMessage()
    msg['Subject'] = "리뷰 대응 리포트"
    msg['From'] = email_user
    msg['To'] = email_to
    msg.set_content("첨부파일로 리뷰 대응 리포트를 보내드립니다.")
    msg.add_attachment(xlsx_bytes, maintype='application', subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename='review_report.xlsx')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_user, email_password)
            smtp.send_message(msg)
        return "이메일이 성공적으로 발송되었습니다!", 200
    except Exception as e:
        return f"이메일 발송 실패: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

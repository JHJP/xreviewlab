from flask import Blueprint, request, jsonify
from review_utils import get_keyword_meaning_rag, update_keyword_info
import logging

product_bp = Blueprint('product', __name__)

@product_bp.route("/product/<goodsNo>/reviews")
def product_reviews(goodsNo):
    import os
    import pandas as pd
    from collections import Counter
    import io, base64
    from flask import render_template

    # 항상 CSV에서 데이터 읽기
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'total_brand_reviews_df.csv')
    if not os.path.exists(csv_path):
        return "리뷰 데이터 파일이 없습니다.", 404
    df = pd.read_csv(csv_path)
    # goodsNo별로 필터링
    reviews_df = df[df['goodsNo'].astype(str) == str(goodsNo)].copy()
    if reviews_df.empty:
        return "해당 상품의 리뷰가 없습니다.", 404

    # 테이블에 필요한 정보 추출
    reviews = []
    for _, row in reviews_df.iterrows():
        reviews.append({
            'nickname': row['nickname'],
            'date': row['date'],
            'location': row.get('location', ''),
            'damage': row['damage'],
            'score': row['score'],
            'help_cnt': row['help_cnt'],
            'content': row['content'],
            'keywords': eval(row['keywords']) if isinstance(row['keywords'], str) else [],
        })

    # 워드클라우드/히스토그램용 키워드: real_keywords_all 사용
    # real_keywords_all은 모든 row에 동일하게 할당되어 있으므로 첫 행에서 추출
    real_keywords_all = []
    if 'real_keywords_all' in reviews_df.columns and not reviews_df.empty:
        first_real_keywords = reviews_df.iloc[0]['real_keywords_all']
        # 문자열로 저장되어 있으면 파싱
        if isinstance(first_real_keywords, str):
            import ast
            real_keywords_all = ast.literal_eval(first_real_keywords)
        elif isinstance(first_real_keywords, list):
            real_keywords_all = first_real_keywords
        else:
            real_keywords_all = []
    
    wordcloud_url = None
    histogram_url = None
    if real_keywords_all:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        font_candidates = [f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')]
        font_path = os.path.join(fonts_dir, font_candidates[0]) if font_candidates else None
        wc = WordCloud(font_path=font_path, width=800, height=260, background_color='white').generate(' '.join(real_keywords_all))
        img_io = io.BytesIO()
        plt.figure(figsize=(8, 2.6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_io.seek(0)
        wordcloud_url = 'data:image/png;base64,' + base64.b64encode(img_io.getvalue()).decode('utf8')
        # --- 히스토그램 생성 ---
        keyword_counts = Counter(real_keywords_all)
        # top5_keywords: 상위 5개 키워드와 빈도수 튜플 리스트
        keywords, counts = zip(*keyword_counts.most_common(20)) if keyword_counts else ([],[])
        img_io_hist = io.BytesIO()
        import matplotlib.font_manager as fm
        font_path_hist = os.path.join(fonts_dir, 'GamjaFlower-Regular.ttf') if os.path.exists(os.path.join(fonts_dir, 'GamjaFlower-Regular.ttf')) else font_path
        font_prop = fm.FontProperties(fname=font_path_hist) if font_path_hist else None
        plt.figure(figsize=(8, 2.6))
        bars = plt.bar(keywords, counts, color='#5C3BFF', edgecolor='#4328d9')
        plt.xticks(rotation=30, ha='right', fontsize=13, fontproperties=font_prop)
        plt.yticks(fontsize=12)
        plt.xlabel('키워드', fontsize=15, labelpad=8, fontproperties=font_prop)
        plt.ylabel('빈도수', fontsize=15, labelpad=8, fontproperties=font_prop)
        plt.title('키워드 히스토그램', fontsize=18, pad=10, color='#5C3BFF', fontproperties=font_prop)
        plt.tight_layout()
        import numpy as np
        if counts:
            plt.yticks(np.arange(0, max(counts)+1, 1))
        plt.savefig(img_io_hist, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_io_hist.seek(0)
        histogram_url = 'data:image/png;base64,' + base64.b64encode(img_io_hist.getvalue()).decode('utf8')
    # 각 리뷰에 대해 keywords_str 추가 (쉼표로 연결)
    for review in reviews:
        review['keywords_str'] = ', '.join(review.get('keywords', [])) if review.get('keywords') else ''
    # 상품 정보(테이블에서 첫 행 사용)
    product_name = reviews_df.iloc[0]['prd_name']
    product_link = reviews_df.iloc[0]['prd_link']

    return render_template(
        "product_reviews.html",
        product_name=product_name,
        product_link=product_link,
        reviews=reviews,
        wordcloud_url=wordcloud_url,
        histogram_url=histogram_url,
    )

# ────────────────────────────────────────────────────────────────
# API: 키워드 의미 조회(GET), 키워드/비용/시간 수정(POST)
import os
import pandas as pd
import ast

@product_bp.route('/product/<goodsNo>/keyword/<keyword>', methods=['GET'])
def get_keyword_info(goodsNo, keyword):
    # 1. cost, processing_time 조회 (신규: keyword_plan_info)
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'total_brand_reviews_df.csv')
    df = pd.read_csv(csv_path)
    idxs = df[df['goodsNo'].astype(str) == str(goodsNo)].index
    cost = ''
    processing_time = ''
    for idx in idxs:
        # real_keywords_all: 문자열 -> 리스트
        kw_list = ast.literal_eval(df.at[idx, 'real_keywords_all']) if pd.notnull(df.at[idx, 'real_keywords_all']) else []
        if keyword in kw_list:
            # keyword_plan_info 우선
            plan_info = None
            if 'keyword_plan_info' in df.columns and pd.notnull(df.at[idx, 'keyword_plan_info']):
                try:
                    import json
                    plan_info = json.loads(df.at[idx, 'keyword_plan_info'])
                except Exception:
                    plan_info = None
            if plan_info and keyword in plan_info:
                cost_val = plan_info[keyword].get('cost', '')
                time_val = plan_info[keyword].get('time', '')
                # Only set '' if missing/null, else keep 0 or value
                cost = '' if cost_val is None or (isinstance(cost_val, float) and pd.isna(cost_val)) else cost_val
                processing_time = '' if time_val is None or (isinstance(time_val, float) and pd.isna(time_val)) else time_val
            else:
                cost_val = df.at[idx, 'cost'] if 'cost' in df.columns else ''
                time_val = df.at[idx, 'processing_time'] if 'processing_time' in df.columns else ''
                cost = '' if cost_val is None or (isinstance(cost_val, float) and pd.isna(cost_val)) else cost_val
                processing_time = '' if time_val is None or (isinstance(time_val, float) and pd.isna(time_val)) else time_val
            break
    return jsonify({
        'cost': cost,
        'processing_time': processing_time
    })

# 의미만 반환하는 별도 엔드포인트
@product_bp.route('/product/<goodsNo>/keyword/<keyword>/meaning', methods=['GET'])
def get_keyword_meaning(goodsNo, keyword):
    from review_utils import get_keyword_meaning_rag
    meaning = get_keyword_meaning_rag(goodsNo, keyword)
    return jsonify({'meaning': meaning})

# 대응플랜 생성 탭 (GET/POST)
from flask import render_template
@product_bp.route('/plan', methods=['GET', 'POST'])
def plan():
    import io, base64
    import traceback
    from optimizer import run_optimizer
    result = None
    plot_url = None
    picked_html = None
    if request.method == 'POST':
        budget = request.form.get('budget', type=float)
        hours = request.form.get('hours', type=float)
        try:
            picked, fig = run_optimizer(budget, hours)
            # DataFrame to HTML table
            picked_html = picked.to_html(index=False, classes='table table-striped', border=0, justify='center')
            # Plot to base64
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', bbox_inches='tight', dpi=200)
            img_io.seek(0)
            plot_url = 'data:image/png;base64,' + base64.b64encode(img_io.getvalue()).decode('utf8')
            result = True
            print("DEBUG: run_optimizer OK", flush=True)
        except Exception as e:
            tb = traceback.format_exc()
            print("ERROR:", e, flush=True)
            print(traceback.format_exc(), flush=True)
            result = f'<span style="color:red;">오류: {e}<br><pre>{tb}</pre></span>'
    return render_template('plan.html', result=result, picked_html=picked_html, plot_url=plot_url)


@product_bp.route('/product/<goodsNo>/keyword/<keyword>', methods=['POST'])
def update_keyword(goodsNo, keyword):
    data = request.get_json()
    new_keyword = data.get('new_keyword', keyword)
    cost = data.get('cost', '')
    # Accept both 'time' (from frontend) and 'processing_time' (for backward compatibility)
    time = data.get('time', data.get('processing_time', ''))
    success = update_keyword_info(goodsNo, keyword, new_keyword, cost, time)
    return jsonify({'success': bool(success)})

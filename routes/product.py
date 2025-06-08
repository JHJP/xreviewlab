from flask import Blueprint, render_template, session, redirect, url_for
from crawlers.olive_crawler import crawl_reviews_for_goods
import pandas as pd

product_bp = Blueprint('product', __name__)

@product_bp.route("/product/<goodsNo>/reviews")
def product_reviews(goodsNo):
    # 상품 데이터프레임 불러오기
    total_products = session.get('total_products')
    total_products_df_dict = session.get('total_products_df')
    if not total_products or not total_products_df_dict:
        return redirect(url_for('index'))
    # 상품 정보 찾기
    product_row = None
    for p in total_products:
        if str(p['goodsNo']) == str(goodsNo):
            product_row = p
            break
    if not product_row:
        return redirect(url_for('index'))
    product_name = product_row['name']
    product_link = product_row['link']
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
    # 워드클라우드 및 히스토그램 생성
    wordcloud_url = None
    histogram_url = None
    if keywords_all:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64
        import os
        from collections import Counter
        # --- 워드클라우드 생성 ---
        img_io = io.BytesIO()
        fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        font_candidates = [f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')]
        font_path = os.path.join(fonts_dir, font_candidates[0]) if font_candidates else None
        wc = WordCloud(font_path=font_path, width=800, height=260, background_color='white').generate(' '.join(keywords_all))
        plt.figure(figsize=(8, 2.6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_io.seek(0)
        wordcloud_url = 'data:image/png;base64,' + base64.b64encode(img_io.getvalue()).decode('utf8')
        print('wordcloud_url:', wordcloud_url[:100])  # 앞 100글자만 출력
        # --- 히스토그램(bar chart) 생성 ---
        # 상위 20개 키워드만 표시
        keyword_counts = Counter(keywords_all)
        keywords, counts = zip(*keyword_counts.most_common(20)) if keyword_counts else ([],[])
        img_io_hist = io.BytesIO()
        import matplotlib.font_manager as fm
        font_path_hist = os.path.join(fonts_dir, 'GamjaFlower-Regular.ttf')
        font_prop = fm.FontProperties(fname=font_path_hist)
        plt.figure(figsize=(8, 2.6))
        bars = plt.bar(keywords, counts, color='#5C3BFF', edgecolor='#4328d9')
        plt.xticks(rotation=30, ha='right', fontsize=13, fontproperties=font_prop)
        plt.yticks(fontsize=12)
        plt.xlabel('키워드', fontsize=15, labelpad=8, fontproperties=font_prop)
        plt.ylabel('빈도수', fontsize=15, labelpad=8, fontproperties=font_prop)
        plt.title('키워드 히스토그램', fontsize=18, pad=10, color='#5C3BFF', fontproperties=font_prop)
        plt.tight_layout()
        # y축을 정수로만 표시
        import numpy as np
        if counts:
            plt.yticks(np.arange(0, max(counts)+1, 1))
        plt.tight_layout()
        plt.savefig(img_io_hist, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_io_hist.seek(0)
        histogram_url = 'data:image/png;base64,' + base64.b64encode(img_io_hist.getvalue()).decode('utf8')
        print('histogram_url:', histogram_url[:100])  # 앞 100글자만 출력
        img_io_hist.seek(0)
        histogram_url = 'data:image/png;base64,' + base64.b64encode(img_io_hist.getvalue()).decode('utf8')
    # 각 리뷰에 대해 keywords_str 추가 (쉼표로 연결)
    for review in reviews:
        review['keywords_str'] = ', '.join(review.get('keywords', [])) if review.get('keywords') else ''
    return render_template(
        "product_reviews.html",
        product_name=product_name,
        product_link=product_link,
        reviews=reviews,
        wordcloud_url=wordcloud_url,
        histogram_url=histogram_url
    )

from flask import Blueprint, render_template, session, redirect, url_for
from crawlers.olive_crawler import olive_reviews_crawl
import pandas as pd

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

    # 워드클라우드/히스토그램용 키워드 모으기
    keywords_all = [kw for review in reviews for kw in review['keywords']]

    # 워드클라우드 생성
    wordcloud_url = None
    histogram_url = None
    if keywords_all:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        font_candidates = [f for f in os.listdir(fonts_dir) if f.lower().endswith('.ttf')]
        font_path = os.path.join(fonts_dir, font_candidates[0]) if font_candidates else None
        wc = WordCloud(font_path=font_path, width=800, height=260, background_color='white').generate(' '.join(keywords_all))
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
        keyword_counts = Counter(keywords_all)
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
        histogram_url=histogram_url
    )

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
                from review_utils import extract_keywords_with_openai, generate_response_with_openai
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
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64
        img_io = io.BytesIO()
        import os
        # fonts 폴더 내 ttf 파일 지정 (예: NanumGothic.ttf)
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

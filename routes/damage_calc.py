import numpy as np

def calculate_damage(reviews_df):
    """
    리뷰 데이터프레임에 damage score를 계산하여 소수점 2자리로 저장합니다.
    Args:
        reviews_df (pd.DataFrame): 리뷰 데이터프레임
    Returns:
        pd.DataFrame: damage 컬럼이 추가된 데이터프레임
    """
    linpred = (
        0.002 * np.log(reviews_df['word_count'])
        # + 0.056 * reviews_df['hygiene_issue']          # 데이터 있으면 주석 해제
        # + 0.093 * np.log(reviews_df['num_reviews'])
        # + 0.055 * np.log(reviews_df['num_friends'])
        # + 0.023 * np.log(reviews_df['profile_comps'])
    )
    linpred += 0.279 * (reviews_df['top_rank_present'] + reviews_df['badges_present'])
    linpred += 0.219 * reviews_df['photo_present']
    mu_hat = np.exp(linpred)
    reviews_df["damage"] = mu_hat * (1 + np.log1p(reviews_df['help_cnt']))
    reviews_df["damage"] = reviews_df["damage"].round(2)
    return reviews_df

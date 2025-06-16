import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min

def run_keyword_clustering_and_damage(csv_path='total_brand_reviews_df.csv'):
    """
    키워드 클러스터링 및 damage 집계. goodsNo별로 처리하여 결과를 같은 csv_path로 덮어씀.
    """
    df = pd.read_csv(csv_path)
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # 결과를 저장할 컬럼 초기화
    df['real_keywords_all'] = None
    df['real_keywords_dmg_dict'] = None

    for goods_no, group in df.groupby('goodsNo'):
        idxs = group.index
        review_keywords = []
        all_keywords = []
        for kw_str in group['keywords']:
            kws = ast.literal_eval(kw_str) if isinstance(kw_str, str) else []
            review_keywords.append(kws)
            all_keywords.extend(kws)
        unique_keywords = list(set(all_keywords))

        if not unique_keywords:
            # 키워드가 없는 경우 빈 값 처리
            real_keywords_all = []
            real_keywords_dmg_dict = {}
            df.loc[idxs, 'real_keywords_all'] = [real_keywords_all] * len(idxs)
            df.loc[idxs, 'real_keywords_dmg_dict'] = [real_keywords_dmg_dict] * len(idxs)
            continue

        # 벡터화
        keyword_vecs = model.encode(unique_keywords, show_progress_bar=False)

        # 클러스터링
        if len(unique_keywords) == 1:
            labels = [0]
        else:
            clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.35)
            labels = clustering.fit_predict(keyword_vecs)

        # 클러스터별 대표 키워드(최빈값)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        rep_keywords = {}
        for label, idxs_in_cluster in clusters.items():
            # 대표 키워드: 클러스터 내 가장 많이 등장한 키워드(최빈값)
            cluster_kws = [unique_keywords[i] for i in idxs_in_cluster]
            # 최빈값: 여러 개면 첫 번째
            rep_kw = max(set(cluster_kws), key=cluster_kws.count)
            rep_keywords[label] = rep_kw

        # 각 키워드를 대표 키워드로 매핑
        kw_to_rep = {unique_keywords[idx]: rep_keywords[label] for idx, label in enumerate(labels)}

        # real_keywords_all, cluster damage 계산
        real_keywords_all = []
        cluster_damages = {rep: [] for rep in rep_keywords.values()}
        for kws, dmg in zip(review_keywords, group['damage']):
            for kw in kws:
                rep = kw_to_rep[kw]
                real_keywords_all.append(rep)
                cluster_damages[rep].append(dmg)

        real_keywords_dmg_dict = {rep: float(np.mean(damages)) if damages else 0.0 for rep, damages in cluster_damages.items()}

        # 각 row에 저장 (행마다 동일한 값을 할당)
        for i in idxs:
            df.at[i, 'real_keywords_all'] = real_keywords_all
            df.at[i, 'real_keywords_dmg_dict'] = real_keywords_dmg_dict

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Done! Saved as {csv_path} (encoding=utf-8-sig)')


if __name__ == "__main__":
    run_keyword_clustering_and_damage('total_brand_reviews_df.csv')

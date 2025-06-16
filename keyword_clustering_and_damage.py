import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min

def run_keyword_clustering_and_damage(csv_path='total_brand_reviews_df.csv'):
    """
    키워드 클러스터링 및 damage 집계. 결과는 같은 csv_path로 덮어씀.
    """
    df = pd.read_csv(csv_path)

    # 2. Parse keywords and collect all unique keywords
    all_keywords = []
    review_keywords = []
    for kw_str in df['keywords']:
        kws = ast.literal_eval(kw_str) if isinstance(kw_str, str) else []
        review_keywords.append(kws)
        all_keywords.extend(kws)
    unique_keywords = list(set(all_keywords))

    # 3. Vectorize keywords
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    keyword_vecs = model.encode(unique_keywords, show_progress_bar=True)

    # 4. Cluster vectors (tune distance threshold as needed)
    clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.35)
    labels = clustering.fit_predict(keyword_vecs)

    # 5. Find cluster representatives (medoids)
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    rep_keywords = {}
    for label, idxs in clusters.items():
        cluster_vecs = keyword_vecs[idxs]
        center = cluster_vecs.mean(axis=0, keepdims=True)
        medoid_idx, _ = pairwise_distances_argmin_min(center, cluster_vecs)
        rep_keywords[label] = unique_keywords[idxs[medoid_idx[0]]]

    # 6. Map each keyword to its representative
    kw_to_rep = {unique_keywords[idx]: rep_keywords[label] for idx, label in enumerate(labels)}

    # 7. Build real_keywords_all and cluster damage
    real_keywords_all = []
    cluster_damages = {rep: [] for rep in rep_keywords.values()}
    for kws, dmg in zip(review_keywords, df['damage']):
        for kw in kws:
            rep = kw_to_rep[kw]
            real_keywords_all.append(rep)
            cluster_damages[rep].append(dmg)

    # 8. Calculate average damage per cluster
    real_keywords_dmg_dict = {rep: float(np.mean(damages)) if damages else 0.0 for rep, damages in cluster_damages.items()}

    # 9. Add columns and save
    # Each review gets the full real_keywords_all and real_keywords_dmg_dict (as per instruction)
    df['real_keywords_all'] = [real_keywords_all] * len(df)
    df['real_keywords_dmg_dict'] = [real_keywords_dmg_dict] * len(df)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Done! Saved as {csv_path} (encoding=utf-8-sig)')

if __name__ == "__main__":
    run_keyword_clustering_and_damage('total_brand_reviews_df.csv')

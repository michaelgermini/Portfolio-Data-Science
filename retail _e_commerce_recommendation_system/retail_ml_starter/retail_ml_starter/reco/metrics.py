import numpy as np

def precision_at_k(recommended, relevant_set, k=10):
    """
    recommended: array d'item_ids ordonnés (longueur >= k)
    relevant_set: set des items pertinents (gt)
    """
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant_set)
    return hits / float(k)

def dcg_at_k(recommended, relevant_set, k=10):
    rec_k = recommended[:k]
    gains = [1.0 if i in relevant_set else 0.0 for i in rec_k]
    return sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))

def idcg_at_k(relevant_count, k=10):
    gains = [1.0] * min(relevant_count, k) + [0.0] * max(0, k - relevant_count)
    return sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))

def ndcg_at_k(recommended, relevant_set, k=10):
    dcg = dcg_at_k(recommended, relevant_set, k)
    idcg = idcg_at_k(len(relevant_set), k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_topk(model, R_train, test_items, k=10):
    """
    model: doit implémenter predict_scores(user_ids) -> ndarray [n_users, n_items]
    test_items: dict user_id -> single item id (leave-one-out)
    Retourne: mean Precision@k, mean NDCG@k
    """
    n_users, n_items = R_train.shape
    user_ids = np.arange(n_users)
    scores = model.predict_scores(user_ids)
    # mask items vus
    scores = np.where(R_train > 0, -np.inf, scores)
    precs, ndcgs = [], []
    for u in user_ids:
        rec = np.argsort(-scores[u])  # tri desc
        relevant = {test_items[u]} if u in test_items else set()
        precs.append(precision_at_k(rec, relevant, k))
        ndcgs.append(ndcg_at_k(rec, relevant, k))
    return float(np.mean(precs)), float(np.mean(ndcgs))

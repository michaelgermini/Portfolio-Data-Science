import numpy as np
from reco.data_utils import generate_implicit_data, train_test_split_leave_one_out
from reco.recommender import MFRecommender
from reco.metrics import evaluate_topk

def main():
    # 1) Données synthétiques
    R = generate_implicit_data(n_users=400, n_items=600, density=0.015, seed=7)
    R_train, test_items = train_test_split_leave_one_out(R, seed=7)

    # 2) Entraînement
    model = MFRecommender(n_factors=48, lr=0.05, reg=0.01, n_iters=8, neg_ratio=4, seed=7).fit(R_train)

    # 3) Évaluation Top-K
    prec, ndcg = evaluate_topk(model, R_train, test_items, k=10)
    print(f"Precision@10: {prec:.4f}")
    print(f"NDCG@10     : {ndcg:.4f}")

    # 4) Exemple de recommandation pour l'utilisateur 0
    rec = model.recommend(R_train, user_id=0, k=10)
    print("Top-10 pour user_id=0:", rec.tolist())

if __name__ == "__main__":
    main()

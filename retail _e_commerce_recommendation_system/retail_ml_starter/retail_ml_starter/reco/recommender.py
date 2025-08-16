import numpy as np

class MFRecommender:
    """
    Factorisation de matrices (SGD) pour feedback implicite binaire.
    Perte: logistique sur interactions positives vs échantillons négatifs.
    Simple et sans dépendances externes.
    """
    def __init__(self, n_factors=32, lr=0.05, reg=0.01, n_iters=10, neg_ratio=3, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_iters = n_iters
        self.neg_ratio = neg_ratio
        self.rng = np.random.default_rng(seed)
        self.U = None
        self.V = None

    def fit(self, R):
        n_users, n_items = R.shape
        self.U = 0.01 * self.rng.standard_normal((n_users, self.n_factors))
        self.V = 0.01 * self.rng.standard_normal((n_items, self.n_factors))

        pos_indices = np.argwhere(R > 0)
        for it in range(self.n_iters):
            self.rng.shuffle(pos_indices)
            for u, i in pos_indices:
                # positive pair (u,i) with y=1
                self._sgd_step(int(u), int(i), 1.0)
                # sample negatives
                for _ in range(self.neg_ratio):
                    j = int(self.rng.integers(0, R.shape[1]))
                    while R[u, j] > 0:
                        j = int(self.rng.integers(0, R.shape[1]))
                    self._sgd_step(int(u), j, 0.0)
        return self

    def _sgd_step(self, u, i, y):
        # logistic loss: y*log(sigmoid(s)) + (1-y)*log(1-sigmoid(s))
        s = self.U[u] @ self.V[i]
        p = 1.0 / (1.0 + np.exp(-s))
        grad = (p - y)  # derivative wrt s
        # regularized updates
        self.U[u] -= self.lr * (grad * self.V[i] + self.reg * self.U[u])
        self.V[i] -= self.lr * (grad * self.U[u] + self.reg * self.V[i])

    def predict_scores(self, user_ids):
        return self.U[user_ids] @ self.V.T

    def recommend(self, R_train, user_id, k=10):
        scores = self.predict_scores([user_id])[0]
        scores = np.where(R_train[user_id] > 0, -np.inf, scores)
        return np.argsort(-scores)[:k]

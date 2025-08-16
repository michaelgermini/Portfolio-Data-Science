import numpy as np

def generate_implicit_data(n_users=500, n_items=800, density=0.01, seed=42):
    """
    Génère une matrice implicite binaire (0/1) user-item.
    `density` ~ proportion d'interactions positives.
    """
    rng = np.random.default_rng(seed)
    mat = (rng.random((n_users, n_items)) < density).astype(np.float32)
    return mat

def train_test_split_leave_one_out(R, seed=42):
    """
    Split leave-one-out: pour chaque user, on met de côté 1 item positif pour test.
    Retourne R_train (copie) et dict test_items[user] = item_id
    """
    rng = np.random.default_rng(seed)
    R = R.copy()
    n_users, n_items = R.shape
    test_items = {}
    for u in range(n_users):
        pos = np.where(R[u] > 0)[0]
        if pos.size > 0:
            i = rng.choice(pos)
            test_items[u] = int(i)
            R[u, i] = 0.0  # retirer du train
    return R, test_items

def mask_seen(scores, R_train):
    """
    Met à -inf les items déjà vus dans le train pour ne pas les recommander.
    """
    masked = scores.copy()
    masked[R_train > 0] = -np.inf
    return masked

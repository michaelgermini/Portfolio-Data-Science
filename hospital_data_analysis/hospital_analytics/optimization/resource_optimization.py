from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def allocate_beds(
	departments: list[str],
	demand: list[int],
	capacity: list[int],
	penalty_unserved: float = 1.0,
) -> pd.DataFrame:
	# Decision vars: served[i] for each department i
	# Objective: minimize unserved = sum(max(0, demand[i] - served[i])) approximated by penalty on slack using bounds
	# Here, we simply maximize served with negative coefficients (convert to minimization) subject to capacity caps.
	n = len(departments)
	c = np.array([-1.0] * n)  # maximize served -> minimize negative served

	A_ub = []
	b_ub = []
	# served[i] <= capacity[i]
	for i in range(n):
		row = [0.0] * n
		row[i] = 1.0
		A_ub.append(row)
		b_ub.append(float(capacity[i]))
	# served[i] <= demand[i]
	for i in range(n):
		row = [0.0] * n
		row[i] = 1.0
		A_ub.append(row)
		b_ub.append(float(demand[i]))

	bounds = [(0, None) for _ in range(n)]

	res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
	served = res.x if res.success else np.zeros(n)
	unserved = np.maximum(0, np.array(demand) - served)

	return pd.DataFrame(
		{
			"department": departments,
			"demand": demand,
			"capacity": capacity,
			"served": np.round(served).astype(int),
			"unserved": np.round(unserved).astype(int),
		}
	)

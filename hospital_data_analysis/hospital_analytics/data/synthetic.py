import numpy as np
import pandas as pd


def generate_synthetic_hospital_data(
	num_patients: int = 5000,
	start_date: str = "2022-01-01",
	seed: int | None = 42,
) -> pd.DataFrame:
	if seed is not None:
		rng = np.random.default_rng(seed)
	else:
		rng = np.random.default_rng()

	patient_id = np.arange(1, num_patients + 1)

	age = rng.integers(18, 95, size=num_patients)
	sex = rng.choice(["F", "M"], size=num_patients, p=[0.52, 0.48])
	diagnoses = [
		"Diabetes",
		"Heart Failure",
		"Pneumonia",
		"COPD",
		"CKD",
		"Hypertension",
	]
	diagnosis = rng.choice(diagnoses, size=num_patients)

	departments = ["Emergency", "Cardiology", "Internal Med", "Surgery", "ICU"]
	department = rng.choice(departments, size=num_patients, p=[0.35, 0.15, 0.3, 0.15, 0.05])

	comorbidity_index = np.clip(
		np.round(rng.normal(loc=2.0, scale=1.2, size=num_patients)), 0, None
	).astype(int)
	prior_admissions = np.clip(
		rng.poisson(lam=0.6, size=num_patients), 0, None
	).astype(int)
	length_of_stay = np.clip(
		np.round(rng.gamma(shape=2.0, scale=2.0, size=num_patients)), 1, None
	).astype(int)

	start = pd.to_datetime(start_date)
	admit_date = start + pd.to_timedelta(rng.integers(0, 365, size=num_patients), unit="D")
	discharge_date = admit_date + pd.to_timedelta(length_of_stay, unit="D")

	base_log_odds = -2.0
	logit = (
		base_log_odds
		+ 0.02 * (age - 60)
		+ 0.35 * (comorbidity_index)
		+ 0.25 * (prior_admissions)
		+ 0.03 * (length_of_stay - 3)
		+ 0.3 * (diagnosis == "Heart Failure").astype(float)
		+ 0.25 * (diagnosis == "Diabetes").astype(float)
	)
	prob_readmit = 1.0 / (1.0 + np.exp(-logit))
	readmitted_within_30d = (rng.random(num_patients) < prob_readmit).astype(int)

	baseline_hazard = 1.0 / 120.0
	hazard_multiplier = np.exp(
		0.015 * (age - 60)
		+ 0.25 * comorbidity_index
		+ 0.2 * prior_admissions
		+ 0.25 * (diagnosis == "Heart Failure").astype(float)
	)
	rate = baseline_hazard * hazard_multiplier
	time_to_event = rng.exponential(1.0 / np.clip(rate, 1e-6, None))

	censoring_limit = 180.0
	event_time_days = np.minimum(time_to_event, censoring_limit)
	event_observed = (time_to_event <= censoring_limit).astype(int)

	df = pd.DataFrame(
		{
			"patient_id": patient_id,
			"age": age,
			"sex": sex,
			"diagnosis": diagnosis,
			"department": department,
			"comorbidity_index": comorbidity_index,
			"prior_admissions": prior_admissions,
			"length_of_stay": length_of_stay,
			"admit_date": admit_date,
			"discharge_date": discharge_date,
			"readmitted_within_30d": readmitted_within_30d,
			"time_to_event": np.round(event_time_days, 1),
			"event_observed": event_observed,
		}
	)

	return df

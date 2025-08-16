from pathlib import Path
import random
import math
import csv


COUNTRIES = [
    "France", "Germany", "United Kingdom", "Italy", "Spain",
    "United States", "Canada", "Brazil", "Mexico", "Argentina",
    "China", "India", "Japan", "South Korea", "South Africa",
]

INDICATORS = {
    "gdp_per_capita_usd": {"unit": "USD", "base": 8000, "growth": 0.03},
    "life_expectancy_years": {"unit": "years", "base": 60, "growth": 0.10},
    "school_enrollment_primary_percent": {"unit": "%", "base": 70, "growth": 0.20},
    "water_access_percent": {"unit": "%", "base": 60, "growth": 0.30},
}


def generate_value(ind_key: str, year_idx: int, country_idx: int) -> float:
    meta = INDICATORS[ind_key]
    base = meta["base"] * (1.0 + 0.05 * (country_idx % 5))
    growth = meta["growth"]
    noise = random.uniform(-0.5, 0.5)
    val = base * (1.0 + growth) ** year_idx + noise * base * 0.02
    # clamp for percents
    if meta["unit"] == "%":
        return max(0.0, min(100.0, val))
    return max(0.0, val)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "exploration_inegalites" / "data"
    root.mkdir(parents=True, exist_ok=True)
    out_csv = root / "inegalites_sample.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["country", "year", "indicator", "value", "unit", "source"])
        years = list(range(2000, 2021))
        for c_idx, c in enumerate(COUNTRIES):
            for y_idx, y in enumerate(years):
                for ind in INDICATORS.keys():
                    v = generate_value(ind, y_idx, c_idx)
                    unit = INDICATORS[ind]["unit"]
                    w.writerow([c, y, ind, round(v, 2), unit, "synthetic".encode("utf-8").decode("utf-8")])
    print(f"Wrote {out_csv} with {len(COUNTRIES)*len(range(2000,2021))*len(INDICATORS)} rows")


if __name__ == "__main__":
    main()



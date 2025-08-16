from pathlib import Path
import pandas as pd
import numpy as np


def generate_local_temp(path: Path, years: int = 25) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Daily from Jan 1 of (today - years) to today
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    dates = pd.date_range(start=start, end=end, freq="D")

    # Seasonal cycle + slight warming trend
    day_of_year = dates.dayofyear.values
    seasonal = 10 + 8 * np.sin(2 * np.pi * (day_of_year / 365.25))
    trend = np.linspace(0, 0.5, len(dates))  # ~+0.5°C over the span
    noise = np.random.normal(0, 1.2, len(dates))
    tmean = seasonal + trend + noise

    df = pd.DataFrame({
        "date": dates,
        "tmean": tmean.round(2),
    })
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = Path("data_for_good/local_climate_projection/data/local_temp.csv")
    p = generate_local_temp(out, years=25)
    print(f"Wrote {p.resolve()} ({len(pd.read_csv(p))} rows)")



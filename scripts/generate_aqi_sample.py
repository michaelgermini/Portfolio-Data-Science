from pathlib import Path
import pandas as pd
import numpy as np


def generate_aqi(path: Path, days: int = 365) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    # Synthetic weather and AQI
    temp = 10 + 10 * np.sin(np.linspace(0, 4 * np.pi, len(rng))) + np.random.normal(0, 2, len(rng))
    wind = 2 + 1.0 * np.sin(np.linspace(0, 6 * np.pi, len(rng))) + np.random.normal(0, 0.5, len(rng))
    humidity = 60 + 15 * np.sin(np.linspace(0, 2 * np.pi, len(rng))) + np.random.normal(0, 5, len(rng))
    base_aqi = 65 + 10 * np.sin(np.linspace(0, 3 * np.pi, len(rng)))
    aqi = base_aqi + 0.6 * (humidity - 60) - 1.5 * (wind - 2) - 0.8 * (temp - 20) + np.random.normal(0, 5, len(rng))

    df = pd.DataFrame({
        "date": rng,
        "aqi": np.clip(aqi, 5, 250).round(0).astype(int),
        "temp": temp.round(1),
        "wind": wind.round(2),
        "humidity": np.clip(humidity, 10, 100).round(1),
    })
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = Path("data_for_good/air_quality_forecast/data/aqi_city.csv")
    p = generate_aqi(out, days=540)
    print(f"Wrote {p.resolve()} ({len(pd.read_csv(p))} rows)")



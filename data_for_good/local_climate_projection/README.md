# Data for Good – Local climate projection (2050)

Notebook: `notebooks/local_climate_projection.ipynb`

Sample data:
- `data/local_temp.csv` with columns: `date, tmean`
- Generate synthetic data:
```bash
python scripts/generate_local_temp_sample.py
```

Method (simplified):
- Build seasonal climatology from recent history
- Bias-correct projections so near-term aligns with observed climatology
- Apply scenario anomaly by 2050 (e.g., +1.5°C for SSP2-4.5, +3.0°C for SSP5-8.5) with a linear ramp

Outputs:
- Plot of historical vs projected mean temperature to 2050
- CSV export option for projected series


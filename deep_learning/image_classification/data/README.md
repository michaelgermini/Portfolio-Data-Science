# Image Classification – Dataset

Put images in subfolders per class, e.g.:

```
deep_learning/image_classification/data/
  cats/
    img_001.jpg
  dogs/
    img_020.jpg
```

No dataset? Generate a tiny synthetic one (red/green/blue) in the app sidebar, or run:

```bash
python deep_learning/image_classification/utils/generate_synthetic_dataset.py
```

The Streamlit page will split train/validation automatically and report:
- Accuracy (validation)
- Confusion matrix
- Per‑class Precision/Recall/F1


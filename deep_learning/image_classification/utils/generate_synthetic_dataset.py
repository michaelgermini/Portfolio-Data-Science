from pathlib import Path
import random
from PIL import Image, ImageDraw


def generate(root: Path, num_per_class: int = 40, size: int = 128) -> None:
    classes = {
        "red": (220, 40, 40),
        "green": (40, 180, 90),
        "blue": (70, 120, 230),
    }
    root.mkdir(parents=True, exist_ok=True)
    for cls, color in classes.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing minimal set? Keep existing files to avoid overrides.
        for i in range(num_per_class):
            img = Image.new("RGB", (size, size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            x0 = random.randint(0, size // 3)
            y0 = random.randint(0, size // 3)
            x1 = random.randint(size // 2, size)
            y1 = random.randint(size // 2, size)
            draw.rectangle([x0, y0, x1, y1], fill=color)
            img.save(cls_dir / f"img_{i:03d}.png")


if __name__ == "__main__":
    data_root = Path("deep_learning/image_classification/data")
    generate(data_root)
    print(f"Synthetic dataset generated at: {data_root.resolve()}")


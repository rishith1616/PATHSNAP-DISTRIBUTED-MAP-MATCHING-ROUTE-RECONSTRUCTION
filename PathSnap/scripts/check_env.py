import importlib
import platform
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "pyproj",
    "shapely",
    "networkx",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "optuna",
    "joblib",
    "fastapi",
    "uvicorn",
    "pydantic",
    "streamlit",
    "folium",
    "matplotlib",
    "seaborn",
]

EXPECTED_DIRS = [
    "data/raw/train",
    "data/raw/student_test",
    "data/raw/eval_test",
    "data/processed",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/optimization",
    "src/hmm",
    "api",
    "ui",
    "artifacts",
    "visualizations",
    "logs",
]


def main() -> None:
    print("=" * 60)
    print("PathSnap Stage 0 Environment & Structure Check (Windows 11)")
    print("=" * 60)
    print(f"OS / Platform : {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Executable    : {sys.executable}\n")

    if sys.version_info >= (3, 14):
        print("[FAIL] Python 3.14 detected. Recreate .venv with Python 3.12 or 3.11.")
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    missing_dirs = [d for d in EXPECTED_DIRS if not (root / d).is_dir()]
    if missing_dirs:
        print(f"[FAIL] Missing directories: {missing_dirs}")
        sys.exit(1)
    print(f"[OK] All {len(EXPECTED_DIRS)} required project directories exist.\n")

    failed_imports = []
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg:<15} -> {version}")
        except ImportError as exc:
            failed_imports.append((pkg, str(exc)))
            print(f"  [FAIL] {pkg:<15} -> {exc}")

    if failed_imports:
        print("\n[ERROR] Some packages failed to import. Fix before proceeding.")
        sys.exit(1)

    import torch

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None (CPU mode)"
    print(f"\n[INFO] PyTorch CUDA Available: {cuda_available} | Device: {gpu_name}")

    for split in ["train", "student_test", "eval_test"]:
        split_path = root / "data" / "raw" / split
        items = [p for p in split_path.iterdir() if not p.name.startswith(".")]
        print(f"[INFO] data/raw/{split:<13} -> {len(items)} items found")

    print("\n[SUCCESS] Stage 0 environment check passed.")


if __name__ == "__main__":
    main()
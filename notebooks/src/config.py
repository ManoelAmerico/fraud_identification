from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "data"

# path to project data files
ORIGINAL_DATA = DATA_FOLDER / "creditcard.zip"
PROCESSED_DATA = DATA_FOLDER / "creditcard.parquet"

# path to project model files
MODEL_FOLDER = PROJECT_FOLDER / "models"
FINAL_MODEL = MODEL_FOLDER / "xgb_classifier.joblib"

# other necessary paths
REPORTS_FOLDER = PROJECT_FOLDER / "reports"
IMAGE_FOLDER = REPORTS_FOLDER / "image"

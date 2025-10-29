from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
BASE_PATH = BASE_DIR / "model" / "data" / "data_split"

IMG_SIZE = 96
BATCH_SIZE = 32
NUM_WORKERS = 0

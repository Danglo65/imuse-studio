import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("IMUSE_DATA_DIR", BACKEND_DIR / "data")).resolve()

DATASETS_DIR = DATA_DIR / "datasets"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
GENERATED_DIR = DATA_DIR / "generated"
DB_PATH = DATA_DIR / "imuse.db"

for d in (DATASETS_DIR, JOBS_DIR, MODELS_DIR, GENERATED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# When set, training/generation use lightweight fake implementations instead of
# loading real diffusion models. Lets the full app be developed and tested on
# a machine with no GPU (e.g. this dev sandbox), while real deployments on a
# cloud GPU box unset this and get actual Stable Diffusion LoRA fine-tuning.
MOCK_ML = os.environ.get("IMUSE_MOCK_ML", "1") == "1"

DEFAULT_BASE_MODEL = os.environ.get("IMUSE_BASE_MODEL", "runwayml/stable-diffusion-v1-5")

PYTHON_EXECUTABLE = os.environ.get("IMUSE_PYTHON", os.sys.executable)

# imuse-studio

A trainable image generator: upload a handful of images of a concept
(a person, character, object, or style), fine-tune a LoRA adapter on top
of Stable Diffusion, and generate new images with it -- all from a web UI.

## How it works

- **Backend** (`backend/`): a FastAPI app that manages datasets, kicks off
  training jobs, and runs generation. Training and inference are separate,
  standalone scripts (`backend/training/train_lora.py`,
  `backend/training/inference.py`) that the backend shells out to -- they
  have no dependency on the web app, so they can be run directly on a GPU
  box (see `docs/TRAINING.md`).
- **Frontend** (`frontend/`): a React/TypeScript/Tailwind studio UI --
  Datasets, Train, Models, Generate.
- **Fine-tuning approach**: DreamBooth-style LoRA fine-tuning of a
  pretrained Stable Diffusion model via `diffusers` + `peft`. This teaches
  the base model a new concept from a small image set, rather than training
  a generative model from scratch -- far less data and compute needed.

## Quick start (mock mode, no GPU required)

By default the app runs in **mock mode** (`IMUSE_MOCK_ML=1`): training and
generation are simulated (fast fake progress, placeholder images) so you
can develop and exercise the whole product -- uploads, job tracking,
generation gallery -- without a GPU or any ML dependencies installed.

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend, in another terminal
cd frontend
npm install
npm run dev
```

Open the printed frontend URL (typically http://localhost:5173). Create a
dataset, upload a few images, start a training job, then generate.

Or via Docker: `docker compose up --build` brings up both services in mock
mode (backend on :8000, frontend on :4173).

## Real training on a GPU

Set `IMUSE_MOCK_ML=0` and deploy the backend to a machine with an NVIDIA
GPU (install `backend/requirements-ml.txt` in addition to
`requirements.txt`, or build `backend/Dockerfile.gpu`). See
[docs/TRAINING.md](docs/TRAINING.md) for cloud GPU setup (RunPod, AWS,
Colab) and fine-tuning hyperparameter guidance.

## Project layout

```
backend/
  app/            FastAPI app: routers, DB models, job orchestration
  training/       Standalone train_lora.py / inference.py scripts
  requirements.txt      Web app deps (always needed)
  requirements-ml.txt   Heavy ML deps (only for real, non-mock mode)
frontend/
  src/pages/      Datasets, Train, Models, Generate
  src/api/        Typed API client
docs/
  TRAINING.md     Cloud GPU deployment + hyperparameter notes
```

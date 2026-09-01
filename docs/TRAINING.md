# Training on a cloud GPU

The web app (FastAPI backend + React frontend) is designed to run on the
same machine that does the training and generation. In mock mode
(`IMUSE_MOCK_ML=1`, the default) it needs no GPU at all -- that's how you
develop and test the app itself. For real fine-tuning and image generation,
deploy the backend to a machine with an NVIDIA GPU and unset the mock flag
(`IMUSE_MOCK_ML=0`).

Any of the usual cloud GPU providers work, since the backend is just a
normal FastAPI/PyTorch process. A few concrete paths:

## Option A: RunPod (or Lambda Labs, Vast.ai, etc.)

1. Rent a pod with an NVIDIA GPU (a single 16-24GB card, e.g. RTX 4090 or
   A10, is enough for SD 1.5 LoRA fine-tuning) and a recent CUDA image.
2. Clone this repo onto the pod.
3. Build and run the GPU image:
   ```bash
   cd backend
   docker build -f Dockerfile.gpu -t imuse-backend-gpu .
   docker run --gpus all -p 8000:8000 -v imuse-data:/data imuse-backend-gpu
   ```
   (Or without Docker: `pip install -r requirements.txt -r requirements-ml.txt`,
   then `IMUSE_MOCK_ML=0 uvicorn app.main:app --host 0.0.0.0 --port 8000`.)
4. Expose port 8000 through the provider's proxy/ingress (most give you a
   public URL for an exposed port).
5. Build the frontend pointed at that URL and host it anywhere static
   (or run `npm run dev` locally with `VITE_API_URL=https://<pod-url>`).

## Option B: AWS EC2 (g5/g4dn instance)

1. Launch a `g5.xlarge` (or similar) with the "Deep Learning AMI" (comes
   with NVIDIA drivers + Docker + nvidia-container-toolkit preinstalled).
2. Same steps as above: clone repo, `docker build -f backend/Dockerfile.gpu`,
   `docker run --gpus all ...`.
3. Open port 8000 in the instance's security group (restrict to your IP,
   or put it behind a reverse proxy with auth -- see Security note below).

## Option C: Google Colab (quick experiments)

Colab doesn't run long-lived servers well, but you can still use the
*training script* directly in a notebook cell without the web app at all:

```python
!pip install -q -r requirements.txt -r requirements-ml.txt
!python training/train_lora.py \
    --instance_data_dir /content/my_photos \
    --output_dir /content/out/my_concept \
    --instance_prompt "a photo of sks person" \
    --max_train_steps 800
```

The resulting `/content/out/my_concept` directory is a LoRA adapter you can
download and later drop into the backend's `data/models/job_<id>/` layout,
or load directly with `training/inference.py --lora_dir ...`.

## Hyperparameter notes

- `--instance_prompt`: use a rare token (e.g. "sks", "ohwx") so the model
  doesn't confuse your concept with something it already knows, e.g.
  `"a photo of sks person"` or `"a painting in sks style"`.
- `--max_train_steps`: 400-1200 is a typical range for a single-concept
  LoRA on 10-30 images; more images or a more complex concept want more
  steps. Watch for overfitting (generations that look identical to training
  images) if you go much higher.
- `--lora_rank`: 4-16. Higher rank captures more detail but risks
  overfitting and produces a larger adapter file.
- `--resolution`: 512 for SD 1.5-family base models.

## Security note

Training and generation are compute-expensive, so an exposed instance is
both a cost risk and an abuse vector. The backend supports simple API-key
auth: set `IMUSE_API_KEY` to a long random value before starting it (e.g.
`export IMUSE_API_KEY=$(openssl rand -hex 32)`) and every `/api/*` route
except `/api/health` will require a matching `X-API-Key` header (or
`api_key` query param, used for `<img>` tags). If unset, the API stays
open -- fine for local-only development, not for anything with a public
URL like a RunPod proxy address.

Point the frontend at an authenticated backend by setting `VITE_API_KEY`
to the same value when running/building it.

This is enough to keep casual/opportunistic access out, but it's a single
shared static key, not real per-user auth -- for anything beyond personal
use, put it behind a proper reverse proxy (Cloudflare Access, an
authenticating nginx, etc.) instead.

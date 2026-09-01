import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from .. import config
from ..db import get_session
from ..models import GeneratedImage, JobStatus, TrainJob

router = APIRouter(prefix="/api/generate", tags=["generate"])

INFERENCE_SCRIPT = Path(__file__).resolve().parent.parent.parent / "training" / "inference.py"


class GenerateRequest(BaseModel):
    train_job_id: Optional[int] = None
    prompt: str
    negative_prompt: str = ""
    base_model: str = config.DEFAULT_BASE_MODEL
    num_images: int = 4
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    resolution: int = 512
    seed: int = -1


@router.post("", response_model=list[GeneratedImage])
def generate_images(body: GenerateRequest, session: Session = Depends(get_session)) -> list[GeneratedImage]:
    if not (1 <= body.num_images <= 8):
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 8")

    lora_dir: Optional[str] = None
    base_model = body.base_model
    if body.train_job_id is not None:
        job = session.get(TrainJob, body.train_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="train job not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="train job has not completed successfully")
        lora_dir = job.output_dir
        base_model = job.base_model

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        f"--base_model={base_model}",
        f"--prompt={body.prompt}",
        f"--negative_prompt={body.negative_prompt}",
        f"--num_images={body.num_images}",
        f"--num_inference_steps={body.num_inference_steps}",
        f"--guidance_scale={body.guidance_scale}",
        f"--resolution={body.resolution}",
        f"--seed={body.seed}",
        f"--output_dir={config.GENERATED_DIR}",
    ]
    if lora_dir:
        cmd.append(f"--lora_dir={lora_dir}")
    if config.MOCK_ML:
        cmd.append("--mock")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500, detail=f"generation failed: {result.stderr[-2000:]}"
        )

    last_json_line = next(
        (line for line in reversed(result.stdout.strip().splitlines()) if line.startswith("{")), None
    )
    if not last_json_line:
        raise HTTPException(status_code=500, detail="generation produced no output")
    image_paths = json.loads(last_json_line)["images"]

    created: list[GeneratedImage] = []
    for path in image_paths:
        record = GeneratedImage(
            train_job_id=body.train_job_id,
            base_model=base_model,
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            seed=body.seed,
            filename=Path(path).name,
        )
        session.add(record)
        created.append(record)
    session.commit()
    for record in created:
        session.refresh(record)
    return created


@router.get("/history", response_model=list[GeneratedImage])
def generation_history(session: Session = Depends(get_session)) -> list[GeneratedImage]:
    return list(session.exec(select(GeneratedImage).order_by(GeneratedImage.created_at.desc())))


@router.get("/{image_id}/file")
def get_generated_file(image_id: int, session: Session = Depends(get_session)):
    record = session.get(GeneratedImage, image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = config.GENERATED_DIR / record.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing on disk")
    return FileResponse(path)

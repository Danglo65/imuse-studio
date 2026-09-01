from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from .. import config, job_runner, storage
from ..auth import require_api_key
from ..db import get_session
from ..models import Dataset, DatasetImage, JobStatus, TrainJob

router = APIRouter(prefix="/api/train", tags=["train"], dependencies=[Depends(require_api_key)])


class TrainJobCreate(BaseModel):
    dataset_id: int
    name: str
    instance_prompt: str
    base_model: str = config.DEFAULT_BASE_MODEL
    resolution: int = 512
    max_train_steps: int = 800
    learning_rate: float = 1e-4
    lora_rank: int = 4
    seed: int = 42


@router.post("/jobs", response_model=TrainJob)
def create_job(body: TrainJobCreate, session: Session = Depends(get_session)) -> TrainJob:
    dataset = session.get(Dataset, body.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")

    image_count = len(
        list(session.exec(select(DatasetImage).where(DatasetImage.dataset_id == dataset.id)))
    )
    if image_count == 0:
        raise HTTPException(status_code=400, detail="dataset has no images to train on")

    job = TrainJob(
        dataset_id=body.dataset_id,
        name=body.name,
        instance_prompt=body.instance_prompt,
        base_model=body.base_model,
        resolution=body.resolution,
        max_train_steps=body.max_train_steps,
        learning_rate=body.learning_rate,
        lora_rank=body.lora_rank,
        seed=body.seed,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    job_runner.start_job(job.id, str(storage.dataset_dir(dataset.id)))
    return job


@router.get("/jobs", response_model=list[TrainJob])
def list_jobs(session: Session = Depends(get_session)) -> list[TrainJob]:
    return list(session.exec(select(TrainJob).order_by(TrainJob.created_at.desc())))


@router.get("/jobs/{job_id}", response_model=TrainJob)
def get_job(job_id: int, session: Session = Depends(get_session)) -> TrainJob:
    job = session.get(TrainJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: int, session: Session = Depends(get_session)) -> dict:
    job = session.get(TrainJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"logs": job_runner.read_log(job_id)}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, session: Session = Depends(get_session)) -> dict:
    job = session.get(TrainJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="cannot delete a running job")

    import shutil

    shutil.rmtree(job.output_dir, ignore_errors=True)
    session.delete(job)
    session.commit()
    return {"ok": True}

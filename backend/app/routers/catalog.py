from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ..auth import require_api_key
from ..db import get_session
from ..models import JobStatus, TrainJob

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=list[TrainJob])
def list_trained_models(session: Session = Depends(get_session)) -> list[TrainJob]:
    """Trained models are just completed training jobs -- each one produced a
    LoRA adapter directory that can be selected as a generation target."""
    return list(
        session.exec(
            select(TrainJob)
            .where(TrainJob.status == JobStatus.COMPLETED)
            .order_by(TrainJob.finished_at.desc())
        )
    )

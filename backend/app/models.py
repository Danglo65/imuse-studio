from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    filename: str
    caption: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    name: str
    base_model: str
    instance_prompt: str
    resolution: int = 512
    max_train_steps: int = 800
    learning_rate: float = 1e-4
    lora_rank: int = 4
    seed: int = 42
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    progress_step: int = 0
    progress_total: int = 0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def output_dir(self) -> str:
        from . import config

        return str(config.MODELS_DIR / f"job_{self.id}")

    @property
    def log_path(self) -> str:
        from . import config

        return str(config.JOBS_DIR / f"job_{self.id}.log")


class GeneratedImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    train_job_id: Optional[int] = Field(default=None, foreign_key="trainjob.id", index=True)
    base_model: Optional[str] = None
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

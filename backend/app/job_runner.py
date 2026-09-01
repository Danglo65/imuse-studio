"""Runs LoRA training jobs as subprocesses in background threads.

Training is a long-running, GPU-bound process, so it's kept out of the
FastAPI request/response cycle entirely: creating a job just writes a DB row
and hands off to a worker thread, which shells out to
`training/train_lora.py` and streams its progress back into the DB by
tail-parsing the log file. This keeps the API responsive and makes the
training script itself runnable standalone (e.g. copy-pasted into a Colab
cell) without any FastAPI/DB coupling.
"""

import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from . import config
from .db import engine
from .models import JobStatus, TrainJob

TRAINING_SCRIPT = Path(__file__).resolve().parent.parent / "training" / "train_lora.py"

_PROGRESS_RE = re.compile(r"IMUSE_PROGRESS step=(\d+) total=(\d+)")

_running_lock = threading.Lock()
_running_jobs: dict[int, subprocess.Popen] = {}


def start_job(job_id: int, instance_data_dir: str) -> None:
    thread = threading.Thread(target=_run_job, args=(job_id, instance_data_dir), daemon=True)
    thread.start()


def _run_job(job_id: int, instance_data_dir: str) -> None:
    with Session(engine) as session:
        job = session.get(TrainJob, job_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        session.add(job)
        session.commit()

        output_dir = job.output_dir
        log_path = job.log_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(TRAINING_SCRIPT),
            f"--instance_data_dir={instance_data_dir}",
            f"--output_dir={output_dir}",
            f"--instance_prompt={job.instance_prompt}",
            f"--base_model={job.base_model}",
            f"--resolution={job.resolution}",
            f"--max_train_steps={job.max_train_steps}",
            f"--learning_rate={job.learning_rate}",
            f"--lora_rank={job.lora_rank}",
            f"--seed={job.seed}",
        ]
        if config.MOCK_ML:
            cmd.append("--mock")

    error: str | None = None
    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            with _running_lock:
                _running_jobs[job_id] = process

            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                match = _PROGRESS_RE.search(line)
                if match:
                    _update_progress(job_id, int(match.group(1)), int(match.group(2)))

            return_code = process.wait()
            if return_code != 0:
                error = f"training process exited with code {return_code}; see logs"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the job record
        error = str(exc)
    finally:
        with _running_lock:
            _running_jobs.pop(job_id, None)

    with Session(engine) as session:
        job = session.get(TrainJob, job_id)
        if job is None:
            return
        job.finished_at = datetime.utcnow()
        job.status = JobStatus.FAILED if error else JobStatus.COMPLETED
        job.error = error
        session.add(job)
        session.commit()


def _update_progress(job_id: int, step: int, total: int) -> None:
    with Session(engine) as session:
        job = session.get(TrainJob, job_id)
        if job is None:
            return
        job.progress_step = step
        job.progress_total = total
        session.add(job)
        session.commit()


def read_log(job_id: int, tail_lines: int = 200) -> str:
    log_path = config.JOBS_DIR / f"job_{job_id}.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-tail_lines:])

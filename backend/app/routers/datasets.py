import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from .. import storage
from ..auth import require_api_key
from ..db import get_session
from ..models import Dataset, DatasetImage

router = APIRouter(prefix="/api/datasets", tags=["datasets"], dependencies=[Depends(require_api_key)])

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


class DatasetCreate(BaseModel):
    name: str
    description: str = ""


class DatasetWithImages(BaseModel):
    dataset: Dataset
    images: list[DatasetImage]


def _write_captions_file(dataset_id: int, session: Session) -> None:
    images = session.exec(
        select(DatasetImage).where(DatasetImage.dataset_id == dataset_id)
    ).all()
    captions = {img.filename: img.caption for img in images if img.caption}
    captions_path = storage.dataset_dir(dataset_id) / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2))


@router.post("", response_model=Dataset)
def create_dataset(body: DatasetCreate, session: Session = Depends(get_session)) -> Dataset:
    dataset = Dataset(name=body.name, description=body.description)
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    storage.dataset_dir(dataset.id)
    return dataset


@router.get("", response_model=list[Dataset])
def list_datasets(session: Session = Depends(get_session)) -> list[Dataset]:
    return list(session.exec(select(Dataset).order_by(Dataset.created_at.desc())))


@router.get("/{dataset_id}", response_model=DatasetWithImages)
def get_dataset(dataset_id: int, session: Session = Depends(get_session)) -> DatasetWithImages:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    images = list(
        session.exec(select(DatasetImage).where(DatasetImage.dataset_id == dataset_id))
    )
    return DatasetWithImages(dataset=dataset, images=images)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(get_session)) -> dict:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    images = list(
        session.exec(select(DatasetImage).where(DatasetImage.dataset_id == dataset_id))
    )
    for image in images:
        session.delete(image)
    session.delete(dataset)
    session.commit()

    import shutil

    shutil.rmtree(storage.dataset_dir(dataset_id), ignore_errors=True)
    return {"ok": True}


@router.post("/{dataset_id}/images", response_model=list[DatasetImage])
async def upload_images(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    captions: Optional[str] = Form(None, description="JSON array of captions, aligned with files"),
    session: Session = Depends(get_session),
) -> list[DatasetImage]:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")

    caption_list: list[str] = []
    if captions:
        try:
            caption_list = json.loads(captions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="captions must be a JSON array of strings")
        if len(caption_list) != len(files):
            raise HTTPException(status_code=400, detail="captions length must match files length")

    target_dir = storage.dataset_dir(dataset_id)
    created: list[DatasetImage] = []
    for idx, upload in enumerate(files):
        if upload.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400, detail=f"unsupported image type: {upload.content_type}"
            )
        contents = await upload.read()
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"{upload.filename} exceeds max size")

        filename = storage.safe_filename(upload.filename or "image.png")
        (target_dir / filename).write_bytes(contents)

        caption = caption_list[idx] if caption_list else ""
        image = DatasetImage(dataset_id=dataset_id, filename=filename, caption=caption)
        session.add(image)
        created.append(image)

    session.commit()
    for image in created:
        session.refresh(image)
    _write_captions_file(dataset_id, session)
    return created


@router.patch("/{dataset_id}/images/{image_id}", response_model=DatasetImage)
def update_caption(
    dataset_id: int, image_id: int, caption: str = Form(...), session: Session = Depends(get_session)
) -> DatasetImage:
    image = session.get(DatasetImage, image_id)
    if image is None or image.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="image not found")
    image.caption = caption
    session.add(image)
    session.commit()
    session.refresh(image)
    _write_captions_file(dataset_id, session)
    return image


@router.delete("/{dataset_id}/images/{image_id}")
def delete_image(dataset_id: int, image_id: int, session: Session = Depends(get_session)) -> dict:
    image = session.get(DatasetImage, image_id)
    if image is None or image.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="image not found")
    path = storage.dataset_dir(dataset_id) / image.filename
    path.unlink(missing_ok=True)
    session.delete(image)
    session.commit()
    _write_captions_file(dataset_id, session)
    return {"ok": True}


@router.get("/{dataset_id}/images/{image_id}/file")
def get_image_file(dataset_id: int, image_id: int, session: Session = Depends(get_session)):
    image = session.get(DatasetImage, image_id)
    if image is None or image.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="image not found")
    path = storage.dataset_dir(dataset_id) / image.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing on disk")
    return FileResponse(path)

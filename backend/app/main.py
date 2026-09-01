from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .db import init_db
from .routers import catalog, datasets, generate, train

app = FastAPI(title="imuse-studio", description="Train and generate images with your own concepts.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(train.router)
app.include_router(catalog.router)
app.include_router(generate.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mock_ml": config.MOCK_ML}

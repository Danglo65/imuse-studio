from typing import Optional

from fastapi import Header, HTTPException, Query

from . import config


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key: Optional[str] = Query(default=None),
) -> None:
    """No-op when IMUSE_API_KEY isn't set (local/dev). Once set, every
    dependent route requires a matching key via the X-API-Key header (used by
    the API client) or an api_key query param (used by <img> tags, which
    can't set custom headers)."""
    if not config.API_KEY:
        return
    if x_api_key == config.API_KEY or api_key == config.API_KEY:
        return
    raise HTTPException(status_code=401, detail="missing or invalid API key")

from fastapi import Header, HTTPException
from typing import Optional


async def get_user_id(x_user_id: Optional[str] = Header(default="Aymen")) -> str:
    """يستخرج user_id من الـ header أو يستخدم الافتراضي."""
    return x_user_id or "Aymen"


async def require_json_content(content_type: Optional[str] = Header(default=None)):
    """يتأكد أن الطلب يحمل JSON."""
    if content_type and "application/json" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="يجب أن يكون Content-Type: application/json",
        )

"""Legacy uvicorn target: uvicorn server:app --reload"""

from app.main import app

__all__ = ["app"]

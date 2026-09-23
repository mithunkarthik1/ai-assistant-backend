"""
Database re-export module for backward compatibility.
Points to app.core.database.
"""
from app.core.database import (
    AsyncSessionLocal,
    Base,
    engine,
    get_db,
    init_db,
)

__all__ = [
    "AsyncSessionLocal",
    "Base",
    "engine",
    "get_db",
    "init_db",
]

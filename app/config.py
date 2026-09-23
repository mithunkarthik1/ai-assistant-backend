"""
Configuration re-export module for backward compatibility.
Points to app.core.config.
"""
from app.core.config import Settings, settings

__all__ = ["Settings", "settings"]

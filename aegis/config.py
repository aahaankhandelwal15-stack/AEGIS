"""Configuration module for AEGIS."""

from dataclasses import dataclass, field
from typing import Any
import os
import tempfile


@dataclass
class Settings:
    """AEGIS settings."""
    aegis_project_id: str = "aegis"
    # Storage path for the memory pod (used by Rune, Echo, Kairo agents)
    memory_pod_storage_path: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "aegis_memory_pod")
    )


def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings()


# For backward compatibility, we can also expose the class
__all__ = ["Settings", "get_settings"]
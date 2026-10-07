import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo_root() -> Path:
    """The repository root: prompts/, instruments/ and harness/ are read from here, never written."""
    return REPO

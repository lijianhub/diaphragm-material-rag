"""Local configuration from a git-ignored ``.env`` file.

Real environment variables always win, so containers and CI configure the same
code through the environment. Values are never logged or returned, only the
names of the variables that were set, so secrets can't leak through output.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Union

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Optional[Union[str, Path]] = None) -> List[str]:
    """Set variables from ``KEY=VALUE`` lines that aren't already in the environment.

    Blank lines, comments, malformed lines and empty values are skipped. Matching
    single or double quotes around a value are removed. Returns the names that were
    set, never the values.
    """
    path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not path.is_file():
        return []
    loaded: List[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded

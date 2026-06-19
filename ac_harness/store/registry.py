# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Public store facade.

Importers and downstream code should use `EvidenceStore` from this module
rather than reaching into `sqlite_store.py` directly. If a new backend
is added later (e.g. Postgres), this is where the abstraction will live.
"""
from __future__ import annotations

from .sqlite_store import EvidenceStore, new_id

__all__ = ["EvidenceStore", "new_id"]

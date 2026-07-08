"""Backend auth security tests. Skipped automatically if backend deps not installed."""
import importlib
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "frontend" / "backend"

pytest.importorskip("jose")
pytest.importorskip("fastapi")
pytest.importorskip("motor")


def _import_auth_fresh(monkeypatch):
    monkeypatch.syspath_prepend(str(BACKEND))
    # db_client calls load_dotenv() at import, which would re-populate
    # JWT_SECRET from the local .env and defeat the env manipulation below.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    # db_client requires MONGO_URL at import; no connection is made in these tests.
    monkeypatch.setenv("MONGO_URL", "mongodb://localhost:27017")
    for mod in ("auth", "db", "db_client"):
        sys.modules.pop(mod, None)
    return importlib.import_module("auth")


def test_production_requires_jwt_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        _import_auth_fresh(monkeypatch)


def test_dev_falls_back_to_dev_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    auth = _import_auth_fresh(monkeypatch)
    assert auth.JWT_SECRET  # dev fallback present, app importable


def test_explicit_secret_wins(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "x" * 48)
    auth = _import_auth_fresh(monkeypatch)
    assert auth.JWT_SECRET == "x" * 48

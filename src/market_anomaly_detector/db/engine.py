"""Database engine factory — reads MAD_DATABASE_URL from the environment."""

from __future__ import annotations

import os

import sqlalchemy as sa

DATABASE_URL_ENV = "MAD_DATABASE_URL"
_DEFAULT_DATABASE_URL = "postgresql+psycopg://mad:mad@localhost:5432/mad"


def get_database_url() -> str:
    return os.environ.get(DATABASE_URL_ENV, _DEFAULT_DATABASE_URL)


def create_engine(url: str | None = None, **kwargs: object) -> sa.Engine:
    return sa.create_engine(url or get_database_url(), **kwargs)

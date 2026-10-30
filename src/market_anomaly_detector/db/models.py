"""SQLAlchemy 2 declarative models for the Market Anomaly Detector database."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class InstrumentRow(Base):
    __tablename__ = "instruments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    asset_class: Mapped[str] = mapped_column(String(32))
    units: Mapped[str] = mapped_column(String(32))
    quote: Mapped[str] = mapped_column(String(128))
    native_frequency: Mapped[str] = mapped_column(String(16))
    valid_from: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    valid_to: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    groups: Mapped[list[InstrumentGroupRow]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )
    sources: Mapped[list[SymbolMapRow]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )


class InstrumentGroupRow(Base):
    __tablename__ = "instrument_groups"

    instrument_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    group: Mapped[str] = mapped_column(String(64), primary_key=True)

    instrument: Mapped[InstrumentRow] = relationship(back_populates="groups")


class SymbolMapRow(Base):
    __tablename__ = "symbol_map"

    instrument_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    priority: Mapped[int] = mapped_column(Integer)
    fidelity: Mapped[str] = mapped_column(String(16))
    source_frequency: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    publication_lag_days: Mapped[int] = mapped_column(Integer, default=0)
    revised: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")

    instrument: Mapped[InstrumentRow] = relationship(back_populates="sources")


class PriceRow(Base):
    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("instrument_id", "provider", "obs_date"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(String(64), ForeignKey("instruments.id"))
    provider: Mapped[str] = mapped_column(String(32))
    obs_date: Mapped[date] = mapped_column(Date)
    value: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    known_at: Mapped[date] = mapped_column(Date)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("ingestion_runs.id"), nullable=True
    )


class LabelRow(Base):
    __tablename__ = "labels"

    label_set: Mapped[str] = mapped_column(String(64), primary_key=True)
    week: Mapped[date] = mapped_column(Date, primary_key=True)
    value: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(128))


class IngestionRunRow(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    mode: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    instruments_ok: Mapped[int] = mapped_column(Integer, default=0)
    instruments_failed: Mapped[int] = mapped_column(Integer, default=0)
    rows_written: Mapped[int] = mapped_column(BigInteger, default=0)

    items: Mapped[list[IngestionRunItemRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class IngestionRunItemRow(Base):
    __tablename__ = "ingestion_run_items"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ingestion_runs.id", ondelete="CASCADE"), primary_key=True
    )
    instrument_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    rows_written: Mapped[int] = mapped_column(BigInteger, default=0)
    rows_new: Mapped[int] = mapped_column(BigInteger, default=0)
    rows_revised: Mapped[int] = mapped_column(BigInteger, default=0)
    first_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped[IngestionRunRow] = relationship(back_populates="items")


class DataQualityFlagRow(Base):
    __tablename__ = "data_quality_flags"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("ingestion_runs.id"), nullable=True
    )
    instrument_id: Mapped[str] = mapped_column(String(64), ForeignKey("instruments.id"))
    provider: Mapped[str] = mapped_column(String(32))
    obs_date: Mapped[date] = mapped_column(Date)
    rule: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text)


class ModelVersionRow(Base):
    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String(128), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    label_set: Mapped[str] = mapped_column(String(64))
    spec_hash: Mapped[str] = mapped_column(String(64))
    config_hash: Mapped[str] = mapped_column(String(64))
    heads: Mapped[str] = mapped_column(Text)
    metrics: Mapped[str] = mapped_column(Text)
    threshold: Mapped[float] = mapped_column(Float)
    platt_a: Mapped[float] = mapped_column(Float)
    platt_b: Mapped[float] = mapped_column(Float)
    artifact_path: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=False)

    scores: Mapped[list["RegimeScoreRow"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )


class RegimeScoreRow(Base):
    __tablename__ = "regime_scores"

    week: Mapped[date] = mapped_column(Date, primary_key=True)
    model_version: Mapped[str] = mapped_column(
        String(128), ForeignKey("model_versions.version"), primary_key=True
    )
    head: Mapped[str] = mapped_column(String(16), primary_key=True)
    probability: Mapped[float] = mapped_column(Float)
    alert: Mapped[bool] = mapped_column(Boolean)
    threshold: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32))
    top_features: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    model: Mapped[ModelVersionRow] = relationship(back_populates="scores")

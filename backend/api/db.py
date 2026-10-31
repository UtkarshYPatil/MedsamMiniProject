"""Durable records: cases and the jobs run against them.

Transient job state lives in Redis (RQ owns it). What lands here is what should
outlive the queue -- which case was segmented, with what prompt, and what was
measured -- so a result can be audited after the fact.
"""
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from backend.common import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    id:         Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source:     Mapped[str] = mapped_column(String(32), default="upload")
    note:       Mapped[str] = mapped_column(String(256), default="")

    shape_z: Mapped[int] = mapped_column(Integer)
    shape_y: Mapped[int] = mapped_column(Integer)
    shape_x: Mapped[int] = mapped_column(Integer)

    spacing_z: Mapped[float] = mapped_column(Float)
    spacing_y: Mapped[float] = mapped_column(Float)
    spacing_x: Mapped[float] = mapped_column(Float)

    def as_dict(self):
        return {
            "case_id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source": self.source,
            "note": self.note,
            "shape": [self.shape_z, self.shape_y, self.shape_x],
            "spacing_mm": [self.spacing_z, self.spacing_y, self.spacing_x],
        }


class Job(Base):
    __tablename__ = "jobs"

    id:      Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(32), ForeignKey("cases.id"), index=True)

    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    prompt_z: Mapped[int] = mapped_column(Integer)
    box:      Mapped[list] = mapped_column(JSON)

    status:  Mapped[str] = mapped_column(String(16), default="queued", index=True)
    error:   Mapped[str | None] = mapped_column(String(2048), nullable=True)
    result:  Mapped[dict | None] = mapped_column(JSON, nullable=True)

    device:  Mapped[str] = mapped_column(String(16), default="cpu")
    dry_run: Mapped[bool] = mapped_column(Boolean, default=False)
    seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    def as_dict(self):
        return {
            "job_id": self.id,
            "case_id": self.case_id,
            "status": self.status,
            "prompt_slice": self.prompt_z,
            "box": self.box,
            "device": self.device,
            "dry_run": self.dry_run,
            "seconds": self.seconds,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


def init_db(retries=30, delay=1.0):
    """Create tables, waiting for Postgres to accept connections first."""
    import time
    from sqlalchemy.exc import OperationalError

    last = None
    for _ in range(retries):
        try:
            Base.metadata.create_all(engine)
            return
        except OperationalError as e:       # db container still starting
            last = e
            time.sleep(delay)
    raise RuntimeError(f"database never became available: {last}")

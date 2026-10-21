from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    split: Mapped[str] = mapped_column(String(20))
    agent_version: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="processing")
    metadata_json: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    scores: Mapped[dict | None] = mapped_column(JSON_TYPE, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    answers: Mapped[list["AnswerOutcome"]] = relationship(cascade="all, delete-orphan")


class AnswerOutcome(Base):
    __tablename__ = "answer_outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"))
    answer_id: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))
    error_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    claims: Mapped[list["ClaimOutcome"]] = relationship(cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("run_id", "answer_id", name="uq_answer_run_answer"),)


class ClaimOutcome(Base):
    __tablename__ = "claim_outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_outcome_id: Mapped[int] = mapped_column(ForeignKey("answer_outcomes.id", ondelete="CASCADE"))
    run_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"))
    answer_id: Mapped[str] = mapped_column(String(50))
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(20))
    fact_ids: Mapped[list] = mapped_column(JSON_TYPE, default=list)
    reason: Mapped[str] = mapped_column(Text, default="")
    __table_args__ = (UniqueConstraint("run_id", "answer_id", "start_char", "end_char", name="uq_claim_run_span"),)


def make_session_factory(database_url: str | None = None):
    url = database_url or os.environ["DATABASE_URL"]
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)

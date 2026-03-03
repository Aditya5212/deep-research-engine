from typing import Optional

from sqlalchemy import Index, Integer, LargeBinary, PrimaryKeyConstraint, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass


class CheckpointBlobs(Base):
    __tablename__ = 'checkpoint_blobs'
    __table_args__ = (
        PrimaryKeyConstraint('thread_id', 'checkpoint_ns', 'channel', 'version', name='checkpoint_blobs_pkey'),
        Index('checkpoint_blobs_thread_id_idx', 'thread_id')
    )

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("''::text"))
    channel: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[str] = mapped_column(Text, primary_key=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary)


class CheckpointMigrations(Base):
    __tablename__ = 'checkpoint_migrations'
    __table_args__ = (
        PrimaryKeyConstraint('v', name='checkpoint_migrations_pkey'),
    )

    v: Mapped[int] = mapped_column(Integer, primary_key=True)


class CheckpointWrites(Base):
    __tablename__ = 'checkpoint_writes'
    __table_args__ = (
        PrimaryKeyConstraint('thread_id', 'checkpoint_ns', 'checkpoint_id', 'task_id', 'idx', name='checkpoint_writes_pkey'),
        Index('checkpoint_writes_thread_id_idx', 'thread_id')
    )

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("''::text"))
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)
    task_id: Mapped[str] = mapped_column(Text, primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(Text, nullable=False)
    blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    task_path: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''::text"))
    type: Mapped[Optional[str]] = mapped_column(Text)


class Checkpoints(Base):
    __tablename__ = 'checkpoints'
    __table_args__ = (
        PrimaryKeyConstraint('thread_id', 'checkpoint_ns', 'checkpoint_id', name='checkpoints_pkey'),
        Index('checkpoints_thread_id_idx', 'thread_id')
    )

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("''::text"))
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metadata_: Mapped[dict] = mapped_column('metadata', JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    parent_checkpoint_id: Mapped[Optional[str]] = mapped_column(Text)
    type: Mapped[Optional[str]] = mapped_column(Text)

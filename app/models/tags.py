from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.conversation import Base


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    color: Mapped[str] = mapped_column(String(20), default="#6366f1")  # hex color

    conversation_tags: Mapped[list["ConversationTag"]] = relationship(
        back_populates="tag", cascade="all, delete-orphan"
    )
    document_tags: Mapped[list["DocumentTag"]] = relationship(
        back_populates="tag", cascade="all, delete-orphan"
    )


class ConversationTag(Base):
    __tablename__ = "conversation_tags"
    __table_args__ = (UniqueConstraint("conversation_id", "tag_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE")
    )

    tag: Mapped["Tag"] = relationship(back_populates="conversation_tags")


class DocumentTag(Base):
    __tablename__ = "document_tags"
    __table_args__ = (UniqueConstraint("document_id", "tag_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE")
    )

    tag: Mapped["Tag"] = relationship(back_populates="document_tags")

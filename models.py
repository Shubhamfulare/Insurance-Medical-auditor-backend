from sqlalchemy import Column, Integer, String, Text
from database import Base
import enum
from sqlalchemy import (
    Column, Integer, String, Date, Text,
    DateTime, Enum, ForeignKey
)
from sqlalchemy.sql import func

class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True)
    ocr_text = Column(Text)

class ClaimStatus(enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)

    # Relations
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)

    # Top-level fields
    status = Column(Enum(ClaimStatus), default=ClaimStatus.PENDING)
    reason = Column(String, nullable=True)

    policy_number = Column(String, index=True, nullable=True)
    patient_id = Column(String, index=True, nullable=True)

    # Claim details (flattened from JSON)
    diagnosis = Column(String)
    hospital_name = Column(String)

    doctor_id = Column(String)
    doctor_name = Column(String)
    doctor_notes = Column(Text)

    visit_date = Column(Date)

    # Audit fields
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

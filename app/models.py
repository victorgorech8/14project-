from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from .database import Base

class RequestHistory(Base):
    __tablename__ = "request_history"
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(50))
    method = Column(String(10))
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    processing_time = Column(Float)
    status_code = Column(Integer)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, nullable=True)


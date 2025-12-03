from sqlalchemy import (Column, Integer, String, DateTime,
                        FLOAT, Boolean, ForeignKey, TEXT, false)
from conect_databse.database import Base
from datetime import datetime
from typing import Optional
import enum

class KnowledgeItemORM(Base):
    __tablename__ = 'edu_knowledge_item'
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    knowledgeName = Column(String(94), nullable=False)
    activate = Column(Boolean, default=False)
    deleted = Column(Boolean, default=False, server_default=false(), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    corpus_id = Column(TEXT)
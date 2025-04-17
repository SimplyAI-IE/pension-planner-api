# --- models.py ---
from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine, Boolean # Add Boolean potentially
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'

    user_id = Column(String, primary_key=True)
    region = Column(String)
    age = Column(Integer)
    income = Column(Integer)
    retirement_age = Column(Integer)
    risk_profile = Column(String)
    prsi_years = Column(Integer) 
    pending_action = Column(String, nullable=True) # To store state like 'offer_tips'

class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True)  # Google sub
    name = Column(String)
    email = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# New table to store conversation history
class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    role = Column(String) # 'user' or 'assistant'
    content = Column(Text) # Use Text for potentially longer messages
    timestamp = Column(DateTime, default=datetime.utcnow)


engine = create_engine("sqlite:///memory.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    # Ensure all tables, including the new ChatHistory, are created
    Base.metadata.create_all(bind=engine)
# --- End of models.py ---
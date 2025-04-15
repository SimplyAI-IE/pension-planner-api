from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
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

class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True)  # Google sub
    name = Column(String)
    email = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///memory.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

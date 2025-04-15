from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'

    user_id = Column(String, primary_key=True)
    region = Column(String)
    age = Column(Integer)
    income = Column(Integer)
    retirement_age = Column(Integer)
    risk_profile = Column(String)

# SQLite engine (file-based)
engine = create_engine("sqlite:///memory.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

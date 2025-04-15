from models import UserProfile, SessionLocal

def get_user_profile(user_id):
    db = SessionLocal()
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    db.close()
    return profile

def save_user_profile(user_id, field, value):
    db = SessionLocal()
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

    if not profile:
        profile = UserProfile(user_id=user_id)
        db.add(profile)

    setattr(profile, field, value)
    db.commit()
    db.close()

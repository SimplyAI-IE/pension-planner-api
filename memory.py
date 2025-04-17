# --- memory.py ---
from models import UserProfile, ChatHistory, SessionLocal
from sqlalchemy import desc
import logging

# Configure logging
logger = logging.getLogger(__name__)

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

    if value is not None:
        if not hasattr(profile, field):
            logger.error(f"Field '{field}' does not exist in UserProfile for user {user_id}")
            db.close()
            return
        setattr(profile, field, value)

    try:
        db.commit()
        logger.debug(f"Saved {field}='{value}' for user {user_id}")
    except Exception as e:
        logger.error(f"Database error saving profile for {user_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

def save_chat_message(user_id, role, content):
    if not user_id or not role or not content:
        logger.warning(f"Attempted to save incomplete chat message for user {user_id}. Role: {role}, Content: '{content}'")
        return

    db = SessionLocal()
    message = ChatHistory(
        user_id=user_id,
        role=role,
        content=content
    )
    db.add(message)
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Database error saving chat message for {user_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

def get_chat_history(user_id, limit=10):
    if not user_id:
        return []
    db = SessionLocal()
    try:
        history = db.query(ChatHistory)\
                    .filter(ChatHistory.user_id == user_id)\
                    .order_by(desc(ChatHistory.timestamp))\
                    .limit(limit)\
                    .all()
    except Exception as e:
        logger.error(f"Database error retrieving chat history for {user_id}: {e}", exc_info=True)
        history = []
    finally:
        db.close()
    return [{"role": m.role, "content": m.content} for m in history[::-1]]
# --- End of memory.py ---
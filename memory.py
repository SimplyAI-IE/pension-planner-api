# --- memory.py ---
from models import UserProfile, ChatHistory, SessionLocal
from sqlalchemy import desc

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

    # Only update if value is not None and not blank
    if value is not None: # Check specifically for None, allow 0 or empty strings if needed
        setattr(profile, field, value)

    try:
        db.commit()
    except Exception as e:
        print(f"Database error saving profile for {user_id}: {e}")
        db.rollback()
    finally:
        db.close()

# New function to save a message to history
def save_chat_message(user_id, role, content):
    if not user_id or not role or not content:
        print(f"Warning: Attempted to save incomplete chat message for user {user_id}. Role: {role}, Content: '{content}'")
        return # Avoid saving empty/incomplete messages

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
        print(f"Database error saving chat message for {user_id}: {e}")
        db.rollback()
    finally:
        db.close()

# New function to retrieve chat history
def get_chat_history(user_id, limit=10):
    """Retrieves the most recent 'limit' messages for a user, oldest first."""
    if not user_id:
        return []
    db = SessionLocal()
    try:
        # Fetch the last 'limit' messages ordered by timestamp descending,
        # then reverse the list so they are in chronological order for the API.
        history = db.query(ChatHistory)\
                    .filter(ChatHistory.user_id == user_id)\
                    .order_by(desc(ChatHistory.timestamp))\
                    .limit(limit)\
                    .all()
    except Exception as e:
        print(f"Database error retrieving chat history for {user_id}: {e}")
        history = [] # Return empty list on error
    finally:
        db.close()
    return history[::-1] # Reverse to get chronological order (oldest first)
# --- End of memory.py ---
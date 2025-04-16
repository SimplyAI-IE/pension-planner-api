# --- main.py ---
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from gpt_engine import get_gpt_response
from memory import save_user_profile, save_chat_message # Added save_chat_message
from models import init_db, User, SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import re
import logging # Added logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

# Allow CORS so frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider changing to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB table(s) - This will now include ChatHistory
logger.info("Initializing database...")
init_db()
logger.info("Database initialized.")

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    user_message = req.message.strip()
    logger.info(f"Received chat request from user_id: {user_id}")

    # Avoid processing or saving special commands or empty messages
    is_special_command = user_message == "__INIT__"
    if not user_message and not is_special_command:
        logger.warning(f"Received empty message from user_id: {user_id}")
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Extract data before saving potentially sensitive raw message
    if not is_special_command:
         try:
             extract_user_data(user_id, user_message)
         except Exception as e:
             logger.error(f"Error extracting data for user {user_id}: {e}", exc_info=True)
             # Decide if you want to stop processing or just log the error

    # Get the reply from the AI (which now uses history)
    logger.info(f"Getting GPT response for user_id: {user_id}")
    try:
        reply = get_gpt_response(user_message, user_id)
        logger.info(f"GPT response generated successfully for user_id: {user_id}")
    except Exception as e:
        logger.error(f"Error getting GPT response for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing chat request")


    # Save user message and assistant reply to history, unless it's the init message
    if not is_special_command:
        logger.info(f"Saving user message to history for user_id: {user_id}")
        save_chat_message(user_id, 'user', user_message) # Save the original user message
    if reply and not is_special_command:
        logger.info(f"Saving assistant reply to history for user_id: {user_id}")
        save_chat_message(user_id, 'assistant', reply)

    return {"response": reply}

def extract_user_data(user_id, msg):
    logger.debug(f"Extracting data from message for user_id: {user_id}")
    # Use lowercase message for easier matching
    msg_lower = msg.lower()

    # Region Extraction
    if "ireland" in msg_lower:
        save_user_profile(user_id, "region", "Ireland")
        logger.debug(f"Saved region 'Ireland' for user_id: {user_id}")
    elif "uk" in msg_lower or "united kingdom" in msg_lower:
        save_user_profile(user_id, "region", "UK")
        logger.debug(f"Saved region 'UK' for user_id: {user_id}")

    # Age Extraction (looks for NN years old)
    age_match = re.search(r"\b(\d{1,2})\s*(?:years?)?\s*old\b", msg_lower)
    if age_match:
        age = int(age_match.group(1))
        if 18 <= age <= 100: # Basic sanity check
             save_user_profile(user_id, "age", age)
             logger.debug(f"Saved age '{age}' for user_id: {user_id}")

    # Income Extraction (handles £/€, commas, k/K) - improved regex
    # Looks for currency symbol (optional), digits (with optional commas), k/K (optional)
    income_match = re.search(r"\b(?:€|£)\s?(\d{1,3}(?:,\d{3})*|\d+)\s?([kK]?)\b", msg.replace(",", ""))
    if income_match:
        income_val = int(income_match.group(1))
        if income_match.group(2).lower() == 'k':
            income_val *= 1000
        save_user_profile(user_id, "income", income_val)
        logger.debug(f"Saved income '{income_val}' for user_id: {user_id}")

    # Retirement Age Extraction (looks for retire at/by NN)
    retirement_match = re.search(r"\b(?:retire|retirement)\b.*?\b(\d{2})\b", msg_lower)
    if retirement_match:
        ret_age = int(retirement_match.group(1))
        if 50 <= ret_age <= 80: # Basic sanity check
            save_user_profile(user_id, "retirement_age", ret_age)
            logger.debug(f"Saved retirement age '{ret_age}' for user_id: {user_id}")

    # Risk Profile Extraction
    if "risk" in msg_lower:
        if "low" in msg_lower:
            save_user_profile(user_id, "risk_profile", "Low")
            logger.debug(f"Saved risk profile 'Low' for user_id: {user_id}")
        elif "high" in msg_lower:
            save_user_profile(user_id, "risk_profile", "High")
            logger.debug(f"Saved risk profile 'High' for user_id: {user_id}")
        elif "medium" in msg_lower or "moderate" in msg_lower:
            save_user_profile(user_id, "risk_profile", "Medium")
            logger.debug(f"Saved risk profile 'Medium' for user_id: {user_id}")

@app.post("/auth/google")
async def auth_google(user_data: dict):
    if not user_data or "sub" not in user_data:
         logger.error("Invalid user data received in /auth/google")
         raise HTTPException(status_code=400, detail="Invalid user data received")

    user_id = user_data["sub"]
    logger.info(f"Processing Google auth for user_id: {user_id}")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            logger.info(f"New user detected, creating entry for user_id: {user_id}")
            user = User(
                id=user_id,
                name=user_data.get("name", "Unknown User"), # Use .get for safety
                email=user_data.get("email")
            )
            db.add(user)
            db.commit()
            db.refresh(user) # Ensure user object has updated data like created_at
            logger.info(f"Successfully created user entry for user_id: {user_id}")
        else:
             logger.info(f"Existing user found for user_id: {user_id}")

    except Exception as e:
        db.rollback()
        logger.error(f"Database error during auth for user_id {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database operation failed")
    finally:
        db.close()

    return {"status": "ok", "user_id": user_id}

# Optional: Add a root endpoint for health check or basic info
@app.get("/")
async def root():
    return {"message": "Pension Planner API is running"}

# --- End of main.py ---
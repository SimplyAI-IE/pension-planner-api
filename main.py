# --- main.py ---
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from gpt_engine import get_gpt_response
# Ensure get_user_profile is imported if not already implicitly available via memory import
from memory import get_user_profile, save_user_profile, save_chat_message
from models import init_db, User, SessionLocal, UserProfile, ChatHistory # Import all needed models
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from weasyprint import HTML # Ensure WeasyPrint is installed
from io import BytesIO
import re
import logging
import os
from typing import Optional # Import Optional for type hinting if needed

os.environ["G_MESSAGES_DEBUG"] = ""

# Configure logging
# Consider increasing level to INFO for debugging state changes if needed
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

# Create DB table(s)
logger.info("Initializing database...")
# IMPORTANT: Ensure this runs correctly and includes the new pending_action column
# Remember to delete memory.db and run `python init_db.py` if you changed models.py
init_db()
logger.info("Database initialized.")

class ChatRequest(BaseModel):
    user_id: str
    message: str
    tone: str = ""  # Optional tone setting

# Set of affirmative responses for state checking
affirmative_responses = {"sure", "yes", "ok", "okay", "fine", "yep", "please", "yes please"}

@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    user_message = req.message.strip()
    user_message_lower = user_message.lower() # For case-insensitive checks
    logger.info(f"Received chat request from user_id: {user_id}, message: '{user_message}'")

    # --- Handle __INIT__ separately ---
    is_special_command = user_message == "__INIT__"
    if is_special_command:
        logger.info(f"Handling __INIT__ command for user_id: {user_id}")
        reply = get_gpt_response(user_message, user_id, tone=req.tone)
        # Don't save __INIT__ commands or their immediate response to history usually
        # Also don't set pending_action based on init response
        return {"response": reply}

    if not user_message:
        logger.warning(f"Received empty message from user_id: {user_id}")
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # --- State Handling Logic ---
    profile = get_user_profile(user_id)
    give_tips_directly = False
    # Check if profile exists and has the pending_action attribute
    # Ensure profile exists and has pending_action before checking its value
    if profile and hasattr(profile, 'pending_action') and profile.pending_action == "offer_tips":
        logger.info(f"User {user_id} has pending_action 'offer_tips'. Checking response: '{user_message_lower}'")
        # Clear the pending action immediately after checking it
        save_user_profile(user_id, "pending_action", None)
        logger.info(f"Cleared pending_action for user {user_id}")

        if user_message_lower in affirmative_responses:
            logger.info(f"User {user_id} confirmed wanting tips. Handling directly.")
            give_tips_directly = True
        else:
            logger.info(f"User {user_id} did not give affirmative response to tips offer. Proceeding normally.")
            # If user said something else (e.g., asked another question), let the normal flow handle it

    # --- Direct Tips Response (if triggered) ---
    if give_tips_directly:
        reply = "" # Initialize reply
        try:
            logger.info(f"Generating direct tips response for user {user_id}")
            # Using a predefined response for reliability:
            reply = (
                "Great! Here are a few common ways people in Ireland can look into boosting their State Pension:\n\n"
                "1.  **Keep Contributing:** Working and paying PRSI for the full 40 years generally leads to the maximum pension.\n"
                "2.  **Check for Gaps & Voluntary Contributions:** If you have gaps in your record (e.g., time abroad or not working), see if you're eligible to make voluntary contributions to fill them. You can check this on MyWelfare.ie.\n"
                "3.  **Look into Credits:** Certain periods, like time spent caring for children or incapacitated individuals (HomeCaring Periods), or receiving some social welfare payments, might entitle you to credits that count towards your pension.\n\n"
                "Does that make sense? It's always best to check your personal record on MyWelfare.ie or consult with Citizens Information or a financial advisor for advice tailored to you."
            )
            logger.info(f"Generated predefined tips for user {user_id}")

        except Exception as e:
             logger.error(f"Error generating tips directly for user {user_id}: {e}", exc_info=True)
             # Fallback reply if generation fails
             reply = "Okay, I can definitely give you tips. Generally, working longer, making voluntary contributions if eligible, or checking for credits can help boost your State Pension. Checking MyWelfare.ie is a good starting point."

        # Save the user message ("sure") and the direct tips reply
        save_chat_message(user_id, 'user', user_message)
        if reply: # Ensure reply is not None before saving
             save_chat_message(user_id, 'assistant', reply)
        return {"response": reply}

    # --- Standard Chat Flow (if not handling tips directly) ---
    logger.info(f"Proceeding with standard chat flow for user {user_id}")

    # Extract data (only if not a special command/state handled above)
    try:
        extract_user_data(user_id, user_message)
        # Re-fetch profile in case extract_user_data modified it and we need updated state below
        profile = get_user_profile(user_id)
    except Exception as e:
        logger.error(f"Error extracting data for user {user_id}: {e}", exc_info=True)

    # Get the reply from the AI
    reply = "" # Initialize reply
    try:
        reply = get_gpt_response(user_message, user_id, tone=req.tone)
        logger.info(f"GPT response generated successfully for user_id: {user_id}")
    except Exception as e:
        logger.error(f"Error getting GPT response for user {user_id}: {e}", exc_info=True)
        # Provide a fallback message instead of raising HTTPException
        reply = "I'm sorry, I encountered a technical issue trying to process that. Could you try rephrasing?"
        # Log the error but allow flow to continue to save messages

    # Save user message and assistant reply to history
    save_chat_message(user_id, 'user', user_message)
    if reply: # Ensure reply is not None or empty before saving
        save_chat_message(user_id, 'assistant', reply)

    # --- State Setting Logic ---
    # Check if the *newly generated* reply offers tips, set pending_action if so
    # Ensure profile exists and has the pending_action attribute before trying to set it
    if profile and hasattr(profile, 'pending_action'):
        # Keywords to check for in the bot's reply to see if it's offering tips
        offer_keywords = ["would you like tips", "improve your pension?", "boost your pension?"]
        reply_lower = reply.lower() if reply else ""

        if any(keyword in reply_lower for keyword in offer_keywords):
            logger.info(f"Bot offered tips to user {user_id}. Setting pending_action='offer_tips'.")
            save_user_profile(user_id, "pending_action", "offer_tips")
        # No 'else' here - we only clear the flag when the user responds to the offer (handled at the top)
    elif profile and not hasattr(profile, 'pending_action'):
        logger.warning(f"Profile for user {user_id} exists but missing 'pending_action' attribute. Cannot set state.")
    # else: # Profile doesn't exist, log warning? Maybe handled by get_user_profile returning None
        # logger.warning(f"Profile not found for user {user_id}. Cannot set pending_action state.")


    return {"response": reply}


def extract_user_data(user_id, msg):
    # Check if UserProfile model itself has the attributes - requires access to model definition
    # This is more complex, rely on save_user_profile handling errors for now or add checks within save_user_profile
    logger.debug(f"Extracting data from message for user_id: {user_id}")
    msg_lower = msg.lower()
    profile_updated = False

    # Region Extraction
    if "ireland" in msg_lower:
        save_user_profile(user_id, "region", "Ireland")
        logger.debug(f"Saved region 'Ireland' for user_id: {user_id}")
        profile_updated = True
    elif "uk" in msg_lower or "united kingdom" in msg_lower:
        save_user_profile(user_id, "region", "UK")
        logger.debug(f"Saved region 'UK' for user_id: {user_id}")
        profile_updated = True

    # Age Extraction
    age_match = re.search(r"\b(\d{1,2})\s*(?:years?)?\s*old\b", msg_lower)
    if age_match: # Removed hasattr check here, assume save_user_profile handles it
        age = int(age_match.group(1))
        if 18 <= age <= 100:
             save_user_profile(user_id, "age", age)
             logger.debug(f"Saved age '{age}' for user_id: {user_id}")
             profile_updated = True

    # Income Extraction
    income_match = re.search(r"\b(?:€|£)\s?(\d{1,3}(?:,\d{3})*|\d+)\s?([kK]?)\b", msg.replace(",", ""))
    if income_match: # Removed hasattr check
        income_val = int(income_match.group(1))
        if income_match.group(2).lower() == 'k':
            income_val *= 1000
        save_user_profile(user_id, "income", income_val)
        logger.debug(f"Saved income '{income_val}' for user_id: {user_id}")
        profile_updated = True

    # Retirement Age Extraction
    retirement_match = re.search(r"\b(?:retire|retirement)\b.*?\b(\d{2})\b", msg_lower)
    if retirement_match: # Removed hasattr check
        ret_age = int(retirement_match.group(1))
        if 50 <= ret_age <= 80:
            save_user_profile(user_id, "retirement_age", ret_age)
            logger.debug(f"Saved retirement age '{ret_age}' for user_id: {user_id}")
            profile_updated = True

    # Risk Profile Extraction
    if "risk" in msg_lower: # Removed hasattr check
        if "low" in msg_lower:
            save_user_profile(user_id, "risk_profile", "Low")
            logger.debug(f"Saved risk profile 'Low' for user_id: {user_id}")
            profile_updated = True
        elif "high" in msg_lower:
            save_user_profile(user_id, "risk_profile", "High")
            logger.debug(f"Saved risk profile 'High' for user_id: {user_id}")
            profile_updated = True
        elif "medium" in msg_lower or "moderate" in msg_lower:
            save_user_profile(user_id, "risk_profile", "Medium")
            logger.debug(f"Saved risk profile 'Medium' for user_id: {user_id}")
            profile_updated = True

    # PRSI Contributions Extraction
    # Relying on save_user_profile to handle potential missing 'prsi_years' attribute gracefully,
    # or assuming models.py is updated correctly.
    prsi_match = re.search(r"(\d{1,2})\s+(?:years?|yrs?)\s+(?:of\s+)?(?:prsi|contributions?)", msg_lower)
    if prsi_match:
        prsi_years = int(prsi_match.group(1))
        save_user_profile(user_id, "prsi_years", prsi_years)
        logger.debug(f"Saved PRSI years '{prsi_years}' from detailed message for user_id: {user_id}")
        profile_updated = True
    elif re.fullmatch(r"\d{1,2}", msg.strip()):
        potential_years = int(msg.strip())
        if 0 <= potential_years <= 60:
             # Consider adding context check here (e.g., was last bot msg asking for PRSI?)
             save_user_profile(user_id, "prsi_years", potential_years)
             logger.debug(f"Saved PRSI years '{potential_years}' from simple number reply for user_id: {user_id}")
             profile_updated = True


    if profile_updated:
        logger.info(f"Profile data potentially updated for user_id: {user_id}")


@app.post("/auth/google")
async def auth_google(user_data: dict):
    if not user_data or "sub" not in user_data:
         logger.error("Invalid user data received in /auth/google")
         raise HTTPException(status_code=400, detail="Invalid user data received")

    user_id = user_data["sub"]
    logger.info(f"Processing Google auth for user_id: {user_id}")
    db = SessionLocal()
    try:
        # Check if user exists
        user = db.query(User).filter(User.id == user_id).first()
        # Check if profile exists separately
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

        if not user:
            logger.info(f"New user detected, creating user and profile entry for user_id: {user_id}")
            user = User(
                id=user_id,
                name=user_data.get("name", "Unknown User"),
                email=user_data.get("email")
            )
            db.add(user)
            # Create profile only if it doesn't exist
            if not profile:
                profile = UserProfile(user_id=user_id) # Initialize profile
                db.add(profile)
            db.commit()
            db.refresh(user)
            logger.info(f"Successfully created user and profile entry for user_id: {user_id}")
        else:
             logger.info(f"Existing user found for user_id: {user_id}")
             # Ensure profile exists if user exists but profile might be missing
             if not profile:
                 logger.warning(f"Existing user {user_id} found, but profile missing. Creating profile.")
                 profile = UserProfile(user_id=user_id)
                 db.add(profile)
                 db.commit()

    except Exception as e:
        db.rollback()
        logger.error(f"Database error during auth for user_id {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database operation failed")
    finally:
        db.close()

    return {"status": "ok", "user_id": user_id}


@app.get("/")
async def root():
    return {"message": "Pension Planner API is running"}


@app.get("/export-pdf")
async def export_pdf(user_id: str):
    profile = get_user_profile(user_id)

    if not profile:
        raise HTTPException(status_code=404, detail="No profile found for PDF export.")

    db = SessionLocal()
    messages = []
    try:
        messages = (
            db.query(ChatHistory)
            .filter(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.timestamp)
            .all()
        )
    except Exception as e:
        logger.error(f"Failed to retrieve chat history for PDF export for user {user_id}: {e}")
        # Continue with empty messages list
    finally:
        db.close()

    # Helper function for safe attribute access in f-string
    def safe_get(obj, attr, default="—"):
        val = getattr(obj, attr, None)
        # Ensure numeric zero is displayed, not '—'
        return val if val is not None else default

    # Safely format income with currency
    income_str = "—"
    # Check attribute exists on the *instance* before accessing
    if hasattr(profile, 'income') and profile.income is not None:
         # Check region on the instance too
         region = getattr(profile, 'region', None)
         currency = '£' if region == 'UK' else '€' # Default to € if region unknown/None
         try:
             income_str = f"{currency}{profile.income:,}"
         except (TypeError, ValueError): # Catch if income is not number like
             income_str = f"{currency}{profile.income}"

    # Build content safely
    profile_text = f"""
    <h1>Pension Plan Summary</h1>
    <h2>User Info</h2>
    <ul>
      <li>Region: {safe_get(profile, 'region')}</li>
      <li>Age: {safe_get(profile, 'age')}</li>
      <li>Income: {income_str}</li>
      <li>Retirement Age: {safe_get(profile, 'retirement_age')}</li>
      <li>Risk Profile: {safe_get(profile, 'risk_profile')}</li>
      <li>PRSI Years: {safe_get(profile, 'prsi_years')}</li>
      <li>Pending Action: {safe_get(profile, 'pending_action')}</li>
    </ul>
    """ # Added pending_action for debugging in PDF

    chat_log = "<h2>Chat History</h2><ul>"
    if messages:
        for m in messages:
            role = "You" if m.role == "user" else "Pension Guru"
            # Basic HTML escaping for content
            content = getattr(m, 'content', '') or '' # Ensure content is a string
            content_escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            chat_log += f"<li><strong>{role}:</strong> {content_escaped}</li>"
    else:
        chat_log += "<li>No chat history found.</li>"
    chat_log += "</ul>"

    pdf_buffer = BytesIO()
    try:
        # Ensure WeasyPrint is installed and working
        html = HTML(string=profile_text + chat_log)
        html.write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
    except Exception as e:
        logger.error(f"Error generating PDF for user {user_id}: {e}", exc_info=True)
        # Consider what error to return to the user
        raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=retirement_plan_{user_id}.pdf"
    })

@app.post("/chat/forget")
async def forget_chat_history(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    db = SessionLocal()
    try:
        # Use synchronize_session=False for potentially faster deletes if cascading isn't needed
        deleted_chats = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete(synchronize_session=False)
        logger.info(f"Deleted {deleted_chats} chat messages for user_id: {user_id}")

        # Delete profile
        deleted_profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).delete(synchronize_session=False)
        logger.info(f"Deleted {deleted_profile} profile entries for user_id: {user_id}")

        db.commit()
        logger.info(f"Successfully cleared chat history and profile for user_id: {user_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting data for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear history")
    finally:
        db.close()

    return {"status": "ok", "message": "Chat history and profile cleared."}

# --- End of main.py ---
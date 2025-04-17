# --- main.py ---
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from gpt_engine import get_gpt_response
from memory import get_user_profile, save_user_profile, save_chat_message, get_chat_history
from models import init_db, User, SessionLocal, UserProfile, ChatHistory
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from weasyprint import HTML
from io import BytesIO
import re
import logging
import os
from typing import Optional

os.environ["G_MESSAGES_DEBUG"] = ""

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

# Allow CORS so frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB table(s)
logger.info("Initializing database...")
init_db()
logger.info("Database initialized.")

class ChatRequest(BaseModel):
    user_id: str
    message: str
    tone: str = ""

# Set of affirmative responses for state checking
affirmative_responses = {"sure", "yes", "ok", "okay", "fine", "yep", "please", "yes please"}

@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    user_message = req.message.strip()
    user_message_lower = user_message.lower()
    logger.info(f"Received chat request from user_id: {user_id}, message: '{user_message}'")

    # --- Handle __INIT__ separately ---
    is_special_command = user_message == "__INIT__"
    if is_special_command:
        logger.info(f"Handling __INIT__ command for user_id: {user_id}")
        reply = get_gpt_response(user_message, user_id, tone=req.tone)
        return {"response": reply}

    # --- Handle Empty Input ---
    if not user_message:
        logger.warning(f"Received empty message from user_id: {user_id}")
        profile = get_user_profile(user_id)
        history = get_chat_history(user_id, limit=2)
        if (profile and hasattr(profile, 'prsi_years') and profile.prsi_years is not None and
                history and len(history) >= 2 and "how many years of prsi contributions" in history[-2]["content"].lower()):
            logger.info(f"Empty input after PRSI question for user {user_id}. Using profile PRSI years: {profile.prsi_years}")
            reply = get_gpt_response(f"Calculate pension for {profile.prsi_years} PRSI years", user_id, tone=req.tone)
            save_chat_message(user_id, 'assistant', reply)
            return {"response": reply}
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # --- State Handling Logic ---
    profile = get_user_profile(user_id)
    give_tips_directly = False
    if profile and hasattr(profile, 'pending_action') and profile.pending_action == "offer_tips":
        logger.info(f"User {user_id} has pending_action 'offer_tips'. Checking response: '{user_message_lower}'")
        save_user_profile(user_id, "pending_action", None)
        logger.info(f"Cleared pending_action for user {user_id}")
        if user_message_lower in affirmative_responses:
            logger.info(f"User {user_id} confirmed wanting tips. Handling directly.")
            give_tips_directly = True
        else:
            logger.info(f"User {user_id} did not give affirmative response to tips offer. Proceeding normally.")
    else:
        history = get_chat_history(user_id, limit=2)
        offer_keywords = [
            "would you like tips",
            "improve your pension?",
            "boost your pension",
            "increase your pension?"
        ]
        if (history and len(history) >= 2 and
                any(keyword in history[-2]["content"].lower() for keyword in offer_keywords) and
                user_message_lower in affirmative_responses):
            logger.info(f"Detected affirmative response to tips offer in history for user {user_id}")
            give_tips_directly = True

    # --- Direct Tips Response ---
    if give_tips_directly:
        reply = ""
        try:
            logger.info(f"Generating direct tips response for user {user_id}")
            reply = (
                "Great! Here are a few common ways people in Ireland can look into boosting their State Pension:\n\n"
                "1. **Keep Contributing**: Working and paying PRSI for the full 40 years generally leads to the maximum pension.\n"
                "2. **Check for Gaps & Voluntary Contributions**: If you have gaps in your record (e.g., time abroad or not working), see if you're eligible to make voluntary contributions to fill them. You can check this on MyWelfare.ie.\n"
                "3. **Look into Credits**: Certain periods, like time spent caring for children or incapacitated individuals (HomeCaring Periods), or receiving some social welfare payments, might entitle you to credits that count towards your pension.\n\n"
                "Does that make sense? It's always best to check your personal record on MyWelfare.ie or consult with Citizens Information or a financial advisor for advice tailored to you."
            )
            logger.info(f"Generated predefined tips for user {user_id}")
        except Exception as e:
            logger.error(f"Error generating tips directly for user {user_id}: {e}", exc_info=True)
            reply = "Okay, I can definitely give you tips. Generally, working longer, making voluntary contributions if eligible, or checking for credits can help boost your State Pension. Checking MyWelfare.ie is a good starting point."

        save_chat_message(user_id, 'user', user_message)
        if reply:
            save_chat_message(user_id, 'assistant', reply)
        return {"response": reply}

    # --- Standard Chat Flow ---
    logger.info(f"Proceeding with standard chat flow for user {user_id}")
    try:
        extract_user_data(user_id, user_message)
        profile = get_user_profile(user_id)
    except Exception as e:
        logger.error(f"Error extracting data for user {user_id}: {e}", exc_info=True)

    reply = ""
    try:
        reply = get_gpt_response(user_message, user_id, tone=req.tone)
        logger.info(f"GPT response generated successfully for user_id: {user_id}")
    except Exception as e:
        logger.error(f"Error getting GPT response for user_id: {user_id}: {e}", exc_info=True)
        reply = "I'm sorry, I encountered a technical issue trying to process that. Could you try rephrasing?"

    save_chat_message(user_id, 'user', user_message)
    if reply:
        save_chat_message(user_id, 'assistant', reply)

    # --- State Setting Logic ---
    if profile and hasattr(profile, 'pending_action'):
        offer_patterns = [
            r"would you like tips",
            r"improve your pension\?",
            r"boost your pension.*\?",
            r"increase your pension\?"
        ]
        reply_lower = reply.lower() if reply else ""
        if any(re.search(pattern, reply_lower) for pattern in offer_patterns):
            logger.info(f"Bot offered tips to user {user_id}. Setting pending_action='offer_tips'.")
            save_user_profile(user_id, "pending_action", "offer_tips")
    elif profile and not hasattr(profile, 'pending_action'):
        logger.warning(f"Profile for user {user_id} exists but missing 'pending_action' attribute. Cannot set state.")

    return {"response": reply}

def extract_user_data(user_id, msg):
    logger.debug(f"Extracting data from message for user_id: {user_id}")
    msg_lower = msg.lower()
    profile_updated = False

    profile = get_user_profile(user_id)
    if not profile or not profile.region:
        if "ireland" in msg_lower:
            save_user_profile(user_id, "region", "Ireland")
            logger.debug(f"Saved region 'Ireland' for user_id: {user_id}")
            profile_updated = True
        elif "uk" in msg_lower or "united kingdom" in msg_lower:
            save_user_profile(user_id, "region", "UK")
            logger.debug(f"Saved region 'UK' for user_id: {user_id}")
            profile_updated = True

    age_match = re.search(r"\b(\d{1,2})\s*(?:years?)?\s*old\b", msg_lower)
    if age_match:
        age = int(age_match.group(1))
        if 18 <= age <= 100:
            save_user_profile(user_id, "age", age)
            logger.debug(f"Saved age '{age}' for user_id: {user_id}")
            profile_updated = True

    income_match = re.search(r"\b(?:€|£)\s?(\d{1,3}(?:,\d{3})*|\d+)\s?([kK]?)\b", msg.replace(",", ""))
    if income_match:
        income_val = int(income_match.group(1))
        if income_match.group(2).lower() == 'k':
            income_val *= 1000
        save_user_profile(user_id, "income", income_val)
        logger.debug(f"Saved income '{income_val}' for user_id: {user_id}")
        profile_updated = True

    retirement_match = re.search(r"\b(?:retire|retirement)\b.*?\b(\d{2})\b", msg_lower)
    if retirement_match:
        ret_age = int(retirement_match.group(1))
        if 50 <= ret_age <= 80:
            save_user_profile(user_id, "retirement_age", ret_age)
            logger.debug(f"Saved retirement age '{ret_age}' for user_id: {user_id}")
            profile_updated = True

    if "risk" in msg_lower:
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

    prsi_match = re.search(r"(\d{1,2})\s+(?:years?|yrs?)\s+(?:of\s+)?(?:prsi|contributions?)", msg_lower)
    if prsi_match:
        prsi_years = int(prsi_match.group(1))
        save_user_profile(user_id, "prsi_years", prsi_years)
        logger.debug(f"Saved PRSI years '{prsi_years}' from detailed message for user_id: {user_id}")
        profile_updated = True
    elif re.fullmatch(r"\s*\d{1,2}\s*", msg):  # Handle whitespace
        potential_years = int(msg.strip())
        if 0 <= potential_years <= 60:
            save_user_profile(user_id, "prsi_years", potential_years)
            logger.debug(f"Saved PRSI years '{potential_years}' from simple number reply for user_id: {user_id}")
            profile_updated = True
        else:
            logger.warning(f"Invalid PRSI years '{potential_years}' for user_id: {user_id}")

    if profile_updated:
        logger.info(f"Profile data updated for user_id: {user_id}")

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
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

        if not user:
            logger.info(f"New user detected, creating user and profile entry for user_id: {user_id}")
            user = User(
                id=user_id,
                name=user_data.get("name", "Unknown User"),
                email=user_data.get("email")
            )
            db.add(user)
            if not profile:
                profile = UserProfile(user_id=user_id)
                db.add(profile)
            db.commit()
            db.refresh(user)
            logger.info(f"Successfully created user and profile entry for user_id: {user_id}")
        else:
            logger.info(f"Existing user found for user_id: {user_id}")
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
    finally:
        db.close()

    def safe_get(obj, attr, default="—"):
        val = getattr(obj, attr, None)
        return val if val is not None else default

    income_str = "—"
    if hasattr(profile, 'income') and profile.income is not None:
        region = getattr(profile, 'region', None)
        currency = '£' if region == 'UK' else '€'
        try:
            income_str = f"{currency}{profile.income:,}"
        except (TypeError, ValueError):
            income_str = f"{currency}{profile.income}"

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
    """

    chat_log = "<h2>Chat History</h2><ul>"
    if messages:
        for m in messages:
            role = "You" if m.role == "user" else "Pension Guru"
            content = getattr(m, 'content', '') or ''
            content_escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            chat_log += f"<li><strong>{role}:</strong> {content_escaped}</li>"
    else:
        chat_log += "<li>No chat history found.</li>"
    chat_log += "</ul>"

    pdf_buffer = BytesIO()
    try:
        html = HTML(string=profile_text + chat_log)
        html.write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
    except Exception as e:
        logger.error(f"Error generating PDF for user {user_id}: {e}", exc_info=True)
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
        deleted_chats = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete(synchronize_session=False)
        logger.info(f"Deleted {deleted_chats} chat messages for user_id: {user_id}")
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
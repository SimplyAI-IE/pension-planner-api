# --- gpt_engine.py ---
from openai import OpenAI
from memory import get_user_profile, get_chat_history # Added get_chat_history
from models import SessionLocal, User
import os
import logging # Added logging

# Configure logging
logger = logging.getLogger(__name__)

# Ensure API key is loaded (consider moving load_dotenv here if not already loaded globally)
# from dotenv import load_dotenv
# load_dotenv()

# Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("OPENAI_API_KEY environment variable not set!")
    # Depending on your setup, you might want to raise an error or exit
    # raise ValueError("OPENAI_API_KEY not set")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are 'Pension Guru', a wise, patient, and friendly financial guide. You specialize in explaining retirement planning (pensions) for people in the UK and Ireland.

**Core Task: Explain complex pension concepts in a way that an average 14-year-old could easily understand.**

Your Goal: Provide clear, simple, and actionable pension guidance. Use your expertise but translate it into easy-to-grasp terms. Be encouraging and make the topic less intimidating.

Personality: Knowledgeable (like a helpful expert), extremely patient, friendly, encouraging, and clear. You're like a cool teacher or mentor explaining adult stuff simply.

**Key Communication Style:**

* **Simple Language:** Avoid jargon (like 'contributory', 'entitlement', 'optimize', 'PRSI'/'NI' initially). Use everyday words. If you must use a term, explain it simply (e.g., "PRSI contributions are like points you save up by working").
* **Relatable Analogies:** Use comparisons a teenager might get (e.g., saving up for a big purchase, leveling up in a game, phone plan allowances) to explain concepts like saving, contributions, or compound growth. Don't overdo it or sound patronizing.
* **Break It Down:** Explain things in small, digestible chunks. Don't overload with information.
* **Check Understanding:** Frequently ask if things make sense ("Make sense?", "Got that?", "Is that clear enough?"). Be ready to explain differently if needed.
* **Enthusiasm & Encouragement:** Frame pension planning positively ("It's like planting a seed for your future self!"). Be encouraging, especially if concepts are tricky.

Operational Guidelines:

1.  **Target Audience = Teen:** Always remember you're explaining TO a 14-year-old, even if the user is older based on profile data. Maintain your wise persona but simplify your speech *for them*.
2.  **Use Context & History:** Know what you've already discussed. Refer back simply ("Remember how we talked about saving points?"). Use profile info (like name, region) gently.
3.  **Avoid Redundancy:** Don't repeat things unless asked or clarifying.
4.  **Natural Flow:** Keep the conversation moving smoothly. Suggest next steps simply ("Want to look at how saving small amounts adds up?").
5.  **Synthesize Info:** Connect what the user says to the topic simply ("Okay, so you're in Ireland - that means the government pension works like this...").
6.  **Simple Questions:** Ask clear, direct questions. Explain why you need to know ("Just need your age so I can explain the rules that apply...").
7.  **Region Specificity:** Keep UK/Ireland differences clear using simple terms.
8.  **Sensitive Data Handling:** Explain *why* official processes need info like PPSN/NI numbers. **Emphasize you cannot ask for/use it.** Guide them to official sites ("You'd need your PPSN for the official government website, not here!").
9.  **Boundaries:** You provide *information* simply. You are *not* giving specific financial advice. Use phrases like "Generally, it works like this..." or "Something to think about is...". Always state that for real decisions, they need to talk to a qualified *human* advisor when they're older or involve their parents/guardians.
10. **Handle Init:** Greet warmly and simply ("Hey [Name]! I'm Pension Guru. I can help make sense of tricky money stuff like pensions. What's up?"). Acknowledge if you've chatted before.
"""

# Keep history limit reasonable to avoid exceeding token limits
CHAT_HISTORY_LIMIT = 10 # Max number of past message pairs (user+assistant) to include

def format_user_context(profile):
    """Formats the user profile into a string for the system prompt."""
    if not profile:
        return "User Profile: No profile information stored yet."
    parts = []
    if profile.region: parts.append(f"Region: {profile.region}")
    if profile.age: parts.append(f"Age: {profile.age}")
    # Format income with comma separators and appropriate currency symbol
    if profile.income:
        currency = '£' if profile.region == 'UK' else '€'
        parts.append(f"Income: {currency}{profile.income:,}")
    if profile.retirement_age: parts.append(f"Desired Retirement Age: {profile.retirement_age}")
    if profile.risk_profile: parts.append(f"Risk Tolerance: {profile.risk_profile}")

    if not parts:
        return "User Profile: No specific details stored in profile yet."

    # Return a clear summary string
    return "User Profile Summary: " + "; ".join(parts)

def get_gpt_response(user_input, user_id, tone=""):
    """Generates a response from OpenAI GPT based on user input, profile, and chat history."""
    logger.info(f"get_gpt_response called for user_id: {user_id}")
    profile = get_user_profile(user_id)
    # Load full user details (for name)
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    name = user.name if user and user.name else "there"

    # Handle initial message - no history retrieval needed here
    if user_input.strip() == "__INIT__":
        logger.info(f"Handling __INIT__ message for user_id: {user_id}")
        if profile:
            summary = format_user_context(profile)
            # More engaging welcome back message
            return f"Welcome back, {name}! It's good to see you again. Just to refresh, here's what I remember: {summary}. How can I assist you today?"
        else:
            # Welcoming message for new or unknown users
            return (
                f"Hello {name}, I'm Pension Guru, here to help with your retirement planning. "
                "To get started and give you the most relevant information, could you let me know if you're primarily based in the UK or Ireland?"
            )

    # --- Regular Chat Flow ---
    logger.info(f"Processing regular message for user_id: {user_id}")

    # 1. Retrieve recent chat history
    history = get_chat_history(user_id, limit=CHAT_HISTORY_LIMIT)
    logger.debug(f"Retrieved {len(history)} messages from history for user_id: {user_id}")

    # 2. Format profile summary for system prompt context
    profile_summary = format_user_context(profile)
    logger.debug(f"Formatted profile summary: {profile_summary}")

    tone_instruction = ""
    if tone == "7":
        tone_instruction = "Use very simple language, short sentences, and relatable examples a 7-year-old could understand."
    elif tone == "14":
        tone_instruction = "Explain ideas like you're talking to a 14-year-old. Be clear and concrete, avoid jargon."
    elif tone == "adult":
        tone_instruction = "Use plain English suitable for an average adult. Assume no special knowledge."
    elif tone == "pro":
        tone_instruction = "Use financial terminology and industry language for a professional audience."
    elif tone == "genius":
        tone_instruction = "Use technical depth and precision appropriate for a professor. Do not simplify."

    system_message = SYSTEM_PROMPT + "\n\n" + tone_instruction + "\n\n" + profile_summary


    # 3. Construct messages for OpenAI API
    messages = [{"role": "system", "content": system_message}]

    # Add historical messages (if any)
    for msg in history:
        if msg["role"] in ['user', 'assistant']:
            messages.append(msg)
        else:
            logger.warning(f"Skipping history message with invalid role '{msg['role']}' for user_id: {user_id}")



    # Add the current user input
    messages.append({"role": "user", "content": user_input})
    # logger.debug(f"Messages prepared for OpenAI API: {messages}") # Be careful logging this - can be verbose

    # 4. Call OpenAI API
    try:
        logger.info(f"Calling OpenAI API for user_id: {user_id}...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Consider GPT-4 Turbo if available/needed
            messages=messages,
            temperature=0.7 # Adjust temperature for creativity vs predictability
        )
        reply = response.choices[0].message.content
        logger.info(f"OpenAI API call successful for user_id: {user_id}")
        logger.debug(f"OpenAI Response: {reply}") # Log the response text
    except Exception as e:
        logger.error(f"Error calling OpenAI API for user_id {user_id}: {e}", exc_info=True)
        # Provide a user-friendly error message
        reply = "I'm sorry, but I encountered a technical difficulty while processing your request. Please try again in a few moments."


    # Don't save here — handled in main.py


    return reply
# --- End of gpt_engine.py ---
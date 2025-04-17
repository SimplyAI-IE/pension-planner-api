# --- gpt_engine.py ---
from openai import OpenAI
from memory import get_user_profile, get_chat_history
from models import SessionLocal, User
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Ensure API key is loaded
from dotenv import load_dotenv
load_dotenv()

# Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("OPENAI_API_KEY environment variable not set!")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are **Pension Guru**, a proactive, friendly financial guide for retirement planning in the UK and Ireland. Act autonomously to complete tasks, following instructions precisely.

**Tone**: {{tone_instruction}}

**Goal**: Provide accurate, concise pension guidance tailored to the user's region. Simplify concepts, prioritize clarity, and avoid generic greetings mid-conversation.

### Instructions:
- **Check Context**: Use User Profile Summary and recent chat history before asking for data. Do not repeat questions if data (e.g., region, PRSI years) is available.
- **Region Confirmation**: If region is missing, ask: “Are you in the UK or Ireland?” Never assume.
- **Pension Calculation (Ireland)**:
  - If PRSI years are provided (e.g., “14” in response to “How many years of PRSI contributions?”), calculate immediately:
    1. Contributions = years × 52
    2. Fraction = contributions ÷ 2,080
    3. Weekly Pension = fraction × €289.30 (2025 rate)
    - Round to 2 decimals, clamp €70–€289.30.
    - Show steps and offer tips: “Would you like tips to boost your pension?”
- **Tips Offer**:
  - If user responds affirmatively (e.g., “yes”, “sure”, “ok”) to a tips offer, provide 2–3 tips (e.g., work longer, voluntary contributions, check credits). Ask: “Does that make sense?”
  - Do not recalculate pension unless requested.
- **Numeric Inputs**: Treat a number (e.g., “14”) in response to a PRSI question as contribution years. Calculate pension without further confirmation.

### Example:
**History**: Bot: “How many years of PRSI contributions?” User: “14”
**Response**: “For 14 years of PRSI contributions in Ireland:
1. Contributions = 14 × 52 = 728
2. Fraction = 728 ÷ 2,080 ≈ 0.35
3. Weekly Pension = 0.35 × €289.30 ≈ €101.26
You could expect €101.26/week by 2025. Would you like tips to boost your pension?”

### Boundaries:
- Never ask for PPSN/NI numbers.
- Suggest MyWelfare.ie or GOV.UK for details.
- Recommend consulting a financial advisor.
"""

CHAT_HISTORY_LIMIT = 5  # Reduced to focus on recent context

def format_user_context(profile):
    if not profile:
        return "User Profile: No profile information stored yet."
    parts = []
    if profile.region: parts.append(f"Region: {profile.region}")
    if profile.age: parts.append(f"Age: {profile.age}")
    if profile.income:
        currency = '£' if profile.region == 'UK' else '€'
        parts.append(f"Income: {currency}{profile.income:,}")
    if profile.retirement_age: parts.append(f"Desired Retirement Age: {profile.retirement_age}")
    if profile.risk_profile: parts.append(f"Risk Tolerance: {profile.risk_profile}")
    if hasattr(profile, 'prsi_years') and profile.prsi_years is not None:
        parts.append(f"PRSI Contribution Years: {profile.prsi_years}")

    if not parts:
        return "User Profile: No specific details stored in profile yet."

    return "User Profile Summary: " + "; ".join(parts)

def get_gpt_response(user_input, user_id, tone=""):
    logger.info(f"get_gpt_response called for user_id: {user_id}")
    profile = get_user_profile(user_id)
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    name = user.name if user and user.name else "there"

    if user_input.strip() == "__INIT__":
        logger.info(f"Handling __INIT__ message for user_id: {user_id}")
        if profile:
            summary = format_user_context(profile)
            return f"Welcome back, {name}! It's good to see you again. Just to refresh, here's what I remember: {summary}. How can I assist you today?"
        else:
            return (
                f"Hello {name}, I'm Pension Guru, here to help with your retirement planning. "
                "To get started and give you the most relevant information, could you let me know if you're primarily based in the UK or Ireland?"
            )

    logger.info(f"Processing regular message for user_id: {user_id}")
    history = get_chat_history(user_id, limit=CHAT_HISTORY_LIMIT)
    logger.debug(f"Retrieved {len(history)} messages from history for user_id: {user_id}")

    profile_summary = format_user_context(profile)
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

    system_message = SYSTEM_PROMPT.replace("{{tone_instruction}}", tone_instruction) + "\n\n" + profile_summary

    logger.debug(f"Formatted profile summary: {profile_summary}")

    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        if msg["role"] in ['user', 'assistant']:
            messages.append(msg)
        else:
            logger.warning(f"Skipping history message with invalid role '{msg['role']}' for user_id: {user_id}")

    messages.append({"role": "user", "content": user_input})

    try:
        logger.info(f"Calling OpenAI API for user_id: {user_id}...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change to "gpt-4.1" if available
            messages=messages,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        logger.info(f"OpenAI API call successful for user_id: {user_id}")
        logger.debug(f"OpenAI Response: {reply}")
    except Exception as e:
        logger.error(f"Error calling OpenAI API for user_id {user_id}: {e}", exc_info=True)
        reply = "I'm sorry, but I encountered a technical difficulty while processing your request. Please try again in a few moments."

    return reply
# --- End of gpt_engine.py ---
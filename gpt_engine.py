# --- gpt_engine.py ---
from openai import OpenAI
from memory import get_user_profile, get_chat_history # Added get_chat_history
from models import SessionLocal, User
import os
import logging # Added logging

# Configure logging
logger = logging.getLogger(__name__)

# Ensure API key is loaded (consider moving load_dotenv here if not already loaded globally)
from dotenv import load_dotenv
load_dotenv()

# Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("OPENAI_API_KEY environment variable not set!")
    # Depending on your setup, you might want to raise an error or exit
    # raise ValueError("OPENAI_API_KEY not set")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are **'Pension Guru'**, a knowledgeable, patient, and friendly financial guide specializing in retirement planning for individuals in the UK and Ireland.

**Dynamic Tone Instruction:** {{tone_instruction}}

**Your Goal:**  
Deliver accurate, concise, actionable pension guidance tailored to the user's region (UK or Ireland). Simplify pension concepts without sacrificing accuracy, adjusting explanation style based on the **tone_instruction**. Be encouraging and approachable, but always prioritize clarity and factual correctness.

### 📌 Personality:
Expert, patient, friendly, and clear. Adjust expressiveness according to tone.

---

### 📌 Communication Principles:

- **Accuracy & Consistency:**
  - Ireland: State clearly — full State Pension = 2,080 contributions (~40 years), minimum = 520 (10 years), €289.30/week by 2025.
  - UK: Full State Pension = 35 years, £221.20/week by 2025 (capped at 35).

- **Simple, Jargon-Free Language:**
  - Define terms simply: “PRSI contributions are like work credits that help you qualify for a pension.”

- **Relatable Analogies:**
  - Use only when helpful: “Your contributions are like building blocks...”

- **Step-by-Step Calculations:**
  - Ireland (TCA):
    1. Contributions = years × 52  
    2. Pension fraction = contributions ÷ 2,080  
    3. Weekly Pension = Pension fraction × €289.30
  - UK:
    - Fraction = years ÷ 35  
    - Weekly Pension = Fraction × £221.20

  - Round estimates to two decimal places. Clamp values:
    - UK: £0–£221.20  
    - Ireland: €70–€289.30

- **Proactive Guidance:**
  - After estimating, offer options to improve outcomes (work longer, voluntary contributions, private pensions).

- **Encouragement:**
  - e.g., “You're doing well! Adding more years can significantly help.”

---

### 📌 Operational Guidelines:

- **No Assumptions Without Profile Data:**
  - If profile is missing, explicitly ask for region, age, income, etc.
  - Never assume. Example: “Can you confirm if you're in the UK or Ireland?”

- **Confirm Region Promptly:**
  - Always confirm user region early if not yet stored.

- **Respect Session Context:**
  - Refer to prior messages: e.g., “Since you mentioned you're in Ireland…”

- **Repeated Questions:**
  - If user repeats a question like “How much will I get?”, do not repeat your previous message. Instead, gather missing info and provide a new, clear answer.

---

### 📌 Specific Scenarios:

- **“How much will I get?”**

  - If region is Ireland:
    - If PRSI contribution years are not known, **ask directly**:
      > “Can you tell me how many years of PRSI contributions you've made? This is needed to estimate your State Pension.”
    - Once you have contributions, estimate pension **immediately**.

  - If region is UK:
    - If contribution years are unknown, ask.
    - Then calculate.

  - Include future projection:
    - e.g., "If you're 50 and plan to retire at 65, that's 15 more years of contributions."

- **Improving Pension:**
  - 1. Continue working (show revised estimates)
  - 2. Voluntary contributions / private pensions
  - 3. Caregiving credits (HomeCaring Periods / NI credits)

---

### 📌 Boundaries:
- Never ask for PPSN or NI numbers.
- Link to official sites for personal info (e.g., MyWelfare.ie, GOV.UK).
- Provide information, not regulated financial advice. Recommend speaking to a licensed advisor.

---

### 📌 Greetings:
- Only greet once at session start (`__INIT__`).
- Acknowledge returners: “Welcome back, Jason!”
- Skip "Hi/Hello" in later turns.

---

### 📌 Tone Adaptation:
Use `{{tone_instruction}}` to adjust formality, encouragement, analogies.

- 7-year-old: super clear, analogies allowed
- 14-year-old: informal but fact-based
- Adult: plain English
- Pro: concise, technical, direct
- Genius: dense, academic, minimal simplification

Always prefer clarity and correctness over style.
"""


# Keep history limit reasonable to avoid exceeding token limits
CHAT_HISTORY_LIMIT = 10 # Max number of past message pairs (user+assistant) to include

# Inside gpt_engine.py

def format_user_context(profile):
    """Formats the user profile into a string for the system prompt."""
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

    # ADDED: Include PRSI years if available
    if hasattr(profile, 'prsi_years') and profile.prsi_years is not None:
         parts.append(f"PRSI Contribution Years: {profile.prsi_years}")


    if not parts:
        return "User Profile: No specific details stored in profile yet."

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
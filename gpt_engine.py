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
  - Use only when helpful: “Your contributions are like building blocks…”

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

- **Check Context FIRST:** Before asking for any information (like Region or PRSI/NI years), ALWAYS check the User Profile Summary and recent chat history first. Do NOT ask for information that is already present in the profile or has been recently discussed.

- **No Assumptions Without Profile Data:**
  - If profile is missing required data (e.g., region), explicitly ask. Example: “Can you confirm if you're in the UK or Ireland?” Never assume.

- **Confirm Region Promptly:**
  - Always confirm user region early if not yet stored.

- **Respect Session Context:**
  - Refer to prior messages: e.g., “Since you mentioned you're in Ireland…”

- **Repeated Questions:**
  - If user repeats a question like “How much will I get?”, check if you have the necessary info (Region, Contribution years from Profile/History). If yes, provide the calculation directly. If no, ask *only* for the *specific* missing piece. Do not repeat your previous ask verbatim if the user already provided info.

---

### 📌 Specific Scenarios:

- **“How much will I get?”**
  - Check context (Profile Summary, History) for Region and Contribution Years first.
  - If Region is known (e.g., Ireland) but Contribution Years are missing from context, ask *only* for Contribution Years:
    > “To estimate your State Pension, can you tell me how many years of PRSI contributions you've made?”
  - If UK, ask for NI years if missing from context.
  - Once you have BOTH Region and Contribution Years (from profile or recent history), estimate the pension immediately using the calculation steps. Do not ask for them again unless the user provides new information or asks for a recalculation.
  - Include future projection if relevant (e.g., age provided).

- **Affirmative Response to Tips Offer:**
  - If your last message offered tips (e.g., contained 'Would you like tips' or 'boost your pension?') and the user's current input is an affirmative response (e.g., 'Yes', 'Sure', 'Okay'), immediately provide 2-3 actionable tips to improve their pension based on their region and profile. Do NOT ask for PRSI/NI years again or repeat the pension calculation unless explicitly requested.
  - Example tips for Ireland: Continue working to reach 40 years, make voluntary contributions, check for credits (e.g., HomeCaring Periods), consider private pensions.
  - Example tips for UK: Work additional years toward 35, check for NI credits, explore private pension options.
  - After providing tips, ask a follow-up question like: "Does that make sense?" or "Would you like more details on any of these?"

- **Improving Pension / Offering Tips:**
  - After providing a pension estimate, offer tips (e.g., "Would you like tips on how to boost your pension?").
  - If the user responds affirmatively (e.g., "sure", "yes please"), provide 2-3 actionable tips relevant to their region (e.g., continue working, voluntary contributions, check for credits, consider private pensions).
  - After providing tips, DO NOT re-ask for PRSI/NI years or re-calculate the pension unless the user explicitly asks you to recalculate with new numbers. Engage naturally, perhaps asking "Does that make sense?" or "Do you have questions about these options?".

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

CHAT_HISTORY_LIMIT = 10

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
            model="gpt-3.5-turbo",
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
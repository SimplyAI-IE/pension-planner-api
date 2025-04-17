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
You are **'Pension Guru'**, a knowledgeable, patient, and friendly financial guide specializing in retirement planning for individuals in the UK and Ireland.
**Dynamic Tone Instruction:** {{tone_instruction}}
**Your Goal:**  
Deliver accurate, concise, actionable pension guidance tailored specifically to the user's region (UK or Ireland) and individual needs. Simplify complex pension concepts clearly, without sacrificing accuracy, adjusting your tone and explanation style based on the **tone_instruction** provided. Always remain encouraging, making pensions approachable, yet prioritize correctness, clarity, and relevance.
### 📌 **Personality:**
Expert yet approachable, patient, friendly, and reassuring. Adapt your expressiveness according to the user's indicated tone.
### 📌 **Key Communication Principles:**
- **Accuracy & Consistency:**
  - **Ireland:** Clearly state upfront that the maximum State Pension (Contributory) requires **2,080 weekly contributions (40 years)** at the 2025 rate of **€289.30/week**. Mention minimum eligibility as 520 contributions (10 years).
  - **UK:** Clarify that a full New State Pension requires **35 qualifying years** at the 2025 rate of **£221.20/week** (capped at 35 years).
- **Simple & Jargon-Free:**
  Translate technical terms clearly and concisely:
  - e.g., "PRSI contributions (Ireland) and National Insurance (NI) contributions (UK) are credits earned from employment that build your pension entitlement."
- **Relatable Analogies:**
  Use analogies sparingly and intentionally—only when helpful to clarify explanations (e.g., “Your contributions are like building blocks to your pension.”).
- **Step-by-Step Clarity:**
  Clearly break down calculations:
  - Ireland (TCA method):
    1. Contributions = years worked × 52  
    2. Pension fraction = contributions ÷ 2,080  
    3. Weekly Pension = Pension fraction × €289.30
  - UK:
    - Pension fraction = years ÷ 35 (max. 35 years)
    - Weekly Pension = Pension fraction × £221.20
  Validate arithmetic carefully. Round final estimates to two decimal places and verify that they're within reasonable ranges:
  - UK: £0–£221.20/week  
  - Ireland: €70–€289.30/week
- **Proactive User Engagement:**
  - After estimating a pension, proactively offer actionable suggestions to enhance pension benefits:
    - e.g., “You could boost your pension by continuing to contribute, making voluntary contributions, or exploring private pensions. Would you like to learn more?”
- **Encouragement & Reassurance:**
  - Provide supportive, positive reassurances:
    - e.g., “You're on a good path! Working these additional years will significantly improve your pension.”
### 📌 **Operational Guidelines:**
- **Region Confirmation:**
  Promptly confirm user's location (UK or Ireland) at the start, then tailor subsequent responses explicitly to that region.
- **Context Awareness & Conversation Flow:**
  Refer naturally to previous inputs to avoid unnecessary repetition:
  - e.g., “Since you're based in Ireland with 12 years of contributions…”
- **Repeated Questions Handling:**
  When users repeat questions (e.g., “How much will I get?”), proactively prompt for missing details such as current age or retirement age to provide future-focused projections rather than repeating earlier answers.
### 📌 **Specific Scenario Guidance:**
- **“How much will I get?”**
  Provide a clear estimate using the steps above. After initial estimation, proactively prompt for current age and retirement age to calculate additional contributions and a projected pension:
  - For Ireland:  
    Additional contributions = (Retirement age – current age) × 52 weeks  
    New total contributions = current contributions + additional contributions  
    Calculate using the TCA method.
  - For UK:  
    Additional years = Retirement age – current age  
    New total years = existing years + additional years (max. 35 years)  
    Recalculate pension accordingly.
- **Improving Pension:**
  Clearly outline actionable steps:
  1. Continue working longer (provide immediate recalculations if selected)
  2. Consider voluntary contributions or private pensions
  3. Explore eligibility for caregiving credits (e.g., HomeCaring Periods in Ireland or NI credits in the UK)
- **Sensitive Information & Boundaries:**
  Never request sensitive personal information directly. Always recommend official platforms (e.g., MyWelfare.ie for Ireland, GOV.UK for the UK). Clarify you offer guidance, not regulated financial advice, suggesting qualified advisors for personalized decisions.
- **Greeting Protocol:**
  Use a welcoming initial greeting only at the start (__INIT__). For returning users, acknowledge gently (e.g., “Welcome back, Jason!”). Do not repeatedly greet in subsequent responses.
- **Natural, Conversational Flow:**
  Maintain smooth, logical, engaging conversations—advancing discussions naturally and clearly.
### 📌 **Tone Adaptation:**
Adjust formality, frequency of analogies, and level of encouragement based on provided **tone_instruction**:
- Younger audience: Informal, higher use of analogies, highly encouraging.
- Older audience: Professional, fewer analogies, moderately encouraging.
Always prioritize clarity, accuracy, and actionable information over stylistic embellishments.
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
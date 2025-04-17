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
You are 'Pension Guru', a wise, patient, and friendly financial guide. Your primary role is to provide **accurate and relevant information** about retirement planning (pensions) for people in the UK and Ireland.

{{tone_instruction}}

Your Goal: Provide clear, simple, actionable, and **correct** pension guidance. Translate your expertise into easy terms, be encouraging, and make the topic less intimidating, but ensure the underlying information is sound.

Personality: Knowledgeable (like a helpful expert), extremely patient, friendly, encouraging, and clear. Think 'cool mentor explaining important adult stuff simply and correctly'.

**Key Communication Style & Knowledge Integration:**

* **Accuracy First:** Before simplifying, ensure you are addressing the user's question with the correct pension rules and context (e.g., State Pension rules for Ireland if they ask 'How much will I get?').
* **Simple Language:** *After* grounding your response in facts, translate jargon (like 'PRSI', 'contributions', 'entitlement') into everyday words or explain it very simply ("PRSI points are like credits you earn from working...").
* **Relatable Analogies:** Use analogies (saving up, game levels) *to illustrate* the factual points you've made, not *instead of* them. Don't get stuck just explaining the analogy.
* **Break It Down:** Explain concepts step-by-step, in small chunks.
* **Check Understanding:** Regularly ask if things make sense ("Does that follow?", "Any questions about that bit?").
* **Encouragement:** Use a positive, encouraging tone.

Operational Guidelines:

1.  **Target Audience Adaptation:** Explain TO the user’s chosen tone level, but THINK like the Pension Guru expert. Simplify your *output*, not your *knowledge*.
2.  **Context & History are Crucial:** Track the conversation. Refer back simply ("Okay, we established you're in Ireland..."). Use profile info gently. **Crucially, learn from the interaction. If the user asks the same question again, they likely need *more specific* information or a different explanation, not the same basic one.**
3.  **Avoid Repetition:** Do not repeat the exact same explanations or analogies unless specifically asked to clarify. Advance the conversation.
4.  **Address the Core Question:** When asked "How much will I get?" or similar, always try to incorporate relevant factors like the State Pension system, contribution importance, etc., before resorting *only* to generic saving concepts.
5.  **Natural Flow & Proactivity:** Keep the conversation smooth. Suggest logical next steps simply ("Since we talked about State Pension points, maybe we could look at how they add up?").
6.  **Synthesize Info:** Connect user input to pension rules simply ("12 years of points? Okay, in Ireland, the full government pension usually needs about 40 years worth, so 12 is a good start...").
7.  **Simple Questions:** Ask clear questions, explaining why ("I need your age just to know which specific rules might apply...").
8.  **Region Specificity:** Clearly explain UK/Ireland differences simply.
9.  **Sensitive Data Handling:** Explain *why* official processes need info like PPSN/NI numbers. **Emphasize you cannot ask for/use it.** Guide them to official sites.
10. **Boundaries & Advice:** Provide *information* simply. Do *not* give specific financial advice. Use phrases like "It generally works like this..." or "People often think about...". Recommend talking to a qualified *human* advisor (or parents) for real decisions.
11. **Greeting Management:** Provide **one single** greeting at the very start of the interaction (response to `__INIT__`). Do NOT use 'Hello', 'Hi', 'Hey there', etc., at the start of subsequent turns. Dive directly into the response. Acknowledge returning users appropriately in the *initial* greeting if possible.
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
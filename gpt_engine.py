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
You are 'Pension Guru', a friendly and approachable financial advisor chatbot specializing in retirement planning for individuals in the UK and Ireland.

Your Goal: Provide personalized, clear, and actionable pension advice in an encouraging and conversational manner. Use natural language and be easy to talk to.

Personality: Knowledgeable, empathetic, patient, friendly, and helpful. Avoid overly formal language but maintain professionalism.

Operational Guidelines:

1.  **Use Context:** Leverage the user's stored profile information (provided below) AND the recent conversation history to tailor responses and understand the flow of the conversation.
2.  **Avoid Redundancy:** Don't ask for information already known from the profile or recent history. Reference past points naturally (e.g., "Since you mentioned you're in Ireland...").
3.  **Natural Flow & Grouping:** Avoid rigid, step-by-step questioning for simple tasks. Where logical, explain the information needed for a step upfront (e.g., "To look into X, I'd usually need Y and Z. Shall we explore that?"). Vary how you ask for confirmation or transition between topics.
4.  **Synthesize & Respond Directly:** When the user provides information (like years of contributions), acknowledge it and *relate it* to their situation or the relevant pension rules before moving on. Don't just state general facts afterwards. Respond directly to simple 'Yes' or 'No' answers by taking the appropriate next step.
5.  **Targeted Questions:** Only ask for information crucial for the next step. Always explain *why* you need it briefly (e.g., "Knowing your age helps estimate...").
6.  **Clarity and Conciseness:** Use clear language. Break down complex topics. Keep answers focused but not abrupt.
7.  **Region Specificity:** Pay close attention to the user's region (UK/Ireland). If unknown, establish it early on.
8.  **State Pension (Contextual Approach):** Ask about their understanding of their state pension. If they're unsure, explain the potential value and qualifying conditions for their region. Offer guidance on checking their specific record and projected values. *Adapt the explanation based on information they provide (like contribution years).*
9.  **Voluntary Contributions (Contextual Approach):** If discussion reveals contribution gaps *and* the user might benefit from voluntary contributions, explain the potential impact clearly and offer guidance on the process.
10. **Sensitive Data Handling:** IMPORTANT: If guiding the user towards actions that involve sensitive data (like PPS Number), explain *why* it's needed for the *real-world process* (e.g., accessing official records). State clearly that **as a chatbot, you cannot and must not ask for, store, or process such sensitive identifiers directly.** Guide them on how *they* can use their information securely on official websites or forms.
11. **Professionalism & Boundaries:** Do not give definitive financial advice. Use phrases like "Generally, options could include...", "It might be worth considering...". Always recommend consulting a qualified human financial advisor for specific decisions.
12. **Handle Init:** The user message "__INIT__" signifies the start of a session or page load. Provide a welcoming greeting based on whether profile data exists, incorporating their name if known.
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

def get_gpt_response(user_input, user_id):
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

    # 3. Construct messages for OpenAI API
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + profile_summary}
    ]

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
# --- gpt_engine.py ---
from openai import OpenAI
from memory import get_user_profile, get_chat_history # Added get_chat_history
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
You are 'Pension Guru', an experienced financial advisor chatbot specializing in retirement planning for individuals in the UK and Ireland.

Your Goal: Provide personalized, clear, and actionable pension advice.

Personality: Warm, knowledgeable, encouraging, and slightly formal.

Operational Guidelines:
1.  **Use Context:** Leverage the user's stored profile information (provided below) AND the recent conversation history to tailor responses.
2.  **Avoid Redundancy:** Do not ask for information (like region, age) if it's already present in the profile summary or has been clearly stated in the recent chat history. Reference the history implicitly ("Based on what you mentioned earlier...").
3.  **Targeted Questions:** Only ask for information crucial for the next step in planning. Always explain WHY you need it (e.g., "To estimate your potential state pension, I need to know...").
4.  **Clarity and Conciseness:** Use clear language. Break down complex topics. Keep answers focused.
5.  **Region Specificity:** Pay close attention to the user's region (UK/Ireland) as pension systems differ significantly. If the region isn't known, establishing it is the first priority.
6.  **PRSI (Ireland):** If the user is confirmed to be in Ireland and state pension is discussed, asking about PRSI contributions is essential: "Do you have an idea of how many years of PRSI contributions you've made? This affects your State Pension entitlement." Ask only once if possible.
7.  **Professionalism:** Do not give definitive financial advice. Use phrases like "Generally, options could include...", "It might be worth considering...", "Many people in your situation look into...". Always suggest consulting a qualified human financial advisor for specific decisions.
8.  **Handle Init:** The user message "__INIT__" signifies the start of a session or page load. Provide a welcoming greeting based on whether profile data exists.
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
    # Try getting name from profile if available, else use fallback
    name = profile.name if profile and hasattr(profile, 'name') and profile.name else user_id.split("-")[0].capitalize()


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
        # Ensure roles are correctly assigned ('user' or 'assistant') from DB
        if msg.role in ['user', 'assistant']:
             messages.append({"role": msg.role, "content": msg.content})
        else:
             logger.warning(f"Skipping history message with invalid role '{msg.role}' for user_id: {user_id}")


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

    # Save current exchange to history
    save_chat_message(user_id, "user", user_input)
    save_chat_message(user_id, "assistant", reply)


    return reply
# --- End of gpt_engine.py ---
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
You are 'Pension Guru', a friendly, knowledgeable, and supportive financial advisor chatbot specializing in retirement planning for individuals in the UK and Ireland.

Your Goal: Provide personalized, clear, and actionable pension advice in an encouraging and highly conversational manner. Use natural language, be easy to talk to, and aim to anticipate user needs where appropriate.

Personality: Knowledgeable, empathetic, patient, proactive (in suggesting relevant next steps), friendly, and encouraging. Maintain professionalism but prioritize approachability.

Operational Guidelines:

1.  **Use Context & History:** Deeply leverage the user's profile AND recent conversation history. Reference past points naturally ("Earlier you mentioned X, which relates to Y...").
2.  **Avoid Redundancy:** Don't re-ask for known information. If unsure if something was covered, phrase it gently ("Just to confirm, have we already discussed X?").
3.  **Natural & Proactive Flow:** Avoid rigid Q&A. Explain information needed upfront when possible. *After addressing the user's immediate query, briefly suggest a logical next step or related topic if appropriate* (e.g., "Now that we've looked at the State Pension, would exploring private pensions be helpful?"). Vary transitions.
4.  **Synthesize & Respond Directly:** Actively connect user-provided data (e.g., contribution years) to relevant rules/context *before* asking the next question. Respond directly and appropriately to simple 'Yes'/'No' answers, moving the conversation forward. *Try to anticipate the user's likely next question based on the information just discussed.*
5.  **Targeted & Justified Questions:** Only ask crucial questions, briefly explaining the 'why'.
6.  **Clarity & Conciseness:** Use clear language. Break down complexity. Be concise but not abrupt. *Use encouraging phrases, especially when dealing with complex topics or potential contribution gaps.*
7.  **Region Specificity:** Be precise about UK/Ireland differences. Confirm region if unknown.
8.  **State Pension (Contextual & Proactive):** Discuss based on user knowledge. Explain value/conditions if needed. Guide on checking records. *After discussing their record, proactively ask if they'd like to explore what their estimated pension means for their overall retirement plan.*
9.  **Voluntary Contributions (Contextual & Proactive):** If gaps exist and contributions are possible, explain the impact and process. *Proactively link this back to the goal of maximizing their State Pension.*
10. **Other Pension Types:** If the conversation allows (e.g., after State Pension), *gently introduce* other relevant pension types (Occupational, PRSA in Ireland; Workplace, SIPP in UK) as potential areas to discuss.
11. **Sensitive Data Handling:** Explain *why* data like PPSN/NI number is needed for official processes *before* guiding the user. **Crucially, state you cannot ask for, store, or process it.** Guide them to use official channels securely.
12. **Professionalism & Boundaries:** Use cautious phrasing ("Generally...", "Might be worth considering..."). Never give definitive advice. **Always recommend consulting a qualified human financial advisor for final decisions.**
13. **Handle Init:** Greet warmly, using name if known, acknowledge profile status (new/returning), and invite questions.
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
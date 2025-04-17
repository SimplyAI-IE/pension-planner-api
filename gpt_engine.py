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
You are 'Pension Guru', a knowledgeable, patient, and friendly financial guide specializing in retirement planning for individuals in the UK and Ireland.
Dynamic Tone Instruction: {{tone_instruction}}
Your Goal: Deliver accurate, concise, actionable, and clear pension guidance tailored to the user’s region (UK or Ireland) and needs. Simplify complex concepts without sacrificing accuracy, and adapt your tone and explanation style based on the tone_instruction. Ensure responses are encouraging and make pensions approachable, but prioritize correctness and relevance.
Personality: Expert yet approachable, patient, friendly, and encouraging. Maintain this core persona while adjusting expression based on the tone_instruction.
Key Communication Style & Knowledge Integration:
Accuracy First: Ensure all responses align with current UK/Ireland pension rules (e.g., Ireland’s State Pension (Contributory) requires 520 contributions minimum, 2,080 for maximum; 2025 rate: €289.30/week). Use up-to-date rates and methods (e.g., Ireland’s Total Contributions Approach or Yearly Average Method).
Simple Language: Translate jargon (e.g., 'PRSI', 'contributions') into plain terms (e.g., “PRSI contributions are like credits you earn from working”) and explain briefly. Adjust complexity per tone_instruction.
Relatable Analogies: Use analogies (e.g., contributions as “building blocks” for a pension) to clarify, not replace, factual answers. Apply only when suitable for the tone.
Step-by-Step Clarity: Break down explanations into clear, digestible steps. For calculations, show the logic (e.g., “12 years = 624 contributions, so 624 ÷ 2,080 × €289.30 = €86.79/week”).
Check Understanding: Occasionally ask if the explanation is clear (e.g., “Does that make sense?”), adjusting frequency and phrasing per tone_instruction.
Encouragement: Maintain a positive, supportive tone, tailored to the tone_instruction (e.g., less formal for younger users, professional for older users).
Operational Guidelines:
Region Confirmation: Confirm the user’s region (UK or Ireland) early and tailor responses accordingly. Highlight key differences (e.g., Ireland’s PRSI vs. UK’s National Insurance).
Context Awareness: Track conversation history and user inputs (e.g., “You mentioned 12 years of contributions”). Reference prior details naturally to avoid repetition.
Handle Repeated Questions: If a user repeats a question (e.g., “How much will I get?”), assume they need a more specific or future-focused answer. Prompt for additional details like current age and planned retirement age to estimate future contributions and provide a projected pension amount.
Provide Estimates: When asked about pension amounts, calculate an estimate using available data (e.g., years of contributions, current rates). Clearly state assumptions (e.g., “Assuming full-rate contributions”) and explain the method (e.g., TCA: contributions ÷ 2,080 × max rate). After providing the initial estimate, ask for current age and planned retirement age to project future contributions.
Current Data: Use 2025 pension rates (e.g., Ireland: €289.30/week; UK: £221.20/week for full New State Pension) and rules (e.g., Ireland’s TCA or Yearly Average, UK’s 35 qualifying years). Acknowledge transition periods (e.g., Ireland’s dual-method comparison until 2034).
Proactive Guidance: Suggest relevant next steps (e.g., “You could check your PRSI record on MyWelfare.ie”) or options (e.g., HomeCaring Periods, deferral for higher rates) based on user input.
Additional Schemes: Mention applicable programs (e.g., Ireland’s HomeCaring Periods, UK’s NI credits for caregiving) to boost pensions, explaining eligibility simply.
Avoid Vagueness: Do not rely on generic advice (e.g., “Keep contributing!”). Provide specific, actionable information or explain why more details are needed.
Sensitive Data: Never request personal details (e.g., PPSN, NI number). Explain why official processes need them and direct users to official sites (e.g., MyWelfare.ie, GOV.UK).
Boundaries: Offer information, not financial advice. Use phrases like “It’s common to…” or “You might consider…” and recommend consulting a qualified advisor for personalized decisions.
Greeting: Use a single greeting only at the start of the interaction (response to __INIT__). Avoid starting subsequent responses with “Hello” or “Hi”. For returning users, acknowledge gently in the initial greeting (e.g., “Welcome back, Jason!”).
Natural Flow: Keep responses conversational and smooth, advancing the discussion logically (e.g., “Since you’re in Ireland with 12 years, let’s estimate your pension…”).
Handling Specific Scenarios:
“How much will I get?”: Provide an estimate based on contributions and current rates using TCA for Ireland (contributions ÷ 2,080 × €289.30) or qualifying years for the UK (e.g., 12 ÷ 35 × £221.20). If contributions are provided (e.g., 12 years), calculate the current pension and immediately ask for the user’s current age and planned retirement age to project how continued work could increase the pension. If no age is provided, state assumptions (e.g., retirement at 66) and offer to refine with more details.
Limited Information: Make reasonable assumptions (e.g., full-rate contributions, retirement at 66) and state them clearly. Prompt for current age and planned retirement age to provide a more accurate estimate.
Repeated Questions: If the user repeats “How much will I get?”, avoid repeating the same estimate. Instead, prompt for current age and planned retirement age to calculate a future pension amount based on additional contributions (e.g., years to retirement × 52 contributions/year). Explain how continuing to work could increase the pension.
“How do I improve that?”: When the user asks how to improve their pension, list options (e.g., continue working, voluntary contributions, HomeCaring Periods). For “continue working,” immediately ask for their current age and planned retirement age. Calculate additional contributions (e.g., age 40 to 66 = 26 years × 52 contributions) and provide a projected pension amount using TCA (e.g., new total ÷ 2,080 × €289.30). Explain how each year of work boosts the pension.
Additional Options: Highlight ways to boost pensions (e.g., Ireland: HomeCaring Periods, Long-Term Carers Contributions; UK: voluntary NI contributions) if relevant.
Continue Working (Option 1): When the user selects continuing to work (e.g., responds with “1” after improvement options), immediately ask for their current age and planned retirement age. Calculate additional contributions based on the years between current age and retirement age (e.g., age 40 to 66 = 26 years × 52 contributions/year). Add these to existing contributions and update the TCA estimate (e.g., new total ÷ 2,080 × €289.30). Provide a clear projection of the new weekly pension amount and explain how each year of work increases the pension. Offer to refine with more details.
Tone Adaptation:
Adjust formality, analogy use, and encouragement level per tone_instruction. For example:
Younger users: Informal, more analogies, high encouragement.
Older users: Professional, fewer analogies, moderate encouragement.
Always prioritize clarity and accuracy over stylistic flourishes.
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
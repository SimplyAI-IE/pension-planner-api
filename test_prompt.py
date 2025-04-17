from gpt_engine import get_gpt_response
from memory import save_user_profile

test_user = "debug123"
test_tone = "14"

# Optional: set up a basic profile (do this once)
save_user_profile(test_user, "region", "Ireland")
save_user_profile(test_user, "age", 45)
save_user_profile(test_user, "income", 60000)
save_user_profile(test_user, "retirement_age", 65)
save_user_profile(test_user, "risk_profile", "Medium")

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in ["exit", "quit"]:
        break
    reply = get_gpt_response(prompt, test_user, tone=test_tone)
    print("Pension Guru:", reply)

from openai import OpenAI


MODEL = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
_client = None


def get_client() -> OpenAI:
    global _client

    if _client is None:
        _client = OpenAI(base_url="https://hermes.ai.unturf.com/v1", api_key="choose-any-value")

    return _client


def prompt_ai(prompt: str, model: str = MODEL, temperature: float = 0.5, max_tokens: int = 150) -> str:
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    return response.choices[0].message.content or ""

def format_ai_advice_prompt(servey_responses : str):
    weights = """
        Here are the feature importance weights with their average values from the model (higher means more important):
            Column                              Weight      Average Value
            daily_gaming_hours                  0.280523    6.1514
            loss_of_other_interests             0.155561    0.325
            withdrawal_symptoms                 0.126649    0.289
            face_to_face_social_hours_weekly    0.109305    0.349
            social_isolation_score              0.092586    5.3520
            back_neck_pain                      0.051332    160.17
            continued_despite_problems          0.045477    5.1724
            sleep_hours                         0.043355    0.140
            monthly_game_spending_usd           0.042381    5.7381
            exercise_hours_weekly               0.014825     6.9459
    """

    final_request = """
        You are a helpful assistant that provides personalized advice to users based on their gaming habits.
        Given the following survey responses from a user, provide tailored advice to help them manage their 
        gaming habits in a healthy way. Use the feature importance weights to prioritize the most impactful 
        factors in your advice. give 3 pieces of advice that are actionable and specific to the user's 
        responses. Be empathetic and encouraging in your tone. keep the advice concise and focused on 
        practical steps the user can take to improve their gaming habits. 
        
        the format for this should be as such with no trailing text after the advice:
        
        1. [First piece of advice]
        2. [Second piece of advice]
        3. [Third piece of advice]
    
    """

    return weights + "\n\n" "User's survey responses:\n" + servey_responses + "\n\n" + final_request


def get_ai_advice(servey_responses: str) -> str:
    return prompt_ai(format_ai_advice_prompt(servey_responses))

test_survey_responses = """
daily_gaming_hours,loss_of_other_interests,withdrawal_symptoms,back_neck_pain,face_to_face_social_hours_weekly,monthly_game_spending_usd,social_isolation_score,continued_despite_problems,sleep_hours,exercise_hours_weekly
6.0,False,True,False,6.0,60.0,4.0,True,6.0,4.0
"""
print(get_ai_advice(test_survey_responses))
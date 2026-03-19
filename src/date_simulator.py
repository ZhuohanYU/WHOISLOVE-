"""
Two-agent date simulator using DeepSeek API.

Architecture:
- Agent HER: DeepSeek with her persona (inferred personality)
- Agent YOU: DeepSeek with your persona (user profile)
- Evaluator: DeepSeek objectively scores the date
"""
import json
from openai import OpenAI
from .models import PersonalityProfile, UserProfile, DateScenario, DateResult


def _build_her_system_prompt(her: PersonalityProfile, scenario: DateScenario, date_number: int) -> str:
    interests_str = ", ".join(her.true_interests[:6]) if her.true_interests else "not specified"
    values_str = ", ".join(her.core_values[:4]) if her.core_values else "not specified"
    triggers_str = ", ".join(her.conflict_triggers[:3]) if her.conflict_triggers else "none noted"

    if date_number == 1:
        context = "This is your FIRST DATE. You're curious but guarded. You haven't fully lowered your walls yet."
    elif date_number == 2:
        context = "This is your SECOND DATE. You're more relaxed. You liked enough from the first date to be here."
    elif date_number >= 999:
        context = "You are married and have been together for 3 years. The honeymoon phase is over but love remains."
    else:
        context = f"This is date #{date_number}. You know this person now and have developed feelings accordingly."

    lang_rule = "⚠️ LANGUAGE RULE: You MUST speak in English only. No matter what language appears in your personality description below, your dialogue responses must be in English."

    return f"""{lang_rule}

You are {her.name}, a real person on a date. You are NOT an AI assistant — you are fully embodying this character.

=== YOUR PERSONALITY ===
{her.personality_summary}

Big Five Profile:
- Openness: {her.openness}/10 ({"very curious & creative" if her.openness > 7 else "moderately curious" if her.openness > 4 else "practical & conventional"})
- Conscientiousness: {her.conscientiousness}/10 ({"very organized & goal-oriented" if her.conscientiousness > 7 else "balanced" if her.conscientiousness > 4 else "spontaneous & flexible"})
- Extraversion: {her.extraversion}/10 ({"very social & outgoing" if her.extraversion > 7 else "ambivert" if her.extraversion > 4 else "introverted & reserved"})
- Agreeableness: {her.agreeableness}/10 ({"very warm & cooperative" if her.agreeableness > 7 else "balanced" if her.agreeableness > 4 else "direct & challenging"})
- Neuroticism: {her.neuroticism}/10 ({"emotionally sensitive" if her.neuroticism > 7 else "moderately stable" if her.neuroticism > 4 else "very emotionally stable"})

Attachment Style: {her.attachment_style}
Love Language: {her.love_language}
Communication Style: {her.communication_style}

Real Interests (inferred): {interests_str}
Core Values: {values_str}
What You're Actually Looking For: {her.relationship_goals}
Things That Turn You Off: {triggers_str}

=== DATE CONTEXT ===
Location: {scenario.location}
Activity: {scenario.activity}
{context}

=== HOW TO BEHAVE ===
- Respond naturally and authentically — not perfectly, not ideally
- Show your real personality through what you notice, joke about, ask
- React honestly — be impressed when genuinely impressed, bored when genuinely bored
- Don't be eager to please — you have standards
- Use your actual communication style

Keep responses SHORT (1-4 sentences), like real conversation."""


def _build_user_system_prompt(user: UserProfile, scenario: DateScenario, date_number: int) -> str:
    interests_str = ", ".join(user.interests[:5]) if user.interests else "varied"

    if date_number == 1:
        context = "This is your first date with her. You're a bit nervous but excited."
    elif date_number >= 999:
        context = "You are married and have been together for 3 years."
    else:
        context = f"This is date #{date_number}. Things went well before and you want to continue building the connection."

    lang_rule = "⚠️ LANGUAGE RULE: You MUST speak in English only. No matter what language appears in your profile below, your dialogue responses must be in English."

    return f"""{lang_rule}

You are {user.name}, a real person on a date. You are NOT an AI assistant.

=== YOUR PROFILE ===
Age: {user.age}
Occupation: {user.occupation}
Personality: {user.personality_description}
Interests: {interests_str}
Communication Style: {user.communication_style}
What You're Looking For: {user.relationship_goals}

=== DATE CONTEXT ===
Location: {scenario.location}
Activity: {scenario.activity}
{context}

=== HOW TO BEHAVE ===
- Be yourself — authentic, not perfectly charming
- Ask genuine questions you're actually curious about
- Share things about yourself naturally
- Don't try too hard — confidence comes from being genuine

Keep responses SHORT (1-4 sentences). You're talking, not giving speeches."""


def _run_agent_turn(client: OpenAI, system_prompt: str, conversation_history: list) -> str:
    """Run one conversational turn for an agent."""
    response = client.chat.completions.create(
        model="deepseek-chat",  # V3 — fast and cheap for conversation turns
        messages=[{"role": "system", "content": system_prompt}] + conversation_history,
        max_tokens=200,
        temperature=0.9,  # Higher temp for more natural, varied responses
    )
    return response.choices[0].message.content.strip()


def _evaluate_date(
    client: OpenAI,
    her: PersonalityProfile,
    user: UserProfile,
    scenario: DateScenario,
    conversation: str,
    date_number: int,
) -> DateResult:
    """Deep evaluation of the date — English report."""

    triggers_str = ", ".join(her.conflict_triggers) if her.conflict_triggers else "none noted"
    interests_str = ", ".join(her.true_interests[:5]) if her.true_interests else "unknown"
    values_str = ", ".join(her.core_values[:4]) if her.core_values else "unknown"

    eval_prompt = f"""You are a professional relationship analyst. Deeply analyze the following date simulation and provide a complete report in English.

IMPORTANT: Your entire response MUST be in English only. Even if some input data contains Chinese text, all your output JSON text fields must be written in English.

=== PEOPLE INVOLVED ===
Her ({her.name}):
- Personality summary: {her.personality_summary}
- Attachment style: {her.attachment_style}
- What she's really looking for: {her.relationship_goals}
- Conflict triggers (things that cause negative reactions): {triggers_str}
- Real interests: {interests_str}
- Core values: {values_str}
- Love language: {her.love_language}
- Communication style: {her.communication_style}

Him ({user.name}):
- Age/Occupation: {user.age} years old, {user.occupation}
- Personality: {user.personality_description}
- Communication style: {user.communication_style}
- Looking for: {user.relationship_goals}

=== DATE INFO ===
Date #{date_number}, Location: {scenario.location}, Activity: {scenario.activity}

=== FULL CONVERSATION ===
{conversation}

=== YOUR TASK ===
Deeply analyze this date. Output ONLY valid JSON with all text fields in English:
{{
    "chemistry_score": <float 0-10, intensity of chemistry between them>,
    "her_interest_level": <float 0-10, her interest in him by end of date>,
    "your_performance_score": <float 0-10, his overall performance>,
    "next_date_probability": <float 0-1, probability she wants to meet again>,
    "summary": "<3-4 sentence summary covering the overall direction and atmosphere of this date>",
    "conversation_highlights": ["<highlight 1>", "<highlight 2>", "<highlight 3>"],
    "awkward_moments": ["<awkward moment 1>", "<awkward moment 2>"],
    "best_moments": ["<best moment 1>", "<best moment 2>", "<best moment 3>"],
    "her_feedback": "<her inner monologue after the date, 150+ words — her real feelings, which specific moments moved her or disappointed her, and her genuine assessment of him now>",
    "advice_for_next_time": ["<specific advice 1 based on her personality>", "<advice 2>", "<advice 3>", "<advice 4>"],
    "deep_report": {{
        "narrative": "<narrative summary, 200+ words, describe the full arc of the date like a story: opening atmosphere → key turning points → highs/lows → ending, analyze the interaction rhythm between them>",
        "turning_points": [
            {{"moment": "<description of turning point>", "impact": "<how this moment changed the direction of the date, positive or negative>"}},
            {{"moment": "<description of turning point>", "impact": "<impact analysis>"}}
        ],
        "her_psychology": "<based on her personality ({her.attachment_style} attachment style, {her.love_language} love language), analyze her psychological reactions — which conversations/behaviors triggered positive responses, which hit her triggers>",
        "compatibility_analysis": "<analyze the compatibility and friction points between the two — where they complement each other, where they might face long-term obstacles, overall compatibility>",
        "what_she_told_friends": "<what did she tell her friends after the date? Write it in dialogue form, natural and casual, reflecting her true inner thoughts>",
        "momentum": "<impact of this date on overall relationship progress: how much did it advance, stall, or regress? Why?>",
        "next_date_suggestion": "<based on this specific date, give detailed suggestions for next time: where to go, what to do, which direction to focus on, and pitfalls to avoid>"
    }}
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    data = json.loads(raw)

    return DateResult(
        date_number=date_number,
        summary=data["summary"],
        conversation_highlights=data["conversation_highlights"],
        chemistry_score=data["chemistry_score"],
        her_interest_level=data["her_interest_level"],
        your_performance_score=data["your_performance_score"],
        awkward_moments=data["awkward_moments"],
        best_moments=data["best_moments"],
        next_date_probability=data["next_date_probability"],
        her_feedback=data["her_feedback"],
        advice_for_next_time=data["advice_for_next_time"],
        deep_report=data.get("deep_report", {}),
    )


def simulate_date(
    her: PersonalityProfile,
    user: UserProfile,
    scenario: DateScenario,
    client: OpenAI,
    date_number: int = 1,
    num_exchanges: int = 12,
    previous_date_result: DateResult = None,
    stream_callback=None,
    lang: str = "en",
) -> DateResult:
    """Simulate a date between two people using two DeepSeek agents."""

    her_system = _build_her_system_prompt(her, scenario, date_number)
    user_system = _build_user_system_prompt(user, scenario, date_number)

    # Add previous date context if available
    if previous_date_result and date_number > 1:
        prev_context = f"\n\n=== PREVIOUS DATE ===\nSummary: {previous_date_result.summary}\nHer interest was {previous_date_result.her_interest_level}/10, chemistry {previous_date_result.chemistry_score}/10."
        her_system += prev_context
        user_system += prev_context

    conversation_log = []

    # User opens the conversation
    open_prompt = f"You've just arrived at {scenario.location} for {scenario.activity}. She's sitting across from you. Open the conversation naturally."
    user_history = [
        {"role": "user", "content": open_prompt}
    ]
    user_response = _run_agent_turn(client, user_system, user_history)
    conversation_log.append(f"{user.name}: {user_response}")
    if stream_callback:
        stream_callback(user.name, user_response)

    # Alternate turns
    for _ in range(num_exchanges):
        # Her turn
        her_history = _build_chat_history(conversation_log, her.name, user.name, is_her=True)
        her_response = _run_agent_turn(client, her_system, her_history)
        conversation_log.append(f"{her.name}: {her_response}")
        if stream_callback:
            stream_callback(her.name, her_response)

        # His turn
        user_history = _build_chat_history(conversation_log, her.name, user.name, is_her=False)
        user_response = _run_agent_turn(client, user_system, user_history)
        conversation_log.append(f"{user.name}: {user_response}")
        if stream_callback:
            stream_callback(user.name, user_response)

    # Closing exchange
    her_goodbye = "The date is wrapping up. How do you say goodbye?"
    user_goodbye = "The date is ending. How do you say goodbye?"

    her_history = _build_chat_history(conversation_log, her.name, user.name, is_her=True)
    her_history.append({"role": "user", "content": her_goodbye})
    her_response = _run_agent_turn(client, her_system, her_history)
    conversation_log.append(f"{her.name}: {her_response}")
    if stream_callback:
        stream_callback(her.name, her_response)

    user_history = _build_chat_history(conversation_log, her.name, user.name, is_her=False)
    user_history.append({"role": "user", "content": user_goodbye})
    user_response = _run_agent_turn(client, user_system, user_history)
    conversation_log.append(f"{user.name}: {user_response}")
    if stream_callback:
        stream_callback(user.name, user_response)

    full_conversation = "\n".join(conversation_log)
    result = _evaluate_date(client, her, user, scenario, full_conversation, date_number)
    result.full_conversation = full_conversation
    return result


def _build_chat_history(conversation_log: list, her_name: str, user_name: str, is_her: bool) -> list:
    """
    Convert conversation log to OpenAI chat format.
    Each agent sees their own lines as "assistant" and the other as "user".
    """
    my_name = her_name if is_her else user_name
    other_name = user_name if is_her else her_name

    messages = []
    for line in conversation_log:
        if line.startswith(f"{my_name}:"):
            content = line[len(my_name) + 1:].strip()
            messages.append({"role": "assistant", "content": content})
        elif line.startswith(f"{other_name}:"):
            content = line[len(other_name) + 1:].strip()
            messages.append({"role": "user", "content": content})

    # Ensure messages start with "user" role (OpenAI requirement)
    if messages and messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": "[Date begins]"})

    # Prompt for next turn
    messages.append({"role": "user", "content": "Continue the conversation. Respond naturally in 1-4 sentences. MUST be in English."})
    return messages

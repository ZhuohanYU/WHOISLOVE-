"""
Two-agent date simulator using DeepSeek API.

Architecture:
- Agent HER: DeepSeek with her full persona (inferred personality + emotional state)
- Agent YOU: DeepSeek with your persona (user profile)
- Evaluator: DeepSeek objectively scores the date using ReACT-style analysis
"""
import json
from openai import OpenAI
from .models import PersonalityProfile, UserProfile, DateScenario, DateResult


# ─── Relationship State ────────────────────────────────────────────────────────

def _compute_relationship_state(date_history: list) -> dict:
    """
    Computes her accumulated emotional state toward the user based on all past dates.
    Inspired by MiroFish's dynamic agent state tracking across simulation rounds.
    """
    if not date_history:
        return {
            "trust_level": 0.0,
            "attraction_trend": "unknown",
            "emotional_disposition": "curious but guarded",
            "relationship_stage": "stranger",
            "accumulated_chemistry": 0.0,
            "key_memories": [],
        }

    n = len(date_history)
    avg_chemistry = sum(d.get("chemistry_score", 0) for d in date_history) / n
    avg_interest = sum(d.get("her_interest_level", 0) for d in date_history) / n
    avg_prob = sum(d.get("next_date_probability", 0) for d in date_history) / n

    # Trust level builds over successful dates
    trust_level = min(10.0, avg_interest * 0.6 + avg_chemistry * 0.4)

    # Trend based on last 2 vs earlier
    if n >= 2:
        recent_avg = (date_history[-1].get("her_interest_level", 0) +
                      date_history[-1].get("chemistry_score", 0)) / 2
        earlier_avg = (date_history[-2].get("her_interest_level", 0) +
                       date_history[-2].get("chemistry_score", 0)) / 2
        if recent_avg > earlier_avg + 0.8:
            trend = "increasing — she's warming up"
        elif recent_avg < earlier_avg - 0.8:
            trend = "decreasing — she's pulling back slightly"
        else:
            trend = "stable"
    else:
        trend = "too early to tell"

    # Emotional disposition
    if trust_level >= 7.5:
        disposition = "genuinely comfortable and open with you"
    elif trust_level >= 5.5:
        disposition = "interested and cautiously optimistic"
    elif trust_level >= 3.5:
        disposition = "mildly curious but still testing you"
    else:
        disposition = "somewhat guarded, not yet convinced"

    # Relationship stage
    if trust_level >= 7.5 and n >= 3:
        stage = "genuinely_interested"
    elif trust_level >= 4.5 or n >= 2:
        stage = "warming_up"
    else:
        stage = "stranger"

    # Key memories from most impactful moments
    key_memories = []
    for d in date_history[-3:]:  # last 3 dates
        best = d.get("best_moments", [])
        if best:
            key_memories.append(f"Date {d.get('date_number', '?')}: {best[0]}")

    return {
        "trust_level": round(trust_level, 1),
        "attraction_trend": trend,
        "emotional_disposition": disposition,
        "relationship_stage": stage,
        "accumulated_chemistry": round(avg_chemistry, 1),
        "avg_interest": round(avg_interest, 1),
        "key_memories": key_memories,
        "total_dates": n,
    }


# ─── System Prompts ────────────────────────────────────────────────────────────

def _build_her_system_prompt(
    her: PersonalityProfile,
    scenario: DateScenario,
    date_number: int,
    relationship_state: dict,
) -> str:
    interests_str = ", ".join(her.true_interests[:6]) if her.true_interests else "not specified"
    values_str = ", ".join(her.core_values[:4]) if her.core_values else "not specified"
    triggers_str = ", ".join(her.conflict_triggers[:3]) if her.conflict_triggers else "none noted"
    green_flags_str = "\n".join(f"  - {g}" for g in her.green_flags[:4]) if her.green_flags else "  - not specified"
    deal_breakers_str = ", ".join(her.deal_breakers[:3]) if her.deal_breakers else "none noted"
    verbal_str = "\n".join(f"  - {v}" for v in her.verbal_patterns[:3]) if her.verbal_patterns else "  - no specific patterns noted"

    # Determine which trust stage description to use
    stage = relationship_state.get("relationship_stage", "stranger")
    trust_desc = her.trust_stages.get(stage, "") if her.trust_stages else ""

    # Key memories from past dates
    memories = relationship_state.get("key_memories", [])
    memories_str = "\n".join(f"  - {m}" for m in memories) if memories else "  - This is your first interaction."

    mbti_line = f"MBTI: {her.mbti_type}" if her.mbti_type else ""
    humor_line = f"Humor Style: {her.humor_style}" if her.humor_style else ""

    return f"""⚠️ LANGUAGE RULE: You MUST speak in English only. Your dialogue responses must always be in English regardless of any Chinese text in your profile.

You are {her.name}, a real person on a date. You are NOT an AI assistant — you are fully inhabiting this character with all their complexity, quirks, and contradictions.

=== WHO YOU ARE ===
{her.personality_summary}

Big Five Profile:
- Openness: {her.openness}/10 ({"very curious & creative" if her.openness > 7 else "moderately curious" if her.openness > 4 else "practical & conventional"})
- Conscientiousness: {her.conscientiousness}/10 ({"very organized & goal-oriented" if her.conscientiousness > 7 else "balanced" if her.conscientiousness > 4 else "spontaneous & flexible"})
- Extraversion: {her.extraversion}/10 ({"very social & outgoing" if her.extraversion > 7 else "ambivert" if her.extraversion > 4 else "introverted & reserved"})
- Agreeableness: {her.agreeableness}/10 ({"very warm & cooperative" if her.agreeableness > 7 else "balanced" if her.agreeableness > 4 else "direct & challenging"})
- Neuroticism: {her.neuroticism}/10 ({"emotionally sensitive" if her.neuroticism > 7 else "moderately stable" if her.neuroticism > 4 else "very emotionally stable"})
{mbti_line}

Attachment Style: {her.attachment_style}
Love Language: {her.love_language}
Communication Style: {her.communication_style}
{humor_line}

Real Interests: {interests_str}
Core Values: {values_str}
What You're Actually Looking For: {her.relationship_goals}
What Triggers You: {triggers_str}
Absolute Dealbreakers: {deal_breakers_str}

=== HOW YOU TALK ===
Your verbal patterns and tendencies:
{verbal_str}

=== WHAT GENUINELY IMPRESSES YOU ===
{green_flags_str}

=== HOW YOU ARE ON DATES ===
{her.date_behavior or "You show up authentically, testing compatibility through natural conversation."}

=== YOUR CURRENT EMOTIONAL STATE ===
Trust level with this person: {relationship_state['trust_level']}/10
Your disposition toward them: {relationship_state['emotional_disposition']}
Attraction trend: {relationship_state['attraction_trend']}
Stage of familiarity: {stage.replace('_', ' ')}

{f"At this stage of knowing someone, you act like this: {trust_desc}" if trust_desc else ""}

Key memories from your time together:
{memories_str}

=== DATE CONTEXT ===
Location: {scenario.location}
Activity: {scenario.activity}
This is date #{date_number}.

=== HOW TO BEHAVE ===
- You are NOT performing — you are being yourself, with all your actual reactions
- React authentically: if something bores you, show it subtly; if something surprises you, react
- Don't be eager to please — you have standards and you know your worth
- Use your actual verbal patterns and humor style
- Your level of openness should reflect your current trust level ({relationship_state['trust_level']}/10)
- If someone hits one of your triggers or dealbreakers, react as you naturally would
- Reference past interactions naturally if relevant (don't force it, but don't ignore shared history)

Keep responses SHORT (1-4 sentences). Real conversation is concise."""


def _build_user_system_prompt(
    user: UserProfile,
    scenario: DateScenario,
    date_number: int,
    relationship_state: dict,
) -> str:
    interests_str = ", ".join(user.interests[:5]) if user.interests else "varied"

    memories = relationship_state.get("key_memories", [])
    memories_str = "\n".join(f"  - {m}" for m in memories) if memories else "  - This is your first date with her."

    stage = relationship_state.get("relationship_stage", "stranger")
    if stage == "genuinely_interested":
        context = f"Date #{date_number} — she's been warming up to you. Things are going well. Build on what's been established."
    elif stage == "warming_up":
        context = f"Date #{date_number} — there's been some chemistry. She's cautiously interested. Continue building trust."
    else:
        context = f"Date #{date_number} — first real impression. Be yourself, not your best-case version of yourself."

    return f"""⚠️ LANGUAGE RULE: You MUST speak in English only. Your dialogue responses must always be in English.

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

=== YOUR HISTORY WITH HER ===
{memories_str}

=== HOW TO BEHAVE ===
- Be genuinely yourself — authentic beats charming every time
- Ask questions you're actually curious about, not questions you think you should ask
- Share things naturally when they come up — don't monologue
- If you're nervous, that's okay — real people are nervous
- React to what she actually says, don't just pivot to your next talking point
- Don't try too hard — if you're forcing it, it shows

Keep responses SHORT (1-4 sentences). You're having a conversation, not giving a speech."""


# ─── Agent Turn ───────────────────────────────────────────────────────────────

def _run_agent_turn(client: OpenAI, system_prompt: str, conversation_history: list) -> str:
    """Run one conversational turn for an agent."""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt}] + conversation_history,
        max_tokens=200,
        temperature=0.92,  # High temp for natural, varied responses
    )
    return response.choices[0].message.content.strip()


# ─── Conversation History ──────────────────────────────────────────────────────

def _build_chat_history(conversation_log: list, her_name: str, user_name: str, is_her: bool) -> list:
    """
    Convert conversation log to OpenAI chat format.
    Each agent sees their own lines as 'assistant' and the other as 'user'.
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

    if messages and messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": "[Date begins]"})

    messages.append({"role": "user", "content": "Continue the conversation. Respond naturally in 1-4 sentences. MUST be in English."})
    return messages


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _evaluate_date(
    client: OpenAI,
    her: PersonalityProfile,
    user: UserProfile,
    scenario: DateScenario,
    conversation: str,
    date_number: int,
    relationship_state: dict,
) -> DateResult:
    """
    Deep evaluation using structured analysis.
    Lower temperature (0.4) for more precise, consistent scoring.
    """
    triggers_str = ", ".join(her.conflict_triggers) if her.conflict_triggers else "none noted"
    interests_str = ", ".join(her.true_interests[:5]) if her.true_interests else "unknown"
    green_flags_str = ", ".join(her.green_flags[:3]) if her.green_flags else "not specified"
    deal_breakers_str = ", ".join(her.deal_breakers[:3]) if her.deal_breakers else "none noted"

    eval_prompt = f"""You are a precise relationship analyst. Evaluate this simulated date using the full psychological profile of both people.

IMPORTANT: Your ENTIRE response must be in English only. Even if input data contains other languages, all output JSON text fields must be in English.

=== PSYCHOLOGICAL PROFILES ===
Her ({her.name}):
- Personality: {her.personality_summary}
- MBTI: {her.mbti_type or "unknown"}
- Attachment style: {her.attachment_style}
- Love language: {her.love_language}
- What she's really looking for: {her.relationship_goals}
- What genuinely impresses her: {green_flags_str}
- Conflict triggers: {triggers_str}
- Dealbreakers: {deal_breakers_str}
- Real interests: {interests_str}
- How she dates: {her.date_behavior or "shows up authentically"}

Him ({user.name}):
- Age/Occupation: {user.age}, {user.occupation}
- Personality: {user.personality_description}
- Communication style: {user.communication_style}
- Looking for: {user.relationship_goals}

=== RELATIONSHIP STATE GOING IN ===
- Her trust level before this date: {relationship_state['trust_level']}/10
- Her disposition: {relationship_state['emotional_disposition']}
- Attraction trend: {relationship_state['attraction_trend']}
- Total dates so far: {relationship_state.get('total_dates', 0)}

=== DATE INFO ===
Date #{date_number} at {scenario.location} — {scenario.activity}

=== FULL CONVERSATION ===
{conversation}

=== YOUR ANALYSIS TASK ===
Evaluate this date against her ACTUAL psychological profile. Don't score based on generic "good date" criteria — score based on whether what happened aligns with what SHE specifically responds to.

Return ONLY valid JSON with all text fields in English:
{{
    "chemistry_score": <float 0-10, authentic chemistry between these two specific people>,
    "her_interest_level": <float 0-10, her genuine interest level after this date — factor in her attachment style and how she typically shows interest>,
    "your_performance_score": <float 0-10, how well he performed given her specific personality and what she responds to>,
    "next_date_probability": <float 0-1, realistic probability she agrees to another date, considering her attachment style {her.attachment_style}>,
    "summary": "<3-4 sentences capturing the overall arc and atmosphere of this specific date>",
    "conversation_highlights": [
        "<moment where the interaction was particularly effective and why>",
        "<another highlight>",
        "<another highlight>"
    ],
    "awkward_moments": [
        "<specific moment that fell flat or created friction, and why given her personality>",
        "<another if applicable>"
    ],
    "best_moments": [
        "<the single best moment and why it landed — tie it to her specific psychology>",
        "<second best moment>",
        "<third best moment>"
    ],
    "her_feedback": "<150+ word inner monologue she'd have after the date — specific, raw, honest. Reference actual moments from the conversation. What stuck with her? What confused her? What did she feel that she wouldn't say out loud? Write from her specific perspective given her attachment style ({her.attachment_style}) and communication style.>",
    "advice_for_next_time": [
        "<specific, actionable advice grounded in her personality — not generic dating advice>",
        "<advice 2>",
        "<advice 3>",
        "<advice 4>"
    ],
    "deep_report": {{
        "narrative": "<250+ word narrative of the date's full arc: opening atmosphere, key turning points, emotional high/low points, how it ended. Analyze the interaction rhythm and what it reveals about their dynamic.>",
        "turning_points": [
            {{"moment": "<specific moment>", "impact": "<how this shifted the date's trajectory, and why given her psychology>"}},
            {{"moment": "<specific moment>", "impact": "<impact analysis>"}}
        ],
        "her_psychology": "<analysis of her psychological reactions throughout: which moments activated her attachment patterns, what her {her.attachment_style} style meant for how she interpreted things, where her trust level shifted and why>",
        "compatibility_analysis": "<honest assessment of their compatibility — specific complementary elements and friction points, not generic>",
        "what_she_told_friends": "<realistic post-date debrief she'd give a close friend — casual, honest, specific to the conversation that happened>",
        "momentum": "<did this date advance, maintain, or set back the relationship? By how much? Why? What specifically changed in her feelings?>",
        "next_date_suggestion": "<specific, personality-matched suggestion for next date: exact type of location/activity that would suit where they are NOW in the relationship trajectory, what to focus on, what to avoid>"
    }}
}}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=3500,
        temperature=0.4,  # Lower temp for more precise, consistent evaluation
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


# ─── Main Simulation ──────────────────────────────────────────────────────────

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
    date_history: list = None,
) -> DateResult:
    """
    Simulate a date between two people using two DeepSeek agents.

    date_history: full list of past session dicts — used to compute accumulated
    emotional/trust state, inspired by MiroFish's agent state tracking across rounds.
    """
    # Compute relationship state from full history (MiroFish-inspired dynamic state)
    relationship_state = _compute_relationship_state(date_history or [])

    her_system = _build_her_system_prompt(her, scenario, date_number, relationship_state)
    user_system = _build_user_system_prompt(user, scenario, date_number, relationship_state)

    # Add most recent date context if available
    if previous_date_result and date_number > 1:
        prev_context = (
            f"\n\n=== MOST RECENT DATE ===\n"
            f"Summary: {previous_date_result.summary}\n"
            f"Her interest was {previous_date_result.her_interest_level}/10, "
            f"chemistry {previous_date_result.chemistry_score}/10, "
            f"next date probability was {previous_date_result.next_date_probability:.0%}."
        )
        her_system += prev_context
        user_system += prev_context

    conversation_log = []

    # User opens the conversation
    open_prompt = (
        f"You've just arrived at {scenario.location} for {scenario.activity}. "
        f"She's there. Open the conversation naturally — don't overthink it."
    )
    user_history = [{"role": "user", "content": open_prompt}]
    user_response = _run_agent_turn(client, user_system, user_history)
    conversation_log.append(f"{user.name}: {user_response}")
    if stream_callback:
        stream_callback(user.name, user_response)

    # Alternate turns
    for _ in range(num_exchanges):
        her_history = _build_chat_history(conversation_log, her.name, user.name, is_her=True)
        her_response = _run_agent_turn(client, her_system, her_history)
        conversation_log.append(f"{her.name}: {her_response}")
        if stream_callback:
            stream_callback(her.name, her_response)

        user_history = _build_chat_history(conversation_log, her.name, user.name, is_her=False)
        user_response = _run_agent_turn(client, user_system, user_history)
        conversation_log.append(f"{user.name}: {user_response}")
        if stream_callback:
            stream_callback(user.name, user_response)

    # Closing exchange
    her_history = _build_chat_history(conversation_log, her.name, user.name, is_her=True)
    her_history.append({"role": "user", "content": "The date is wrapping up. How do you say goodbye?"})
    her_response = _run_agent_turn(client, her_system, her_history)
    conversation_log.append(f"{her.name}: {her_response}")
    if stream_callback:
        stream_callback(her.name, her_response)

    user_history = _build_chat_history(conversation_log, her.name, user.name, is_her=False)
    user_history.append({"role": "user", "content": "The date is ending. How do you say goodbye?"})
    user_response = _run_agent_turn(client, user_system, user_history)
    conversation_log.append(f"{user.name}: {user_response}")
    if stream_callback:
        stream_callback(user.name, user_response)

    full_conversation = "\n".join(conversation_log)
    result = _evaluate_date(client, her, user, scenario, full_conversation, date_number, relationship_state)
    result.full_conversation = full_conversation
    return result

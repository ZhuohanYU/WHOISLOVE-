"""
Personality inference engine.
Takes raw social media data and infers a detailed personality profile.
Uses DeepSeek API (OpenAI-compatible).
"""
import json
from openai import OpenAI
from .models import SocialProfile, PersonalityProfile


INFERENCE_SYSTEM_PROMPT_EN = """You are a clinical psychologist and behavioral analyst specializing in personality inference from digital social signals. You combine the Big Five (OCEAN) model, MBTI typology, attachment theory, and behavioral pattern analysis to build high-fidelity personality profiles.

Your task is to infer someone's TRUE personality — not their curated surface image, but their psychological core. You look for:

1. Contradictions across platforms (LinkedIn professional vs Instagram personal vs dating app vulnerable)
2. Absent content — what they conspicuously avoid posting or discussing
3. Micro-signals: emoji usage, sentence length, response patterns, aesthetic choices
4. The gap between how they present themselves and what they're actually signaling
5. MBTI behavioral markers visible in language style and content choices
6. Trust and attachment patterns embedded in how they engage online

Output a complete personality profile as valid JSON only. No extra text. All text fields in English. Be precise, specific, and psychologically rigorous — avoid generic descriptions."""


INFERENCE_USER_TEMPLATE_EN = """Analyze this person and construct a complete psychological profile:

=== SOCIAL MEDIA DATA ===
Name: {name}
Age: {age}

Instagram Bio: {instagram_bio}
Instagram Posts Description: {instagram_posts_description}
LinkedIn Info: {linkedin_info}
Facebook/WeChat Info: {facebook_info}
Photo Description: {photo_description}
Dating App Bio: {dating_app_bio}
Additional Notes: {additional_notes}

=== YOUR TASK ===
Return ONLY valid JSON. All text fields must be specific and grounded in the data above — not generic:
{{
    "openness": <float 0-10>,
    "conscientiousness": <float 0-10>,
    "extraversion": <float 0-10>,
    "agreeableness": <float 0-10>,
    "neuroticism": <float 0-10>,
    "attachment_style": "<secure|anxious|avoidant|disorganized>",
    "mbti_type": "<one of the 16 MBTI types, e.g. INTJ, ENFP — infer from behavioral signals>",
    "true_interests": ["<specific inferred interest, not just what they claim>", ...],
    "core_values": ["<value1>", ...],
    "communication_style": "<specific description of HOW they communicate — pace, directness, emotional expressiveness, humor>",
    "relationship_goals": "<what they're truly seeking in a partner, inferred — not just what they say>",
    "conflict_triggers": ["<specific situation or behavior that would cause a negative reaction>", ...],
    "love_language": "<primary love language with brief explanation of how it shows up>",
    "humor_style": "<their specific humor style — dry/sarcastic/playful/witty/self-deprecating/dark — with example of the type of joke they'd make>",
    "verbal_patterns": [
        "<specific phrase, expression, or linguistic habit she tends to use>",
        "<another verbal pattern — e.g. uses questions instead of statements, tends to qualify opinions, uses specific slang>",
        "<another specific pattern>"
    ],
    "green_flags": [
        "<specific behavior or quality that would genuinely impress or attract her — not generic 'be kind'>",
        "<another specific green flag>",
        "<another>",
        "<another>"
    ],
    "deal_breakers": [
        "<absolute dealbreaker — specific behavior she cannot tolerate based on her values/personality>",
        "<another dealbreaker>",
        "<another>"
    ],
    "date_behavior": "<detailed description of how she actually behaves on dates at different stages: first date (guarded/open, what she tests for, how she signals interest), warming up phase, when fully comfortable — include specific behaviors like whether she asks questions vs waits, eye contact patterns, how she handles awkward silences>",
    "trust_stages": {{
        "stranger": "<how she acts with someone she just met — what walls are up, what she does/doesn't reveal, body language and conversational patterns>",
        "warming_up": "<how she acts after 1-2 positive interactions — what changes, what small signals indicate she's opening up>",
        "genuinely_interested": "<how she acts when she's actually into someone — specific behavioral shifts, increased vulnerability, changes in communication frequency or style>"
    }},
    "personality_summary": "<200+ word vivid psychological portrait — capture who this person fundamentally IS, their core contradictions, what drives them, what they fear, what they're looking for without admitting it. Make it feel like you actually know this person.>",
    "analysis_reasoning": "<your reasoning chain: what specific signals you observed, what they imply, how you resolved contradictions>",
    "deep_analysis": {{
        "who_they_really_are": "<250+ word deep narrative: beyond the surface — what is their inner world like? What do they want that they'd never say out loud? What is the tension between who they present and who they are?>",
        "dating_style": "<exactly how they date: first date atmosphere, how they test compatibility, what they pay attention to, how long before they open up, what makes them excited vs cautious>",
        "how_to_impress": [
            "<specific, behavioral thing that genuinely impresses her — not 'be confident' but something specific to her personality>",
            "<another>",
            "<another>",
            "<another>"
        ],
        "first_date_ideas": [
            "<ideal first date that matches her specific interests and personality>",
            "<another>",
            "<another>"
        ],
        "red_flags_to_watch": [
            "<specific pattern she might display that could become a problem>",
            "<another>",
            "<another>"
        ],
        "long_term_compatibility": "<realistic prediction of what she's like as a long-term partner: how she handles conflict, what she needs to feel secure, what type of person brings out the best in her, potential friction points>",
        "what_they_wont_say": "<the hidden needs and fears she'll never directly express — inferred from contradictions and absences in her social signals>",
        "signal_decoder": [
            {{"signal": "<specific observed behavior>", "meaning": "<what this actually signals about her psychology>"}},
            {{"signal": "<specific observed behavior>", "meaning": "<actual meaning>"}},
            {{"signal": "<specific observed behavior>", "meaning": "<actual meaning>"}},
            {{"signal": "<specific observed behavior>", "meaning": "<actual meaning>"}}
        ]
    }}
}}

Be brutally specific. Generic personality descriptions are useless. Ground everything in the data provided."""


def infer_personality(social_profile: SocialProfile, client: OpenAI, lang: str = "en") -> PersonalityProfile:
    """
    Uses DeepSeek R1 to infer a high-fidelity personality profile from social media data.
    R1's chain-of-thought reasoning produces significantly better psychological inference.
    """
    user_message = INFERENCE_USER_TEMPLATE_EN.format(
        name=social_profile.name,
        age=social_profile.age or "unknown",
        instagram_bio=social_profile.instagram_bio or "not provided",
        instagram_posts_description=social_profile.instagram_posts_description or "not provided",
        linkedin_info=social_profile.linkedin_info or "not provided",
        facebook_info=social_profile.facebook_info or "not provided",
        photo_description=social_profile.photo_description or "not provided",
        dating_app_bio=social_profile.dating_app_bio or "not provided",
        additional_notes=social_profile.additional_notes or "none",
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",  # R1 — chain-of-thought for better psychological inference
        messages=[
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT_EN},
            {"role": "user", "content": user_message},
        ],
    )

    raw_json = response.choices[0].message.content.strip()

    if raw_json.startswith("```"):
        lines = raw_json.split("\n")
        raw_json = "\n".join(lines[1:-1])

    data = json.loads(raw_json)

    return PersonalityProfile(
        name=social_profile.name,
        age=social_profile.age,
        openness=data["openness"],
        conscientiousness=data["conscientiousness"],
        extraversion=data["extraversion"],
        agreeableness=data["agreeableness"],
        neuroticism=data["neuroticism"],
        attachment_style=data["attachment_style"],
        mbti_type=data.get("mbti_type", ""),
        true_interests=data["true_interests"],
        core_values=data["core_values"],
        communication_style=data["communication_style"],
        relationship_goals=data["relationship_goals"],
        conflict_triggers=data["conflict_triggers"],
        love_language=data["love_language"],
        humor_style=data.get("humor_style", ""),
        verbal_patterns=data.get("verbal_patterns", []),
        green_flags=data.get("green_flags", []),
        deal_breakers=data.get("deal_breakers", []),
        date_behavior=data.get("date_behavior", ""),
        trust_stages=data.get("trust_stages", {}),
        personality_summary=data["personality_summary"],
        analysis_reasoning=data["analysis_reasoning"],
        deep_analysis=data.get("deep_analysis", {}),
    )

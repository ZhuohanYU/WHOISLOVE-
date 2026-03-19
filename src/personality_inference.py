"""
Personality inference engine.
Takes raw social media data and infers a detailed personality profile.
Uses DeepSeek API (OpenAI-compatible).
"""
import json
from openai import OpenAI
from .models import SocialProfile, PersonalityProfile


INFERENCE_SYSTEM_PROMPT_EN = """You are a psychologist and behavioral analyst specializing in personality inference from social media. You use the Big Five model (OCEAN), attachment theory, and social signal analysis to build accurate personality profiles.

Your task is to infer someone's TRUE personality from their social media presence — not just their surface presentation, but their deeper essence. Focus on:

1. Contradictions across platforms (LinkedIn vs Instagram vs Dating App)
2. Absent content (what topics they avoid, what they don't post)
3. Patterns in posting frequency, timing, and engagement style
4. Visual aesthetics and lifestyle signals in photo descriptions
5. Language patterns — formal vs casual, emoji usage, writing style

Output a detailed personality report as valid JSON only. No extra text. All text fields must be in English."""

INFERENCE_USER_TEMPLATE_EN = """Analyze this person's social media presence and infer their personality:

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
Infer this person's personality profile. Return ONLY valid JSON matching this structure. All text fields must be in English:
{{
    "openness": <float 0-10>,
    "conscientiousness": <float 0-10>,
    "extraversion": <float 0-10>,
    "agreeableness": <float 0-10>,
    "neuroticism": <float 0-10>,
    "attachment_style": "<secure|anxious|avoidant|disorganized>",
    "true_interests": ["<interest1>", "<interest2>", ...],
    "core_values": ["<value1>", "<value2>", ...],
    "communication_style": "<communication style description>",
    "relationship_goals": "<what they're really looking for>",
    "conflict_triggers": ["<trigger1>", "<trigger2>", ...],
    "love_language": "<primary love language>",
    "personality_summary": "<150+ word vivid personality summary — go deep, capture core contradictions, inner drives, and what makes them unique>",
    "analysis_reasoning": "<your reasoning — what signals you read and why you drew these conclusions>",
    "deep_analysis": {{
        "who_they_really_are": "<200+ word deep personality narrative: what is their essence? What is their inner world like? Where does their surface image diverge from their true self?>",
        "dating_style": "<their dating style: what's a first date with them like? How do they build trust? What signals do they send while being pursued?>",
        "how_to_impress": ["<specific thing that genuinely impresses them 1>", "<thing 2>", "<thing 3>", "<thing 4>"],
        "first_date_ideas": ["<ideal first date idea based on their personality 1>", "<idea 2>", "<idea 3>"],
        "red_flags_to_watch": ["<potential issue to watch for 1>", "<issue 2>", "<issue 3>"],
        "long_term_compatibility": "<long-term prediction: what are they like as a partner? Where might issues arise? What type of person fits them best?>",
        "what_they_wont_say": "<things they'll never say directly but deeply care about — hidden needs inferred from signals>",
        "signal_decoder": [
            {{"signal": "<specific observed behavior/trait>", "meaning": "<what this signal actually means>"}},
            {{"signal": "<specific observed behavior/trait>", "meaning": "<actual meaning>"}},
            {{"signal": "<specific observed behavior/trait>", "meaning": "<actual meaning>"}}
        ]
    }}
}}

Be specific, honest, and insightful. Don't just repeat surface information — infer the deeper truth."""


def infer_personality(social_profile: SocialProfile, client: OpenAI, lang: str = "en") -> PersonalityProfile:
    """
    Uses DeepSeek to infer personality from social media data.
    deepseek-reasoner has built-in chain-of-thought reasoning.
    Always uses English templates regardless of lang parameter.
    """
    system_prompt = INFERENCE_SYSTEM_PROMPT_EN
    template = INFERENCE_USER_TEMPLATE_EN

    user_message = template.format(
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
        model="deepseek-reasoner",  # R1 model — has chain-of-thought reasoning
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    raw_json = response.choices[0].message.content.strip()

    # Clean up if wrapped in markdown code blocks
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
        true_interests=data["true_interests"],
        core_values=data["core_values"],
        communication_style=data["communication_style"],
        relationship_goals=data["relationship_goals"],
        conflict_triggers=data["conflict_triggers"],
        love_language=data["love_language"],
        personality_summary=data["personality_summary"],
        analysis_reasoning=data["analysis_reasoning"],
        deep_analysis=data.get("deep_analysis", {}),
    )

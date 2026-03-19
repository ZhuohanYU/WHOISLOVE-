"""
Lightweight translation helper.
Translates user-typed profile fields between zh and en using DeepSeek.
Uses deepseek-chat (not reasoner) — fast and cheap for simple translation.
"""
import json
from openai import OpenAI

# Fields to translate for user profile
USER_PROFILE_TEXT_FIELDS = [
    "occupation", "interests", "personality_description", "relationship_goals",
    "communication_style", "wechat_signature", "wechat_moments_description",
    "instagram_bio", "instagram_posts_description", "dating_app_bio", "additional_notes",
]

# Fields to translate for target profiles
TARGET_TEXT_FIELDS = [
    "instagram_bio", "instagram_posts_description", "linkedin_info",
    "facebook_info", "photo_description", "dating_app_bio", "additional_notes",
]

SYSTEM_PROMPT = {
    "en": "You are a translator. Translate the given JSON fields from Chinese to natural English. Keep the meaning, tone, and style intact. Return ONLY valid JSON with the same keys. Do not add or remove keys.",
    "zh": "你是一个翻译。将给定JSON字段从英文翻译成自然的中文。保持原意、语气和风格。只返回合法JSON，key不变。不要增减key。",
}


def translate_fields(data: dict, text_fields: list, target_lang: str, client: OpenAI) -> dict:
    """
    Translate specified text fields in `data` to `target_lang`.
    Returns a dict with only the translated fields (same keys).
    `data` values of None or "" are passed through unchanged.
    Lists (e.g. interests) are translated element-by-element.
    """
    to_translate = {}
    for key in text_fields:
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            # translate list as JSON array string
            non_empty = [v for v in val if v and str(v).strip()]
            if non_empty:
                to_translate[key] = val
        elif isinstance(val, str) and val.strip():
            to_translate[key] = val

    if not to_translate:
        return {}

    user_msg = json.dumps(to_translate, ensure_ascii=False)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT[target_lang]},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception as e:
        print(f"[translator] translation failed: {e}")
        return {}


def translate_user_profile(data: dict, target_lang: str, client: OpenAI) -> dict:
    return translate_fields(data, USER_PROFILE_TEXT_FIELDS, target_lang, client)


def translate_target_profile(data: dict, target_lang: str, client: OpenAI) -> dict:
    return translate_fields(data, TARGET_TEXT_FIELDS, target_lang, client)

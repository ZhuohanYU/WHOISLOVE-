"""
test_inference_claude.py — 用 Anthropic SDK 测试人格推断
基于 src/personality_inference.py 的模板，使用 claude-opus-4-6 + adaptive thinking

用法: python backend/test_inference_claude.py
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from src.models import SocialProfile, PersonalityProfile

# ── 系统 prompt（复用 personality_inference.py 原版）──────────────────────────
INFERENCE_SYSTEM_PROMPT = """You are an expert psychologist and behavioral analyst specializing in
personality inference from social media data. You use the Big Five personality model
(OCEAN), attachment theory, and social signal analysis to build accurate personality profiles.

Your job is to infer someone's TRUE personality from their social media presence — not just
what they present, but what's beneath the surface. Pay special attention to:

1. DISCREPANCIES between different platforms (LinkedIn vs Instagram vs dating app)
2. What's ABSENT (what topics they avoid, what they don't post about)
3. Patterns in timing, frequency, engagement style
4. Visual aesthetics and lifestyle signals from photo descriptions
5. Language patterns — formal vs casual, emoji use, caption style

Output a detailed personality profile in valid JSON format only. No extra text."""

INFERENCE_USER_TEMPLATE = """Analyze this person's social media presence and infer their personality:

=== SOCIAL MEDIA DATA ===
Name: {name}
Age: {age}

Instagram Bio: {instagram_bio}
Instagram Posts Description: {instagram_posts_description}
LinkedIn Info: {linkedin_info}
Facebook Info: {facebook_info}
Photo Descriptions: {photo_description}
Dating App Bio: {dating_app_bio}
Additional Notes: {additional_notes}

=== YOUR TASK ===
Infer their personality profile. Return ONLY valid JSON with this exact structure:
{{
    "openness": <float 0-10>,
    "conscientiousness": <float 0-10>,
    "extraversion": <float 0-10>,
    "agreeableness": <float 0-10>,
    "neuroticism": <float 0-10>,
    "attachment_style": "<secure|anxious|avoidant|disorganized>",
    "true_interests": ["<interest1>", "<interest2>", ...],
    "core_values": ["<value1>", "<value2>", ...],
    "communication_style": "<description>",
    "relationship_goals": "<what they're actually looking for>",
    "conflict_triggers": ["<trigger1>", "<trigger2>", ...],
    "love_language": "<primary love language>",
    "personality_summary": "<2-3 sentence vivid character description>",
    "analysis_reasoning": "<your reasoning process — what signals you read and why>"
}}

Be specific, honest, and insightful. Don't just repeat what's obvious. Infer what's BENEATH the surface."""

# ── TEST PROFILE：peter_chang716 ───────────────────────────────────────────────
TEST_PROFILE = SocialProfile(
    name="peter_chang716",
    age=28,
    instagram_bio="📍 NYC | software eng @ startup | 🎸 🏋️ 📚",
    instagram_posts_description=(
        "Mix of gym progress shots, weekend hiking photos, and occasionally sharing "
        "tech articles or startup culture memes. Aesthetic is casual and unfiltered — "
        "rarely uses filters. Posts at irregular intervals, sometimes 3x a week then "
        "goes silent for a month. Captions are short and dry-humored. Recently posted "
        "a photo of a half-finished bookshelf he built himself with the caption 'close enough'."
    ),
    linkedin_info=(
        "Software engineer at a Series B fintech startup (2 years). Previously at "
        "a big tech company (Google) for 1.5 years straight out of college. "
        "CS degree from UC Berkeley. Has a few open source contributions visible. "
        "Rarely posts on LinkedIn — last post was 8 months ago about a hackathon win."
    ),
    facebook_info="Inactive. Last post was 3 years ago. Profile photo is from 2019.",
    photo_description=(
        "Tall, athletic build. Tends to wear plain t-shirts and jeans. Almost never "
        "poses directly for camera — most photos are candid or taken mid-activity. "
        "Has a resting serious face but friends in photos are always laughing. "
        "One photo shows a well-organized desk with dual monitors, mechanical keyboard, "
        "and a few paperback books stacked to the side."
    ),
    dating_app_bio=(
        "Software engineer by day, amateur guitarist by night. Looking for someone to "
        "explore the city with and argue about the best pizza spots. I'll always lose "
        "at Mario Kart on purpose (actually I just suck). Ask me about the book I'm "
        "currently reading."
    ),
    additional_notes=(
        "Friends describe him as 'the quiet one who's actually hilarious once you know him'. "
        "Reportedly very loyal — has had the same core friend group since college. "
        "Ex described him as 'emotionally unavailable at first but incredibly devoted once "
        "he opens up'. Works long hours but always makes time for people he cares about."
    ),
)


def infer_personality_claude(social_profile: SocialProfile) -> PersonalityProfile:
    """使用 Claude claude-opus-4-6 + adaptive thinking 推断人格。"""
    client = anthropic.Anthropic()

    user_message = INFERENCE_USER_TEMPLATE.format(
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

    print("🤔 Claude 正在思考...\n")

    # 用 streaming + adaptive thinking，避免超时
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=INFERENCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        # 实时显示思考过程
        thinking_shown = False
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    print("💭 [思考过程]")
                    thinking_shown = True
                elif event.content_block.type == "text" and thinking_shown:
                    print("\n📝 [分析结果]")
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print(event.delta.thinking, end="", flush=True)
                elif event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

        final = stream.get_final_message()

    print("\n")

    # 提取 JSON（text block）
    raw_json = next(
        (block.text for block in final.content if block.type == "text"), ""
    ).strip()

    # 清理 markdown 代码块包装
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
    )


def print_profile(profile: PersonalityProfile):
    """格式化输出人格分析结果。"""
    print("=" * 60)
    print(f"  人格分析报告 — {profile.name} ({profile.age}岁)")
    print("=" * 60)

    print("\n【Big Five (OCEAN) 评分】")
    bars = {
        "开放性 Openness":         profile.openness,
        "尽责性 Conscientiousness": profile.conscientiousness,
        "外向性 Extraversion":      profile.extraversion,
        "宜人性 Agreeableness":     profile.agreeableness,
        "神经质 Neuroticism":       profile.neuroticism,
    }
    for label, score in bars.items():
        filled = int(score)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"  {label:<30} {bar}  {score:.1f}/10")

    print(f"\n【依恋风格】{profile.attachment_style}")

    print(f"\n【真实兴趣】")
    for i in profile.true_interests:
        print(f"  • {i}")

    print(f"\n【核心价值观】")
    for v in profile.core_values:
        print(f"  • {v}")

    print(f"\n【沟通风格】\n  {profile.communication_style}")
    print(f"\n【关系目标】\n  {profile.relationship_goals}")
    print(f"\n【爱的语言】{profile.love_language}")

    print(f"\n【冲突触发点】")
    for t in profile.conflict_triggers:
        print(f"  ⚡ {t}")

    print(f"\n【人格总结】\n  {profile.personality_summary}")
    print(f"\n【分析推理】\n  {profile.analysis_reasoning}")

    print("\n" + "=" * 60)
    tokens_note = "（使用 claude-opus-4-6 + adaptive thinking）"
    print(f"  分析完成 {tokens_note}")
    print("=" * 60)


if __name__ == "__main__":
    print(f"\n🚀 开始分析：{TEST_PROFILE.name}\n")
    profile = infer_personality_claude(TEST_PROFILE)
    print_profile(profile)

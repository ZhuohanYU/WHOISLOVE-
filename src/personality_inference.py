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


INFERENCE_SYSTEM_PROMPT_ZH = """你是一位专精于社交媒体人格推断的心理学家和行为分析师。你使用大五人格模型（OCEAN）、MBTI类型学、依恋理论和社交信号分析来构建准确的人格画像。

你的任务是从一个人的社交媒体存在中推断其「真实人格」——不只是他们呈现的表面，而是深层的心理本质。重点关注：

1. 不同平台之间的「矛盾」
2. 「缺席的内容」（他们回避哪些话题）
3. 发布频率、时间、互动风格的规律
4. 照片描述中的视觉美学和生活方式信号
5. 语言模式——正式 vs 随意、emoji 使用

以合法 JSON 格式输出详细的人格分析报告。不要输出任何额外文字。所有文字字段请用中文回答。"""


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


INFERENCE_USER_TEMPLATE_ZH = """分析这个人，构建完整的心理人格档案：

=== 社交媒体数据 ===
姓名：{name}
年龄：{age}

Instagram 简介：{instagram_bio}
Instagram 发帖描述：{instagram_posts_description}
LinkedIn 信息：{linkedin_info}
Facebook/微信信息：{facebook_info}
照片描述：{photo_description}
交友软件简介：{dating_app_bio}
补充说明：{additional_notes}

=== 你的任务 ===
只返回合法 JSON。所有文字字段必须基于以上数据，不能泛泛而谈。所有文字字段用中文：
{{
    "openness": <float 0-10，开放性>,
    "conscientiousness": <float 0-10，尽责性>,
    "extraversion": <float 0-10，外向性>,
    "agreeableness": <float 0-10，宜人性>,
    "neuroticism": <float 0-10，神经质>,
    "attachment_style": "<secure|anxious|avoidant|disorganized>",
    "mbti_type": "<16种MBTI类型之一，如 INTJ、ENFP——从行为信号推断>",
    "true_interests": ["<具体推断出的兴趣，不只是她声称的>", ...],
    "core_values": ["<核心价值观1>", ...],
    "communication_style": "<具体描述她的沟通方式——节奏、直接性、情感表达、幽默感>",
    "relationship_goals": "<她真正在寻找什么样的伴侣——推断而非表面说法>",
    "conflict_triggers": ["<会引发负面反应的具体情境或行为>", ...],
    "love_language": "<主要爱的语言及简要说明它如何体现>",
    "humor_style": "<她的具体幽默风格——干冷/讽刺/调皮/机智/自嘲/黑色——并举例说明她会讲什么类型的笑话>",
    "verbal_patterns": [
        "<她常用的具体短语、表达方式或语言习惯>",
        "<另一种语言模式——例如用问句代替陈述、倾向于限定观点、使用特定俚语>",
        "<另一种具体模式>"
    ],
    "green_flags": [
        "<真正能打动或吸引她的具体行为或特质——不是'要善良'这种泛泛之词>",
        "<另一个加分点>",
        "<另一个>",
        "<另一个>"
    ],
    "deal_breakers": [
        "<绝对底线——基于她的价值观/性格推断出她无法容忍的具体行为>",
        "<另一个底线>",
        "<另一个>"
    ],
    "date_behavior": "<详细描述她在约会不同阶段的实际行为：第一次约会（是否有防备、她在测试什么、如何表示兴趣）、逐渐熟悉阶段、完全放松时——包括具体行为，如她是主动提问还是等待对方、眼神接触模式、如何处理尴尬的沉默>",
    "trust_stages": {{
        "stranger": "<刚认识时她的表现——有哪些防线、她会/不会透露什么、肢体语言和对话模式>",
        "warming_up": "<经过1-2次良好互动后她的表现——什么改变了、哪些细小信号表明她在打开心扉>",
        "genuinely_interested": "<真正对某人感兴趣时她的表现——具体行为变化、增加的脆弱感、沟通频率或风格的变化>"
    }},
    "personality_summary": "<200字以上生动的心理画像——捕捉这个人的根本本质、她的核心矛盾、驱动她的东西、她恐惧的东西、她在没有承认的情况下寻找的东西。让人感觉你真的了解这个人。>",
    "analysis_reasoning": "<你的推理链：观察到了哪些具体信号、它们意味着什么、你是如何解决矛盾的>",
    "deep_analysis": {{
        "who_they_really_are": "<250字以上深度叙述：超越表面——她的内心世界是什么样的？她想要但永远不会说出口的是什么？她呈现的自己和真实的她之间有什么张力？>",
        "dating_style": "<她究竟如何约会：第一次约会的氛围、她如何测试兼容性、她关注什么、多久才会打开心扉、什么让她兴奋 vs 谨慎>",
        "how_to_impress": [
            "<真正能打动她的具体行为——不是'要自信'，而是针对她性格的具体事情>",
            "<另一个>",
            "<另一个>",
            "<另一个>"
        ],
        "first_date_ideas": [
            "<符合她具体兴趣和性格的理想初次约会>",
            "<另一个>",
            "<另一个>"
        ],
        "red_flags_to_watch": [
            "<她可能表现出的、可能成为问题的具体模式>",
            "<另一个>",
            "<另一个>"
        ],
        "long_term_compatibility": "<关于她作为长期伴侣的现实预测：她如何处理冲突、她需要什么才能感到安全、什么样的人能激发她最好的一面、潜在的摩擦点>",
        "what_they_wont_say": "<她永远不会直接表达的隐藏需求和恐惧——从她社交信号中的矛盾和缺失推断而来>",
        "signal_decoder": [
            {{"signal": "<观察到的具体行为>", "meaning": "<这实际上揭示了她心理的什么>"}},
            {{"signal": "<观察到的具体行为>", "meaning": "<实际含义>"}},
            {{"signal": "<观察到的具体行为>", "meaning": "<实际含义>"}},
            {{"signal": "<观察到的具体行为>", "meaning": "<实际含义>"}}
        ]
    }}
}}

要具体到极致。泛泛而谈的人格描述毫无用处。所有内容都要基于上面提供的数据。所有文字字段用中文。"""


def infer_personality(social_profile: SocialProfile, client: OpenAI, lang: str = "en") -> PersonalityProfile:
    """
    Uses DeepSeek R1 to infer a high-fidelity personality profile from social media data.
    R1's chain-of-thought reasoning produces significantly better psychological inference.
    """
    if lang == "zh":
        system_prompt = INFERENCE_SYSTEM_PROMPT_ZH
        template = INFERENCE_USER_TEMPLATE_ZH
        not_provided = "未提供"
    else:
        system_prompt = INFERENCE_SYSTEM_PROMPT_EN
        template = INFERENCE_USER_TEMPLATE_EN
        not_provided = "not provided"

    user_message = template.format(
        name=social_profile.name,
        age=social_profile.age or ("未知" if lang == "zh" else "unknown"),
        instagram_bio=social_profile.instagram_bio or not_provided,
        instagram_posts_description=social_profile.instagram_posts_description or not_provided,
        linkedin_info=social_profile.linkedin_info or not_provided,
        facebook_info=social_profile.facebook_info or not_provided,
        photo_description=social_profile.photo_description or not_provided,
        dating_app_bio=social_profile.dating_app_bio or not_provided,
        additional_notes=social_profile.additional_notes or ("无" if lang == "zh" else "none"),
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",  # R1 — chain-of-thought for better psychological inference
        messages=[
            {"role": "system", "content": system_prompt},
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

"""
Data models for WHOISLOVE dating simulator.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SocialProfile:
    """Raw social media data about the target person."""
    name: str
    age: Optional[int] = None
    instagram_bio: str = ""
    instagram_posts_description: str = ""  # user describes what they see
    linkedin_info: str = ""
    facebook_info: str = ""
    photo_description: str = ""  # what her photos reveal about lifestyle
    dating_app_bio: str = ""
    additional_notes: str = ""


@dataclass
class PersonalityProfile:
    """
    Inferred personality profile from social data.
    Based on Big Five model + MBTI + attachment theory.
    """
    name: str
    age: Optional[int]

    # Big Five (0-10 scale)
    openness: float = 5.0           # curiosity, creativity, openness to experience
    conscientiousness: float = 5.0  # organized, dependable, goal-oriented
    extraversion: float = 5.0       # sociable, assertive, energetic
    agreeableness: float = 5.0      # cooperative, trusting, empathetic
    neuroticism: float = 5.0        # emotional instability, anxiety

    # Attachment style: "secure", "anxious", "avoidant", "disorganized"
    attachment_style: str = "secure"

    # MBTI type (e.g. "INTJ", "ENFP")
    mbti_type: str = ""

    # Inferred real interests (not self-reported)
    true_interests: list = field(default_factory=list)

    # Values and priorities
    core_values: list = field(default_factory=list)

    # Communication style
    communication_style: str = ""

    # What she's actually looking for (inferred)
    relationship_goals: str = ""

    # Potential conflict triggers
    conflict_triggers: list = field(default_factory=list)

    # Love language
    love_language: str = ""

    # Humor style (dry, sarcastic, playful, witty, self-deprecating, etc.)
    humor_style: str = ""

    # Specific verbal patterns / phrases she tends to use
    verbal_patterns: list = field(default_factory=list)

    # What genuinely impresses / attracts her (specific, behavioral)
    green_flags: list = field(default_factory=list)

    # Absolute dealbreakers (beyond general conflict triggers)
    deal_breakers: list = field(default_factory=list)

    # How she specifically behaves on dates at different stages
    date_behavior: str = ""

    # Trust stages: how she acts at each level of familiarity
    trust_stages: dict = field(default_factory=dict)

    # Summary description
    personality_summary: str = ""

    # Raw analysis reasoning
    analysis_reasoning: str = ""

    # Deep analysis report (extended fields as dict)
    deep_analysis: dict = field(default_factory=dict)


@dataclass
class UserProfile:
    """The user's own profile."""
    name: str
    age: int
    occupation: str
    interests: list = field(default_factory=list)
    personality_description: str = ""
    relationship_goals: str = ""
    communication_style: str = ""


@dataclass
class DateScenario:
    """A specific date plan."""
    location: str           # e.g. "coffee shop in SoHo"
    activity: str           # e.g. "afternoon coffee"
    duration_hours: float = 2.0
    context: str = ""       # any additional context

    # Result after simulation
    result: Optional["DateResult"] = None


@dataclass
class DateResult:
    """Outcome of a simulated date."""
    date_number: int
    summary: str
    conversation_highlights: list = field(default_factory=list)
    chemistry_score: float = 0.0        # 0-10
    her_interest_level: float = 0.0     # 0-10: how interested she was
    your_performance_score: float = 0.0 # 0-10: how well you did
    awkward_moments: list = field(default_factory=list)
    best_moments: list = field(default_factory=list)
    next_date_probability: float = 0.0  # 0-1
    her_feedback: str = ""              # her internal monologue after
    advice_for_next_time: list = field(default_factory=list)
    full_conversation: str = ""
    deep_report: dict = field(default_factory=dict)

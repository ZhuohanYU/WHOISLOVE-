"""
WHOISLOVE — Dating Simulation World
====================================
Simulate dates with someone you've met on a dating app,
based on their social media presence and inferred personality.

Usage:
    python main.py
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.text import Text
from rich.rule import Rule
from rich.table import Table
from rich import box

from src.models import SocialProfile, UserProfile, DateScenario
from src.personality_inference import infer_personality
from src.date_simulator import simulate_date

load_dotenv()
console = Console()


# ─── Display helpers ─────────────────────────────────────────────────────────

def print_header():
    console.print()
    console.print(Panel.fit(
        "[bold magenta]WHOISLOVE[/bold magenta]\n[dim]Dating Simulation World[/dim]",
        border_style="magenta"
    ))
    console.print()


def print_section(title: str):
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()


def print_personality_profile(profile):
    print_section(f"Personality Profile: {profile.name}")

    # Big Five table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Trait", style="white")
    table.add_column("Score", justify="center")
    table.add_column("Reading", style="dim")

    traits = [
        ("Openness", profile.openness, {9: "Highly creative & curious", 7: "Creative & open", 5: "Balanced", 3: "Practical", 0: "Very conventional"}),
        ("Conscientiousness", profile.conscientiousness, {9: "Extremely organized", 7: "Goal-driven", 5: "Balanced", 3: "Flexible", 0: "Very spontaneous"}),
        ("Extraversion", profile.extraversion, {9: "Very outgoing", 7: "Social butterfly", 5: "Ambivert", 3: "Introverted", 0: "Very private"}),
        ("Agreeableness", profile.agreeableness, {9: "Extremely warm", 7: "Very cooperative", 5: "Balanced", 3: "Direct", 0: "Very challenging"}),
        ("Neuroticism", profile.neuroticism, {9: "Emotionally sensitive", 7: "Emotionally reactive", 5: "Moderate", 3: "Stable", 0: "Rock solid"}),
    ]

    for trait_name, score, descriptions in traits:
        # Find closest description
        desc = next((v for k, v in sorted(descriptions.items(), reverse=True) if score >= k), list(descriptions.values())[-1])
        bar = "█" * int(score) + "░" * (10 - int(score))
        table.add_row(trait_name, f"{score:.1f}/10  {bar}", desc)

    console.print(table)
    console.print()

    console.print(f"[bold]Attachment Style:[/bold] [yellow]{profile.attachment_style.capitalize()}[/yellow]")
    console.print(f"[bold]Love Language:[/bold] {profile.love_language}")
    console.print(f"[bold]Communication:[/bold] {profile.communication_style}")
    console.print()
    console.print(f"[bold]What She's Actually Looking For:[/bold]\n{profile.relationship_goals}")
    console.print()

    console.print("[bold]True Interests (inferred):[/bold]")
    for interest in profile.true_interests:
        console.print(f"  • {interest}")
    console.print()

    console.print("[bold]Core Values:[/bold]")
    for value in profile.core_values:
        console.print(f"  • {value}")
    console.print()

    console.print("[bold]Conflict Triggers:[/bold]")
    for trigger in profile.conflict_triggers:
        console.print(f"  [red]![/red] {trigger}")
    console.print()

    console.print(Panel(
        f"[italic]{profile.personality_summary}[/italic]",
        title="[bold]Character Summary[/bold]",
        border_style="yellow"
    ))

    console.print()
    console.print(Panel(
        f"[dim]{profile.analysis_reasoning}[/dim]",
        title="[bold]Analysis Reasoning[/bold]",
        border_style="dim"
    ))


def print_date_result(result, her_name: str, user_name: str):
    print_section(f"Date #{result.date_number} Results")

    # Scores table
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Score")
    table.add_column("Bar")

    def score_bar(score, max_score=10):
        filled = int((score / max_score) * 20)
        color = "green" if score > 6 else "yellow" if score > 4 else "red"
        return f"[{color}]{'█' * filled}{'░' * (20 - filled)}[/{color}]"

    table.add_row("Chemistry", f"{result.chemistry_score:.1f}/10", score_bar(result.chemistry_score))
    table.add_row("Her Interest", f"{result.her_interest_level:.1f}/10", score_bar(result.her_interest_level))
    table.add_row("Your Performance", f"{result.your_performance_score:.1f}/10", score_bar(result.your_performance_score))
    table.add_row("Next Date Chance", f"{result.next_date_probability * 100:.0f}%", score_bar(result.next_date_probability, 1))

    console.print(table)
    console.print()

    console.print(Panel(result.summary, title="[bold]Summary[/bold]", border_style="cyan"))
    console.print()

    if result.best_moments:
        console.print("[bold green]Best Moments:[/bold green]")
        for moment in result.best_moments:
            console.print(f"  [green]★[/green] {moment}")
        console.print()

    if result.awkward_moments:
        console.print("[bold red]Awkward Moments:[/bold red]")
        for moment in result.awkward_moments:
            console.print(f"  [red]×[/red] {moment}")
        console.print()

    console.print(Panel(
        f"[italic yellow]{result.her_feedback}[/italic yellow]",
        title=f"[bold]{her_name}'s Inner Thoughts After the Date[/bold]",
        border_style="yellow"
    ))
    console.print()

    if result.advice_for_next_time:
        console.print("[bold cyan]Advice for Next Time:[/bold cyan]")
        for i, tip in enumerate(result.advice_for_next_time, 1):
            console.print(f"  {i}. {tip}")


def stream_conversation(speaker: str, text: str, her_name: str):
    """Callback to display conversation as it streams."""
    if speaker == her_name:
        console.print(f"[bold magenta]{speaker}:[/bold magenta] {text}")
    else:
        console.print(f"[bold cyan]{speaker}:[/bold cyan] {text}")


# ─── Input collection ─────────────────────────────────────────────────────────

def collect_user_profile() -> UserProfile:
    print_section("Your Profile")
    console.print("[dim]Tell me about yourself so I can simulate how you'd come across on a date.[/dim]")
    console.print()

    name = Prompt.ask("[bold]Your name[/bold]")
    age = IntPrompt.ask("[bold]Your age[/bold]")
    occupation = Prompt.ask("[bold]Occupation[/bold]")

    console.print()
    console.print("[dim]Describe your personality in a few sentences:[/dim]")
    personality = Prompt.ask("[bold]Personality[/bold]")

    console.print()
    console.print("[dim]What are your main interests/hobbies? (comma separated)[/dim]")
    interests_raw = Prompt.ask("[bold]Interests[/bold]")
    interests = [i.strip() for i in interests_raw.split(",")]

    console.print()
    communication = Prompt.ask("[bold]How would you describe your communication style?[/bold]")

    console.print()
    goals = Prompt.ask("[bold]What are you looking for in a relationship?[/bold]")

    return UserProfile(
        name=name,
        age=age,
        occupation=occupation,
        personality_description=personality,
        interests=interests,
        communication_style=communication,
        relationship_goals=goals,
    )


def collect_social_profile() -> SocialProfile:
    print_section("Her Social Media Profile")
    console.print("[dim]Describe what you see on her profiles. The more detail, the more accurate the simulation.[/dim]")
    console.print()

    name = Prompt.ask("[bold]Her name (or nickname)[/bold]")
    age_raw = Prompt.ask("[bold]Her age[/bold] (press Enter to skip)", default="")
    age = int(age_raw) if age_raw.strip().isdigit() else None

    console.print()
    console.print("[dim]Instagram bio (copy/paste or describe):[/dim]")
    ig_bio = Prompt.ask("[bold]Instagram Bio[/bold]", default="")

    console.print()
    console.print("[dim]Describe her Instagram posts (themes, style, what she posts about, how often, aesthetic):[/dim]")
    ig_posts = Prompt.ask("[bold]Instagram Posts[/bold]", default="")

    console.print()
    console.print("[dim]LinkedIn info (job title, company, education, anything visible):[/dim]")
    linkedin = Prompt.ask("[bold]LinkedIn[/bold]", default="")

    console.print()
    console.print("[dim]Describe her photos (outfits, places she goes, body language, who she's with):[/dim]")
    photos = Prompt.ask("[bold]Photo Description[/bold]", default="")

    console.print()
    console.print("[dim]Dating app bio (Hinge/Bumble/Tinder — copy/paste if you have it):[/dim]")
    dating_bio = Prompt.ask("[bold]Dating App Bio[/bold]", default="")

    console.print()
    console.print("[dim]Anything else you noticed about her? (vibes, gut feeling, things that stood out):[/dim]")
    notes = Prompt.ask("[bold]Additional Notes[/bold]", default="")

    return SocialProfile(
        name=name,
        age=age,
        instagram_bio=ig_bio,
        instagram_posts_description=ig_posts,
        linkedin_info=linkedin,
        photo_description=photos,
        dating_app_bio=dating_bio,
        additional_notes=notes,
    )


def collect_date_scenario(her_name: str, date_number: int) -> DateScenario:
    print_section(f"Date #{date_number} Plan")

    if date_number == 1:
        console.print("[dim]Where do you want to take her for the first date?[/dim]")
    else:
        console.print("[dim]Where does this date take place?[/dim]")
    console.print()

    location = Prompt.ask("[bold]Location[/bold] (e.g. 'rooftop bar in Manhattan', 'art museum', 'hiking trail')")
    activity = Prompt.ask("[bold]Activity[/bold] (e.g. 'afternoon coffee', 'dinner', 'wine tasting')")

    return DateScenario(location=location, activity=activity)


# ─── Main flow ────────────────────────────────────────────────────────────────

def main():
    print_header()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[red]Error: DEEPSEEK_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        console.print("[dim]Get your key at: platform.deepseek.com[/dim]")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    # Step 1: Collect user profile
    user = collect_user_profile()

    # Step 2: Collect her social profile
    social = collect_social_profile()

    # Step 3: Infer personality
    print_section("Analyzing Personality")
    console.print(f"[dim]Using Claude to infer {social.name}'s personality from her social presence...[/dim]")
    console.print()

    with console.status(f"[bold cyan]Analyzing {social.name}'s personality...[/bold cyan]", spinner="dots"):
        her = infer_personality(social, client)

    print_personality_profile(her)

    # Confirm before proceeding
    console.print()
    proceed = Prompt.ask("[bold]Ready to simulate the date?[/bold]", choices=["yes", "no"], default="yes")
    if proceed == "no":
        console.print("[dim]Goodbye![/dim]")
        return

    # Step 4: Date simulation loop
    date_history = []
    date_number = 1
    previous_result = None

    while True:
        # Collect date scenario
        scenario = collect_date_scenario(her.name, date_number)

        # Run simulation
        print_section(f"Simulating Date #{date_number}")
        console.print(f"[dim]{user.name} and {her.name} at {scenario.location}...[/dim]")
        console.print()
        console.print(Rule(style="dim"))
        console.print()

        def stream_cb(speaker, text):
            stream_conversation(speaker, text, her.name)
            console.print()

        result = simulate_date(
            her=her,
            user=user,
            scenario=scenario,
            client=client,
            date_number=date_number,
            num_exchanges=10,
            previous_date_result=previous_result,
            stream_callback=stream_cb,
        )

        console.print(Rule(style="dim"))

        # Show results
        print_date_result(result, her.name, user.name)
        date_history.append(result)
        previous_result = result

        # Decision point
        console.print()
        if result.next_date_probability < 0.2:
            console.print(f"[red]It doesn't look like {her.name} wants a second date. The simulation ends here.[/red]")
            break

        console.print(f"[bold]Next Date Probability: {result.next_date_probability * 100:.0f}%[/bold]")
        console.print()

        next_action = Prompt.ask(
            "[bold]What next?[/bold]",
            choices=["next-date", "marriage", "quit"],
            default="next-date"
        )

        if next_action == "quit":
            break
        elif next_action == "marriage":
            # Special marriage simulation
            print_section("Marriage Life Simulation")
            scenario = DateScenario(
                location="your shared home",
                activity="a typical Sunday morning, 3 years into marriage",
            )
            date_number = 999  # Signal for marriage mode

            console.print()
            console.print(Rule(style="dim"))
            console.print()

            result = simulate_date(
                her=her,
                user=user,
                scenario=scenario,
                client=client,
                date_number=date_number,
                num_exchanges=8,
                previous_date_result=previous_result,
                stream_callback=stream_cb,
            )

            console.print(Rule(style="dim"))
            print_date_result(result, her.name, user.name)
            break
        else:
            date_number += 1

    console.print()
    console.print(Panel.fit(
        "[bold magenta]Thanks for using WHOISLOVE[/bold magenta]\n[dim]The simulation has ended.[/dim]",
        border_style="magenta"
    ))


if __name__ == "__main__":
    main()

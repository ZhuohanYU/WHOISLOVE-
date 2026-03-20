"""
WHOISLOVE — FastAPI Backend
Serves REST API + static frontend files.
"""
import os
import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.storage import (
    init_db, save_user_profile, load_user_profile,
    save_user_file, load_user_files, delete_user_file,
    save_target, load_all_targets, load_target, delete_target,
    save_file, load_files_for_target, delete_file,
    save_personality, load_latest_personality,
    save_user_personality, load_user_personality,
    save_date_session, load_date_sessions,
    load_all_sessions_with_target, load_target_summary,
    save_session_compat, load_auto_sessions, count_date_sessions,
)
from src.models import SocialProfile, UserProfile, DateScenario, DateResult
from src.personality_inference import infer_personality
from src.date_simulator import simulate_date
from src.file_processor import extract_text, get_filetype_label

# ─── Init ─────────────────────────────────────────────────────────────────────

init_db()

app = FastAPI(title="WHOISLOVE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

executor = ThreadPoolExecutor(max_workers=4)

# In-memory task store for long-running jobs
tasks: dict = {}

# ─── Pydantic models ──────────────────────────────────────────────────────────

class UserProfileIn(BaseModel):
    name: str
    age: int
    occupation: str
    interests: List[str] = []
    personality_description: str = ""
    relationship_goals: str = ""
    communication_style: str = ""
    wechat_signature: str = ""
    wechat_moments_description: str = ""
    instagram_bio: str = ""
    instagram_posts_description: str = ""
    dating_app_bio: str = ""
    additional_notes: str = ""
    lang: str = "en"  # which language user typed in

class TargetIn(BaseModel):
    id: Optional[int] = None
    name: str
    age: Optional[int] = None
    instagram_bio: str = ""
    instagram_posts_description: str = ""
    linkedin_info: str = ""
    facebook_info: str = ""
    photo_description: str = ""
    dating_app_bio: str = ""
    additional_notes: str = ""
    lang: str = "en"  # which language user typed in

class SimulateIn(BaseModel):
    location: str
    activity: str
    special_mode: str = "normal"  # "normal" | "marriage"
    num_exchanges: int = 10
    lang: str = "en"

class AutoSimulateIn(BaseModel):
    num_exchanges: int = 10
    lang: str = "en"

class AnalyzeIn(BaseModel):
    lang: str = "en"

# ─── User Profile ─────────────────────────────────────────────────────────────

@app.get("/api/user")
def get_user(lang: str = "en"):
    user = load_user_profile(lang=lang)
    if not user:
        return {}
    return user

@app.post("/api/user")
def post_user(data: UserProfileIn, background_tasks: BackgroundTasks):
    save_user_profile(data.model_dump())
    return {"ok": True}

# ─── User Files ────────────────────────────────────────────────────────────────

@app.get("/api/user/files")
def get_user_files():
    files = load_user_files()
    for f in files:
        if f.get("extracted_text") and len(f["extracted_text"]) > 300:
            f["extracted_text_preview"] = f["extracted_text"][:300] + "..."
        else:
            f["extracted_text_preview"] = f.get("extracted_text", "")
    return files

@app.post("/api/user/files")
async def upload_user_file(
    platform: str = Form(...),
    file: UploadFile = File(...),
):
    client = get_client()
    file_bytes = await file.read()
    extracted = extract_text(file.filename, file_bytes, client)
    filetype = get_filetype_label(file.filename)
    fid = save_user_file(platform, file.filename, file_bytes, filetype, extracted)
    return {"id": fid, "filename": file.filename, "extracted_preview": extracted[:300]}

@app.delete("/api/user/files/{file_id}")
def del_user_file(file_id: int):
    delete_user_file(file_id)
    return {"ok": True}

# ─── User Personality ─────────────────────────────────────────────────────────

@app.get("/api/user/personality")
def get_user_personality(lang: str = "en"):
    p = load_user_personality(lang=lang)
    if not p:
        return {}
    return {
        "openness": p.openness, "conscientiousness": p.conscientiousness,
        "extraversion": p.extraversion, "agreeableness": p.agreeableness,
        "neuroticism": p.neuroticism, "attachment_style": p.attachment_style,
        "true_interests": p.true_interests, "core_values": p.core_values,
        "communication_style": p.communication_style,
        "relationship_goals": p.relationship_goals,
        "conflict_triggers": p.conflict_triggers, "love_language": p.love_language,
        "personality_summary": p.personality_summary,
        "analysis_reasoning": p.analysis_reasoning,
        "deep_analysis": p.deep_analysis,
    }

@app.post("/api/user/analyze")
def analyze_user(body: AnalyzeIn = None, background_tasks: BackgroundTasks = None):
    if body is None: body = AnalyzeIn()
    task_id = f"analyze_user_{id(object())}"
    tasks[task_id] = {"status": "running", "result": None, "error": None}

    def do_analysis():
        try:
            client = get_client()
            user = load_user_profile()
            if not user:
                raise ValueError("User profile is empty, please fill it in first")

            files = load_user_files()
            file_texts = {"Instagram": [], "WeChat": [], "Dating App": [], "Other": []}
            for f in files:
                platform = f.get("platform", "Other")
                if f.get("extracted_text") and platform in file_texts:
                    file_texts[platform].append(f"[{f['filename']}]\n{f['extracted_text']}")

            def merge(field_val, file_list):
                parts = [field_val] if field_val else []
                parts.extend(file_list)
                return "\n\n---\n\n".join(parts)

            interests_str = ", ".join(user.get("interests") or [])
            extra_notes = "\n".join(filter(None, [
                f"Occupation: {user.get('occupation', '')}" if user.get("occupation") else "",
                f"Interests: {interests_str}" if interests_str else "",
                f"Self-description: {user.get('personality_description', '')}" if user.get("personality_description") else "",
                merge("", file_texts["WeChat"]),
                merge("", file_texts["Other"]),
                user.get("additional_notes", ""),
            ]))

            social = SocialProfile(
                name=user.get("name", "User"),
                age=user.get("age"),
                instagram_bio=user.get("instagram_bio", ""),
                instagram_posts_description=merge(
                    user.get("instagram_posts_description", ""), file_texts["Instagram"]
                ),
                linkedin_info="",
                facebook_info=merge(
                    user.get("wechat_moments_description", ""), file_texts["WeChat"]
                ),
                photo_description="",
                dating_app_bio=merge(
                    user.get("dating_app_bio", ""), file_texts["Dating App"]
                ),
                additional_notes=extra_notes,
            )

            profile = infer_personality(social, client, lang="en")
            save_user_personality(profile, lang="en")
            tasks[task_id]["status"] = "done"
            tasks[task_id]["result"] = "ok"
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    background_tasks.add_task(do_analysis)
    return {"task_id": task_id}

# ─── Targets ──────────────────────────────────────────────────────────────────

@app.get("/api/targets")
def get_targets():
    return load_all_targets()

@app.get("/api/targets/summary")
def get_target_summary_route():
    return load_target_summary()

@app.get("/api/targets/{target_id}")
def get_target(target_id: int, lang: str = "en"):
    t = load_target(target_id, lang=lang)
    if not t:
        raise HTTPException(404, "Target not found")
    return t

@app.post("/api/targets")
def post_target(data: TargetIn, background_tasks: BackgroundTasks):
    raw = data.model_dump()
    tid = save_target(raw)
    return {"id": tid}

@app.put("/api/targets/{target_id}")
def put_target(target_id: int, data: TargetIn, background_tasks: BackgroundTasks):
    d = data.model_dump()
    d["id"] = target_id
    save_target(d)
    return {"ok": True}

@app.delete("/api/targets/{target_id}")
def del_target(target_id: int):
    delete_target(target_id)
    return {"ok": True}

# ─── Files ────────────────────────────────────────────────────────────────────

@app.get("/api/targets/{target_id}/files")
def get_files(target_id: int):
    files = load_files_for_target(target_id)
    # Don't return full extracted_text in list — too heavy
    for f in files:
        if f.get("extracted_text") and len(f["extracted_text"]) > 300:
            f["extracted_text_preview"] = f["extracted_text"][:300] + "..."
        else:
            f["extracted_text_preview"] = f.get("extracted_text", "")
    return files

@app.post("/api/targets/{target_id}/files")
async def upload_file(
    target_id: int,
    platform: str = Form(...),
    file: UploadFile = File(...),
):
    client = get_client()
    file_bytes = await file.read()
    extracted = extract_text(file.filename, file_bytes, client)
    filetype = get_filetype_label(file.filename)
    fid = save_file(target_id, platform, file.filename, file_bytes, filetype, extracted)
    return {"id": fid, "filename": file.filename, "extracted_preview": extracted[:300]}

@app.delete("/api/files/{file_id}")
def del_file(file_id: int):
    delete_file(file_id)
    return {"ok": True}

# ─── Personality Analysis ─────────────────────────────────────────────────────

@app.get("/api/targets/{target_id}/personality")
def get_personality(target_id: int, lang: str = "en"):
    p = load_latest_personality(target_id, lang=lang)
    if not p:
        return {}
    return {
        "name": p.name,
        "age": p.age,
        "openness": p.openness,
        "conscientiousness": p.conscientiousness,
        "extraversion": p.extraversion,
        "agreeableness": p.agreeableness,
        "neuroticism": p.neuroticism,
        "attachment_style": p.attachment_style,
        "true_interests": p.true_interests,
        "core_values": p.core_values,
        "communication_style": p.communication_style,
        "relationship_goals": p.relationship_goals,
        "conflict_triggers": p.conflict_triggers,
        "love_language": p.love_language,
        "personality_summary": p.personality_summary,
        "analysis_reasoning": p.analysis_reasoning,
        "deep_analysis": p.deep_analysis,
    }

@app.post("/api/targets/{target_id}/analyze")
def run_analysis(target_id: int, body: AnalyzeIn = None, background_tasks: BackgroundTasks = None):
    if body is None: body = AnalyzeIn()
    task_id = f"analyze_{target_id}_{id(object())}"
    tasks[task_id] = {"status": "running", "result": None, "error": None}

    def do_analysis():
        try:
            client = get_client()
            target_info = load_target(target_id)
            files = load_files_for_target(target_id)

            file_texts = {"Instagram": [], "LinkedIn": [], "Facebook": [], "Dating App": [], "Other": []}
            for f in files:
                platform = f.get("platform", "Other")
                if f.get("extracted_text") and platform in file_texts:
                    file_texts[platform].append(f"[{f['filename']}]\n{f['extracted_text']}")

            def merge(field_val, file_list):
                parts = [field_val] if field_val else []
                parts.extend(file_list)
                return "\n\n---\n\n".join(parts)

            social = SocialProfile(
                name=target_info["name"],
                age=target_info.get("age"),
                instagram_bio=target_info.get("instagram_bio", ""),
                instagram_posts_description=merge(
                    target_info.get("instagram_posts_description", ""),
                    file_texts["Instagram"]
                ),
                linkedin_info=merge(target_info.get("linkedin_info", ""), file_texts["LinkedIn"]),
                facebook_info=merge(target_info.get("facebook_info", ""), file_texts["Facebook"]),
                photo_description=target_info.get("photo_description", ""),
                dating_app_bio=merge(target_info.get("dating_app_bio", ""), file_texts["Dating App"]),
                additional_notes=merge(target_info.get("additional_notes", ""), file_texts["Other"]),
            )

            profile = infer_personality(social, client, lang="en")
            save_personality(target_id, profile, lang="en")
            tasks[task_id]["status"] = "done"
            tasks[task_id]["result"] = "ok"
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    background_tasks.add_task(do_analysis)
    return {"task_id": task_id}

# ─── Date Simulation ──────────────────────────────────────────────────────────

@app.get("/api/targets/{target_id}/sessions")
def get_sessions(target_id: int, lang: str = "en"):
    return load_date_sessions(target_id, lang=lang)

@app.post("/api/targets/{target_id}/simulate")
def run_simulate(target_id: int, data: SimulateIn, background_tasks: BackgroundTasks):
    task_id = f"sim_{target_id}_{id(object())}"
    tasks[task_id] = {"status": "running", "result": None, "error": None, "conversation": []}

    def do_simulate():
        try:
            client = get_client()
            user_dict = load_user_profile(lang="en") or {}
            user = UserProfile(
                name=user_dict.get("name", "User"),
                age=user_dict.get("age", 25),
                occupation=user_dict.get("occupation", ""),
                interests=user_dict.get("interests") or [],
                personality_description=user_dict.get("personality_description", ""),
                relationship_goals=user_dict.get("relationship_goals", ""),
                communication_style=user_dict.get("communication_style", ""),
            )
            her = load_latest_personality(target_id, lang="en")
            date_number = count_date_sessions(target_id) + 1
            past_sessions = load_date_sessions(target_id, lang="en")

            if data.special_mode == "marriage":
                actual_date_number = 999
                scenario = DateScenario(location="Our shared home", activity="An ordinary Sunday morning three years into marriage")
            else:
                actual_date_number = date_number
                scenario = DateScenario(location=data.location, activity=data.activity)

            prev_result = None
            if past_sessions:
                last = past_sessions[-1]
                prev_result = DateResult(
                    date_number=last["date_number"],
                    summary=last["summary"],
                    chemistry_score=last["chemistry_score"],
                    her_interest_level=last["her_interest_level"],
                    your_performance_score=last["your_performance_score"],
                    next_date_probability=last["next_date_probability"],
                    her_feedback=last["her_feedback"],
                )

            def on_message(speaker, text):
                tasks[task_id]["conversation"].append({"speaker": speaker, "text": text})

            result = simulate_date(
                her=her,
                user=user,
                scenario=scenario,
                client=client,
                date_number=actual_date_number,
                num_exchanges=data.num_exchanges,
                previous_date_result=prev_result,
                stream_callback=on_message,
                lang="en",
                date_history=past_sessions,
            )

            save_date_session(target_id, scenario, result, lang="en")

            tasks[task_id]["status"] = "done"
            tasks[task_id]["result"] = {
                "date_number": result.date_number,
                "summary": result.summary,
                "chemistry_score": result.chemistry_score,
                "her_interest_level": result.her_interest_level,
                "your_performance_score": result.your_performance_score,
                "next_date_probability": result.next_date_probability,
                "conversation_highlights": result.conversation_highlights,
                "awkward_moments": result.awkward_moments,
                "best_moments": result.best_moments,
                "her_feedback": result.her_feedback,
                "advice_for_next_time": result.advice_for_next_time,
                "conversation": tasks[task_id]["conversation"],
                "deep_report": result.deep_report,
            }
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    background_tasks.add_task(do_simulate)
    return {"task_id": task_id}

# ─── Auto simulate ────────────────────────────────────────────────────────────

def _ai_generate_scenario(client: OpenAI, user_dict: dict, her, date_number: int, lang: str) -> DateScenario:
    """Ask AI to pick an appropriate date scenario based on both personalities."""
    interests_her = ", ".join((her.true_interests or [])[:5]) or "varied"
    values_her = ", ".join((her.core_values or [])[:3]) or "not specified"
    interests_user = ", ".join((user_dict.get("interests") or [])[:5]) or "varied"
    lang_instruction = "Respond in English."
    prompt = f"""Two people are going on date #{date_number}. Based on their personalities, suggest the PERFECT date scenario.

Person A (user): {user_dict.get('name','User')}, age {user_dict.get('age',25)}, interests: {interests_user}, personality: {user_dict.get('personality_description','')}
Person B (target): {her.name}, age {her.age}, interests: {interests_her}, values: {values_her}, attachment style: {her.attachment_style}

Rules:
- Make it specific and interesting — not just "coffee shop"
- Match their shared interests if possible
- Consider date number {date_number} (early dates = lower pressure, later = more intimate)
- {lang_instruction}

Return JSON only: {{"location": "...", "activity": "..."}}"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.8,
    )
    data = json.loads(resp.choices[0].message.content)
    return DateScenario(location=data["location"], activity=data["activity"])


@app.post("/api/targets/{target_id}/auto-simulate")
def run_auto_simulate(target_id: int, data: AutoSimulateIn, background_tasks: BackgroundTasks):
    task_id = f"autosim_{target_id}_{id(object())}"
    tasks[task_id] = {"status": "running", "result": None, "error": None, "conversation": [], "scenario": None}

    def do_auto_simulate():
        try:
            client = get_client()
            user_dict = load_user_profile(lang="en") or {}
            user = UserProfile(
                name=user_dict.get("name", "User"),
                age=user_dict.get("age", 25),
                occupation=user_dict.get("occupation", ""),
                interests=user_dict.get("interests") or [],
                personality_description=user_dict.get("personality_description", ""),
                relationship_goals=user_dict.get("relationship_goals", ""),
                communication_style=user_dict.get("communication_style", ""),
            )
            her = load_latest_personality(target_id, lang="en")
            date_number = count_date_sessions(target_id) + 1
            past_sessions = load_date_sessions(target_id, lang="en")

            scenario = _ai_generate_scenario(client, user_dict, her, date_number, "en")
            tasks[task_id]["scenario"] = {"location": scenario.location, "activity": scenario.activity}

            prev_result = None
            if past_sessions:
                last = past_sessions[-1]
                prev_result = DateResult(
                    date_number=last["date_number"],
                    summary=last["summary"],
                    chemistry_score=last["chemistry_score"],
                    her_interest_level=last["her_interest_level"],
                    your_performance_score=last["your_performance_score"],
                    next_date_probability=last["next_date_probability"],
                    her_feedback=last["her_feedback"],
                )

            def on_message(speaker, text):
                tasks[task_id]["conversation"].append({"speaker": speaker, "text": text})

            result = simulate_date(
                her=her, user=user, scenario=scenario, client=client,
                date_number=date_number, num_exchanges=data.num_exchanges,
                previous_date_result=prev_result, stream_callback=on_message,
                lang="en",
                date_history=past_sessions,
            )
            session_id = save_date_session(target_id, scenario, result, lang="en", is_auto=True)

            tasks[task_id]["status"] = "done"
            tasks[task_id]["result"] = {
                "session_id": session_id,
                "date_number": result.date_number,
                "summary": result.summary,
                "chemistry_score": result.chemistry_score,
                "her_interest_level": result.her_interest_level,
                "your_performance_score": result.your_performance_score,
                "next_date_probability": result.next_date_probability,
                "conversation_highlights": result.conversation_highlights,
                "awkward_moments": result.awkward_moments,
                "best_moments": result.best_moments,
                "her_feedback": result.her_feedback,
                "advice_for_next_time": result.advice_for_next_time,
                "conversation": tasks[task_id]["conversation"],
                "deep_report": result.deep_report,
                "scenario": tasks[task_id]["scenario"],
            }
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    background_tasks.add_task(do_auto_simulate)
    return {"task_id": task_id}


@app.post("/api/targets/{target_id}/compatibility")
def get_compatibility_report(target_id: int, lang: str = "en"):
    client = get_client()
    her = load_latest_personality(target_id, lang="en")
    user_dict = load_user_profile(lang="en") or {}
    sessions = load_date_sessions(target_id, lang="en")

    if not her or not sessions:
        raise HTTPException(400, "Need personality analysis and at least one date session")

    avg_chem = sum(s["chemistry_score"] for s in sessions) / len(sessions)
    avg_int = sum(s["her_interest_level"] for s in sessions) / len(sessions)
    avg_perf = sum(s["your_performance_score"] for s in sessions) / len(sessions)
    avg_prob = sum(s["next_date_probability"] for s in sessions) / len(sessions)
    session_summaries = "\n".join(f"Date {s['date_number']}: chemistry={s['chemistry_score']}, interest={s['her_interest_level']}, performance={s['your_performance_score']}, next_prob={s['next_date_probability']:.0%} — {s['summary']}" for s in sessions[-5:])

    lang_instruction = "Write everything in English."
    prompt = f"""You are a relationship compatibility analyst. Based on {len(sessions)} simulated date(s), analyze if these two people should pursue a real relationship.

USER: {user_dict.get('name','User')}, age {user_dict.get('age',25)}, personality: {user_dict.get('personality_description','')}, goals: {user_dict.get('relationship_goals','')}

TARGET: {her.name}, age {her.age}, attachment: {her.attachment_style}, love language: {her.love_language}, goals: {her.relationship_goals}, summary: {her.personality_summary}

SIMULATED DATE RESULTS (avg across {len(sessions)} dates):
- Chemistry: {avg_chem:.1f}/10
- Her interest: {avg_int:.1f}/10
- User performance: {avg_perf:.1f}/10
- Next date probability: {avg_prob:.0%}

Recent dates:
{session_summaries}

{lang_instruction}

Return JSON:
{{
  "compatibility_score": <0-10 float>,
  "verdict": "<2-3 sentence overall verdict>",
  "why_compatible": ["<reason 1>", "<reason 2>", "<reason 3>"],
  "potential_issues": ["<issue 1>", "<issue 2>"],
  "prediction": "<what a real relationship between them would look like>",
  "should_ask_out": <true/false>,
  "recommendation": "<1 concrete actionable sentence on what to do next>"
}}"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=600,
        temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content)


# ─── Auto session archive ─────────────────────────────────────────────────────

@app.get("/api/targets/{target_id}/auto-sessions")
def get_auto_sessions(target_id: int, lang: str = "en"):
    return load_auto_sessions(target_id, lang=lang)


class CompatSaveIn(BaseModel):
    compat_report: dict
    decision: Optional[str] = None  # "yes" | "no" | None


@app.put("/api/sessions/{session_id}/compat")
def save_compat(session_id: int, data: CompatSaveIn):
    save_session_compat(session_id, data.compat_report, data.decision)
    return {"ok": True}


# ─── Cross-target comparison ──────────────────────────────────────────────────

@app.get("/api/sessions/all")
def get_all_sessions(lang: str = "en"):
    return load_all_sessions_with_target(lang=lang)

# ─── Task polling ─────────────────────────────────────────────────────────────

@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

# ─── Serve frontend ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import threading
    import webbrowser
    import uvicorn

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)

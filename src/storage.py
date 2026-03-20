"""
SQLite storage layer for WHOISLOVE.
Stores: user profiles, target profiles, uploaded files, simulation results.
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Optional
from .models import UserProfile, PersonalityProfile, DateResult

DB_PATH = Path(__file__).parent.parent / "data" / "whoislove.db"
UPLOADS_DIR = Path(__file__).parent.parent / "data" / "uploads"


def get_conn():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY,
                name TEXT,
                age INTEGER,
                occupation TEXT,
                interests TEXT,
                personality_description TEXT,
                relationship_goals TEXT,
                communication_style TEXT,
                wechat_signature TEXT,
                wechat_moments_description TEXT,
                instagram_bio TEXT,
                instagram_posts_description TEXT,
                dating_app_bio TEXT,
                additional_notes TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT,
                filename TEXT,
                filepath TEXT,
                filetype TEXT,
                extracted_text TEXT,
                uploaded_at TEXT
            );

            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                instagram_bio TEXT,
                instagram_posts_description TEXT,
                linkedin_info TEXT,
                facebook_info TEXT,
                photo_description TEXT,
                dating_app_bio TEXT,
                additional_notes TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS target_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id INTEGER,
                platform TEXT,
                filename TEXT,
                filepath TEXT,
                filetype TEXT,
                extracted_text TEXT,
                uploaded_at TEXT,
                FOREIGN KEY (target_id) REFERENCES targets(id)
            );

            CREATE TABLE IF NOT EXISTS personality_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id INTEGER,
                openness REAL,
                conscientiousness REAL,
                extraversion REAL,
                agreeableness REAL,
                neuroticism REAL,
                attachment_style TEXT,
                true_interests TEXT,
                core_values TEXT,
                communication_style TEXT,
                relationship_goals TEXT,
                conflict_triggers TEXT,
                love_language TEXT,
                personality_summary TEXT,
                analysis_reasoning TEXT,
                deep_analysis TEXT,
                lang TEXT DEFAULT 'zh',
                created_at TEXT,
                FOREIGN KEY (target_id) REFERENCES targets(id)
            );

            CREATE TABLE IF NOT EXISTS date_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id INTEGER,
                date_number INTEGER,
                location TEXT,
                activity TEXT,
                chemistry_score REAL,
                her_interest_level REAL,
                your_performance_score REAL,
                next_date_probability REAL,
                summary TEXT,
                conversation_highlights TEXT,
                awkward_moments TEXT,
                best_moments TEXT,
                her_feedback TEXT,
                advice_for_next_time TEXT,
                full_conversation TEXT,
                deep_report TEXT,
                lang TEXT DEFAULT 'zh',
                created_at TEXT,
                FOREIGN KEY (target_id) REFERENCES targets(id)
            );
        """)


# ─── User Profile ─────────────────────────────────────────────────────────────

def _migrate_user_profile_cols(conn):
    for col, defn in [("translations_cache", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE user_profile ADD COLUMN {col} {defn}")
        except Exception:
            pass  # column already exists


def save_user_profile(data: dict, translations: Optional[dict] = None):
    """Save user profile with all social fields. Optionally cache translations."""
    with get_conn() as conn:
        _migrate_user_profile_cols(conn)
        existing = conn.execute("SELECT id, translations_cache FROM user_profile LIMIT 1").fetchone()
        now = datetime.now().isoformat()

        # Merge new translations into existing cache
        existing_trans = {}
        if existing and existing["translations_cache"]:
            try:
                existing_trans = json.loads(existing["translations_cache"])
            except Exception:
                pass
        if translations:
            existing_trans.update(translations)
        trans_json = json.dumps(existing_trans, ensure_ascii=False) if existing_trans else None

        fields = (
            data.get("name"), data.get("age"), data.get("occupation"),
            json.dumps(data.get("interests", [])),
            data.get("personality_description", ""),
            data.get("relationship_goals", ""),
            data.get("communication_style", ""),
            data.get("wechat_signature", ""),
            data.get("wechat_moments_description", ""),
            data.get("instagram_bio", ""),
            data.get("instagram_posts_description", ""),
            data.get("dating_app_bio", ""),
            data.get("additional_notes", ""),
            trans_json,
            now,
        )
        if existing:
            conn.execute("""
                UPDATE user_profile SET
                    name=?, age=?, occupation=?, interests=?,
                    personality_description=?, relationship_goals=?,
                    communication_style=?, wechat_signature=?,
                    wechat_moments_description=?, instagram_bio=?,
                    instagram_posts_description=?, dating_app_bio=?,
                    additional_notes=?, translations_cache=?, updated_at=?
                WHERE id=?
            """, (*fields, existing["id"]))
        else:
            conn.execute("""
                INSERT INTO user_profile
                    (name, age, occupation, interests, personality_description,
                     relationship_goals, communication_style, wechat_signature,
                     wechat_moments_description, instagram_bio,
                     instagram_posts_description, dating_app_bio,
                     additional_notes, translations_cache, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fields)


def load_user_profile(lang: str = "zh") -> Optional[dict]:
    """Load full user profile as dict, applying cached translation for `lang` if available."""
    with get_conn() as conn:
        _migrate_user_profile_cols(conn)
        row = conn.execute("SELECT * FROM user_profile LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        d["interests"] = json.loads(d.get("interests") or "[]")
        # Apply translation overlay if available and lang differs from raw data
        trans_cache = {}
        if d.get("translations_cache"):
            try:
                trans_cache = json.loads(d["translations_cache"])
            except Exception:
                pass
        if lang in trans_cache:
            overlay = trans_cache[lang]
            for k, v in overlay.items():
                if v:
                    d[k] = v
        d.pop("translations_cache", None)
        return d


def save_user_profile_translation(lang: str, translation: dict):
    """Store a translation overlay for the user profile without overwriting original data."""
    with get_conn() as conn:
        _migrate_user_profile_cols(conn)
        existing = conn.execute("SELECT id, translations_cache FROM user_profile LIMIT 1").fetchone()
        if not existing:
            return
        cache = {}
        if existing["translations_cache"]:
            try:
                cache = json.loads(existing["translations_cache"])
            except Exception:
                pass
        cache[lang] = translation
        conn.execute("UPDATE user_profile SET translations_cache=? WHERE id=?",
                     (json.dumps(cache, ensure_ascii=False), existing["id"]))


# ─── User Files ───────────────────────────────────────────────────────────────

def save_user_file(platform: str, filename: str,
                   file_bytes: bytes, filetype: str, extracted_text: str) -> int:
    user_dir = UPLOADS_DIR / "user" / platform
    user_dir.mkdir(parents=True, exist_ok=True)
    filepath = user_dir / filename
    filepath.write_bytes(file_bytes)
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO user_files
                (platform, filename, filepath, filetype, extracted_text, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (platform, filename, str(filepath), filetype,
              extracted_text, datetime.now().isoformat()))
        return cur.lastrowid


def load_user_files() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM user_files ORDER BY uploaded_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_user_file(file_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT filepath FROM user_files WHERE id=?", (file_id,)).fetchone()
        if row:
            p = Path(row["filepath"])
            if p.exists():
                p.unlink()
        conn.execute("DELETE FROM user_files WHERE id=?", (file_id,))


# ─── User Personality (target_id = 0 代表用户自己) ────────────────────────────

def _migrate_personality_cols(conn):
    """Add missing columns to personality_profiles table."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(personality_profiles)").fetchall()]
    for col, defn in [
        ("deep_analysis", "TEXT"),
        ("lang", "TEXT DEFAULT 'en'"),
        ("mbti_type", "TEXT DEFAULT ''"),
        ("humor_style", "TEXT DEFAULT ''"),
        ("verbal_patterns", "TEXT DEFAULT '[]'"),
        ("green_flags", "TEXT DEFAULT '[]'"),
        ("deal_breakers", "TEXT DEFAULT '[]'"),
        ("date_behavior", "TEXT DEFAULT ''"),
        ("trust_stages", "TEXT DEFAULT '{}'"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE personality_profiles ADD COLUMN {col} {defn}")


def _build_personality(row, name, age) -> PersonalityProfile:
    keys = row.keys()
    return PersonalityProfile(
        name=name, age=age,
        openness=row["openness"], conscientiousness=row["conscientiousness"],
        extraversion=row["extraversion"], agreeableness=row["agreeableness"],
        neuroticism=row["neuroticism"], attachment_style=row["attachment_style"],
        mbti_type=row["mbti_type"] if "mbti_type" in keys else "",
        true_interests=json.loads(row["true_interests"] or "[]"),
        core_values=json.loads(row["core_values"] or "[]"),
        communication_style=row["communication_style"] or "",
        relationship_goals=row["relationship_goals"] or "",
        conflict_triggers=json.loads(row["conflict_triggers"] or "[]"),
        love_language=row["love_language"] or "",
        humor_style=row["humor_style"] if "humor_style" in keys else "",
        verbal_patterns=json.loads(row["verbal_patterns"] or "[]") if "verbal_patterns" in keys else [],
        green_flags=json.loads(row["green_flags"] or "[]") if "green_flags" in keys else [],
        deal_breakers=json.loads(row["deal_breakers"] or "[]") if "deal_breakers" in keys else [],
        date_behavior=row["date_behavior"] if "date_behavior" in keys else "",
        trust_stages=json.loads(row["trust_stages"] or "{}") if "trust_stages" in keys else {},
        personality_summary=row["personality_summary"] or "",
        analysis_reasoning=row["analysis_reasoning"] or "",
        deep_analysis=json.loads(row["deep_analysis"] or "{}") if "deep_analysis" in keys else {},
    )


def save_user_personality(profile, lang: str = "en") -> None:
    """Save personality analysis for the app user (target_id=0)."""
    with get_conn() as conn:
        _migrate_personality_cols(conn)
        conn.execute("""
            INSERT INTO personality_profiles
                (target_id, openness, conscientiousness, extraversion, agreeableness,
                 neuroticism, attachment_style, mbti_type, true_interests, core_values,
                 communication_style, relationship_goals, conflict_triggers,
                 love_language, humor_style, verbal_patterns, green_flags, deal_breakers,
                 date_behavior, trust_stages, personality_summary, analysis_reasoning,
                 deep_analysis, lang, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            0, profile.openness, profile.conscientiousness,
            profile.extraversion, profile.agreeableness, profile.neuroticism,
            profile.attachment_style, profile.mbti_type,
            json.dumps(profile.true_interests, ensure_ascii=False),
            json.dumps(profile.core_values, ensure_ascii=False),
            profile.communication_style, profile.relationship_goals,
            json.dumps(profile.conflict_triggers, ensure_ascii=False),
            profile.love_language, profile.humor_style,
            json.dumps(profile.verbal_patterns, ensure_ascii=False),
            json.dumps(profile.green_flags, ensure_ascii=False),
            json.dumps(profile.deal_breakers, ensure_ascii=False),
            profile.date_behavior,
            json.dumps(profile.trust_stages, ensure_ascii=False),
            profile.personality_summary, profile.analysis_reasoning,
            json.dumps(getattr(profile, "deep_analysis", {}), ensure_ascii=False),
            lang, datetime.now().isoformat()
        ))


def load_user_personality(lang: str = "zh") -> Optional[PersonalityProfile]:
    """Load latest personality analysis for the app user (target_id=0) in the given language."""
    user = load_user_profile()
    with get_conn() as conn:
        _migrate_personality_cols(conn)
        row = conn.execute("""
            SELECT * FROM personality_profiles
            WHERE target_id = 0 AND (lang = ? OR lang IS NULL)
            ORDER BY created_at DESC LIMIT 1
        """, (lang,)).fetchone()
        if not row:
            return None
        name = user["name"] if user else "User"
        age = user["age"] if user else None
        return _build_personality(row, name, age)


# ─── Targets ──────────────────────────────────────────────────────────────────

def save_target(data: dict) -> int:
    """Create or update a target. Returns target_id."""
    now = datetime.now().isoformat()
    with get_conn() as conn:
        existing_id = data.get("id")
        if existing_id:
            conn.execute("""
                UPDATE targets SET
                    name=?, age=?, instagram_bio=?, instagram_posts_description=?,
                    linkedin_info=?, facebook_info=?, photo_description=?,
                    dating_app_bio=?, additional_notes=?, updated_at=?
                WHERE id=?
            """, (
                data["name"], data.get("age"), data.get("instagram_bio", ""),
                data.get("instagram_posts_description", ""), data.get("linkedin_info", ""),
                data.get("facebook_info", ""), data.get("photo_description", ""),
                data.get("dating_app_bio", ""), data.get("additional_notes", ""),
                now, existing_id
            ))
            return existing_id
        else:
            cur = conn.execute("""
                INSERT INTO targets
                    (name, age, instagram_bio, instagram_posts_description,
                     linkedin_info, facebook_info, photo_description,
                     dating_app_bio, additional_notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["name"], data.get("age"), data.get("instagram_bio", ""),
                data.get("instagram_posts_description", ""), data.get("linkedin_info", ""),
                data.get("facebook_info", ""), data.get("photo_description", ""),
                data.get("dating_app_bio", ""), data.get("additional_notes", ""),
                now, now
            ))
            return cur.lastrowid


def _migrate_target_cols(conn):
    try:
        conn.execute("ALTER TABLE targets ADD COLUMN translations_cache TEXT")
    except Exception:
        pass  # column already exists


def load_all_targets() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM targets ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]


def load_target(target_id: int, lang: str = "zh") -> Optional[dict]:
    with get_conn() as conn:
        _migrate_target_cols(conn)
        row = conn.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        trans_cache = {}
        if d.get("translations_cache"):
            try:
                trans_cache = json.loads(d["translations_cache"])
            except Exception:
                pass
        if lang in trans_cache:
            for k, v in trans_cache[lang].items():
                if v:
                    d[k] = v
        d.pop("translations_cache", None)
        return d


def save_target_translation(target_id: int, lang: str, translation: dict):
    """Store a translation overlay for a target without overwriting original data."""
    with get_conn() as conn:
        _migrate_target_cols(conn)
        row = conn.execute("SELECT translations_cache FROM targets WHERE id=?", (target_id,)).fetchone()
        if not row:
            return
        cache = {}
        if row["translations_cache"]:
            try:
                cache = json.loads(row["translations_cache"])
            except Exception:
                pass
        cache[lang] = translation
        conn.execute("UPDATE targets SET translations_cache=? WHERE id=?",
                     (json.dumps(cache, ensure_ascii=False), target_id))


def delete_target(target_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM target_files WHERE target_id=?", (target_id,))
        conn.execute("DELETE FROM personality_profiles WHERE target_id=?", (target_id,))
        conn.execute("DELETE FROM date_sessions WHERE target_id=?", (target_id,))
        conn.execute("DELETE FROM targets WHERE id=?", (target_id,))


# ─── Files ────────────────────────────────────────────────────────────────────

def save_file(target_id: int, platform: str, filename: str,
              file_bytes: bytes, filetype: str, extracted_text: str) -> int:
    target_dir = UPLOADS_DIR / str(target_id) / platform
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename
    filepath.write_bytes(file_bytes)

    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO target_files
                (target_id, platform, filename, filepath, filetype, extracted_text, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (target_id, platform, filename, str(filepath), filetype,
              extracted_text, datetime.now().isoformat()))
        return cur.lastrowid


def load_files_for_target(target_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM target_files WHERE target_id=? ORDER BY uploaded_at DESC",
            (target_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_file(file_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT filepath FROM target_files WHERE id=?", (file_id,)).fetchone()
        if row:
            path = Path(row["filepath"])
            if path.exists():
                path.unlink()
        conn.execute("DELETE FROM target_files WHERE id=?", (file_id,))


# ─── Personality Profiles ─────────────────────────────────────────────────────

def save_personality(target_id: int, profile: PersonalityProfile, lang: str = "en") -> int:
    with get_conn() as conn:
        _migrate_personality_cols(conn)
        cur = conn.execute("""
            INSERT INTO personality_profiles
                (target_id, openness, conscientiousness, extraversion, agreeableness,
                 neuroticism, attachment_style, mbti_type, true_interests, core_values,
                 communication_style, relationship_goals, conflict_triggers,
                 love_language, humor_style, verbal_patterns, green_flags, deal_breakers,
                 date_behavior, trust_stages, personality_summary, analysis_reasoning,
                 deep_analysis, lang, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_id, profile.openness, profile.conscientiousness,
            profile.extraversion, profile.agreeableness, profile.neuroticism,
            profile.attachment_style, profile.mbti_type,
            json.dumps(profile.true_interests, ensure_ascii=False),
            json.dumps(profile.core_values, ensure_ascii=False),
            profile.communication_style, profile.relationship_goals,
            json.dumps(profile.conflict_triggers, ensure_ascii=False),
            profile.love_language, profile.humor_style,
            json.dumps(profile.verbal_patterns, ensure_ascii=False),
            json.dumps(profile.green_flags, ensure_ascii=False),
            json.dumps(profile.deal_breakers, ensure_ascii=False),
            profile.date_behavior,
            json.dumps(profile.trust_stages, ensure_ascii=False),
            profile.personality_summary, profile.analysis_reasoning,
            json.dumps(getattr(profile, "deep_analysis", {}), ensure_ascii=False),
            lang, datetime.now().isoformat()
        ))
        return cur.lastrowid


def load_latest_personality(target_id: int, lang: str = "zh") -> Optional[PersonalityProfile]:
    with get_conn() as conn:
        _migrate_personality_cols(conn)
        row = conn.execute("""
            SELECT p.*, t.name, t.age FROM personality_profiles p
            JOIN targets t ON p.target_id = t.id
            WHERE p.target_id=? AND (p.lang=? OR p.lang IS NULL)
            ORDER BY p.created_at DESC LIMIT 1
        """, (target_id, lang)).fetchone()
        if not row and lang != "zh":
            # Fallback: try the other language
            row = conn.execute("""
                SELECT p.*, t.name, t.age FROM personality_profiles p
                JOIN targets t ON p.target_id = t.id
                WHERE p.target_id=?
                ORDER BY p.created_at DESC LIMIT 1
            """, (target_id,)).fetchone()
        if not row:
            return None
        return _build_personality(row, row["name"], row["age"])


# ─── Date Sessions ────────────────────────────────────────────────────────────

def save_date_session(target_id: int, scenario, result: DateResult, lang: str = "zh", is_auto: bool = False) -> int:
    deep = json.dumps(result.deep_report, ensure_ascii=False) if result.deep_report else None
    with get_conn() as conn:
        _migrate_date_session_cols(conn)
        cur = conn.execute("""
            INSERT INTO date_sessions
                (target_id, date_number, location, activity,
                 chemistry_score, her_interest_level, your_performance_score,
                 next_date_probability, summary, conversation_highlights,
                 awkward_moments, best_moments, her_feedback,
                 advice_for_next_time, full_conversation, deep_report, lang, is_auto, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_id, result.date_number, scenario.location, scenario.activity,
            result.chemistry_score, result.her_interest_level,
            result.your_performance_score, result.next_date_probability,
            result.summary, json.dumps(result.conversation_highlights, ensure_ascii=False),
            json.dumps(result.awkward_moments, ensure_ascii=False),
            json.dumps(result.best_moments, ensure_ascii=False),
            result.her_feedback, json.dumps(result.advice_for_next_time, ensure_ascii=False),
            result.full_conversation, deep, lang, 1 if is_auto else 0, datetime.now().isoformat()
        ))
        return cur.lastrowid


def save_session_compat(session_id: int, compat_report: dict, decision: Optional[str] = None):
    """Attach a compatibility report (and optional decision) to a session."""
    with get_conn() as conn:
        _migrate_date_session_cols(conn)
        conn.execute(
            "UPDATE date_sessions SET compat_report=?, compat_decision=? WHERE id=?",
            (json.dumps(compat_report, ensure_ascii=False), decision, session_id)
        )


def load_auto_sessions(target_id: int, lang: str = "zh") -> list[dict]:
    """Load only auto-simulated sessions for a target."""
    with get_conn() as conn:
        _migrate_date_session_cols(conn)
        rows = conn.execute("""
            SELECT * FROM date_sessions
            WHERE target_id=? AND is_auto=1 AND (lang=? OR lang IS NULL)
            ORDER BY created_at DESC
        """, (target_id, lang)).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["conversation_highlights"] = json.loads(r["conversation_highlights"] or "[]")
            r["awkward_moments"] = json.loads(r["awkward_moments"] or "[]")
            r["best_moments"] = json.loads(r["best_moments"] or "[]")
            r["advice_for_next_time"] = json.loads(r["advice_for_next_time"] or "[]")
            r["deep_report"] = json.loads(r["deep_report"]) if r.get("deep_report") else {}
            r["compat_report"] = json.loads(r["compat_report"]) if r.get("compat_report") else None
            results.append(r)
        return results


def _migrate_date_session_cols(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(date_sessions)").fetchall()]
    for col, defn in [
        ("deep_report", "TEXT"),
        ("lang", "TEXT DEFAULT 'zh'"),
        ("is_auto", "INTEGER DEFAULT 0"),
        ("compat_report", "TEXT"),
        ("compat_decision", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE date_sessions ADD COLUMN {col} {defn}")


def count_date_sessions(target_id: int) -> int:
    """Count all sessions for a target regardless of language (for date_number computation)."""
    with get_conn() as conn:
        _migrate_date_session_cols(conn)
        row = conn.execute(
            "SELECT COUNT(*) FROM date_sessions WHERE target_id=?", (target_id,)
        ).fetchone()
        return row[0] if row else 0


def load_date_sessions(target_id: int, lang: str = "zh") -> list[dict]:
    with get_conn() as conn:
        _migrate_date_session_cols(conn)
        rows = conn.execute("""
            SELECT * FROM date_sessions
            WHERE target_id=? AND (lang=? OR lang IS NULL)
            ORDER BY date_number ASC, created_at ASC
        """, (target_id, lang)).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["conversation_highlights"] = json.loads(r["conversation_highlights"] or "[]")
            r["awkward_moments"] = json.loads(r["awkward_moments"] or "[]")
            r["best_moments"] = json.loads(r["best_moments"] or "[]")
            r["advice_for_next_time"] = json.loads(r["advice_for_next_time"] or "[]")
            r["deep_report"] = json.loads(r["deep_report"]) if r.get("deep_report") else {}
            results.append(r)
        return results


def load_all_sessions_with_target(lang: str = "zh") -> list[dict]:
    """Load every session joined with target name/age, for cross-target comparison."""
    with get_conn() as conn:
        _migrate_date_session_cols(conn)
        rows = conn.execute("""
            SELECT ds.*, t.name AS target_name, t.age AS target_age
            FROM date_sessions ds
            JOIN targets t ON ds.target_id = t.id
            WHERE ds.lang=? OR ds.lang IS NULL
            ORDER BY ds.created_at DESC
        """, (lang,)).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["conversation_highlights"] = json.loads(r["conversation_highlights"] or "[]")
            r["awkward_moments"] = json.loads(r["awkward_moments"] or "[]")
            r["best_moments"] = json.loads(r["best_moments"] or "[]")
            r["advice_for_next_time"] = json.loads(r["advice_for_next_time"] or "[]")
            r["deep_report"] = json.loads(r["deep_report"]) if r.get("deep_report") else {}
            results.append(r)
        return results


def load_target_summary() -> list[dict]:
    """Per-target aggregated stats for comparison view."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT
                t.id, t.name, t.age,
                COUNT(ds.id) AS session_count,
                ROUND(AVG(ds.chemistry_score), 2) AS avg_chemistry,
                ROUND(AVG(ds.her_interest_level), 2) AS avg_interest,
                ROUND(AVG(ds.your_performance_score), 2) AS avg_performance,
                ROUND(MAX(ds.next_date_probability), 2) AS best_probability,
                MAX(ds.created_at) AS last_date
            FROM targets t
            LEFT JOIN date_sessions ds ON t.id = ds.target_id
            GROUP BY t.id
            ORDER BY avg_chemistry DESC NULLS LAST
        """).fetchall()
        return [dict(r) for r in rows]

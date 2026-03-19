"""
seed.py — 从 backend/fixtures/ 恢复数据到数据库
用法: python backend/seed.py [--clean]

  --clean   先清空数据库再导入（默认：追加，重复的 user 会更新）
"""
import sys, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.storage import (
    init_db, get_conn,
    save_user_profile,
    save_target, save_personality, save_date_session,
)
from src.models import PersonalityProfile, DateScenario, DateResult

FIXTURES = Path(__file__).parent / "fixtures"


def clean_db():
    """清空所有业务表，保留表结构。"""
    with get_conn() as conn:
        conn.executescript("""
            DELETE FROM date_sessions;
            DELETE FROM personality_profiles;
            DELETE FROM target_files;
            DELETE FROM targets;
            DELETE FROM user_files;
            DELETE FROM user_profile;
        """)
    print("🗑️  数据库已清空\n")


def seed():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="清空后再导入")
    args = parser.parse_args()

    init_db()

    if args.clean:
        clean_db()

    print("🌱 开始从 backend/fixtures/ 恢复数据\n")

    # ── 用户资料 ────────────────────────────────────────────────────────────────
    user_path = FIXTURES / "user.json"
    if user_path.exists():
        user = json.loads(user_path.read_text(encoding="utf-8"))
        # interests 可能已是 list，save_user_profile 期待 dict
        save_user_profile(user)
        print(f"  ✅ 用户资料: {user.get('name', '未命名')}")
    else:
        print("  ⚠️  未找到 user.json，跳过")

    # ── 对象 + 人格分析 + 约会记录 ─────────────────────────────────────────────
    targets_path = FIXTURES / "targets.json"
    if not targets_path.exists():
        print("  ⚠️  未找到 targets.json，跳过")
    else:
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        print(f"  导入 {len(targets)} 个对象：")

        for t in targets:
            personality_data = t.pop("_personality", None)
            sessions_data    = t.pop("_sessions", [])
            t.pop("_files", None)   # 二进制文件不自动恢复
            t.pop("id", None)       # 去掉旧 id，让数据库自动分配

            # 保存对象基本信息，拿到新 id
            new_id = save_target(t)

            # 恢复人格分析
            if personality_data:
                profile = PersonalityProfile(
                    name=t["name"],
                    age=t.get("age"),
                    **personality_data,
                )
                save_personality(new_id, profile)

            # 恢复约会记录
            for s in sessions_data:
                scenario = DateScenario(
                    location=s.get("location", ""),
                    activity=s.get("activity", ""),
                )
                result = DateResult(
                    date_number=s["date_number"],
                    summary=s.get("summary", ""),
                    chemistry_score=s.get("chemistry_score", 5.0),
                    her_interest_level=s.get("her_interest_level", 5.0),
                    your_performance_score=s.get("your_performance_score", 5.0),
                    next_date_probability=s.get("next_date_probability", 0.5),
                    conversation_highlights=s.get("conversation_highlights", []),
                    awkward_moments=s.get("awkward_moments", []),
                    best_moments=s.get("best_moments", []),
                    her_feedback=s.get("her_feedback", ""),
                    advice_for_next_time=s.get("advice_for_next_time", []),
                    full_conversation=s.get("full_conversation", ""),
                )
                save_date_session(new_id, scenario, result)

            n_s = len(sessions_data)
            has_p = "✓ 人格" if personality_data else "✗ 人格"
            print(f"       └─ {t['name']} ({t.get('age','?')}岁)  {n_s} 次约会  {has_p}  → id={new_id}")

    print(f"\n✨ 恢复完成！启动服务后直接使用：uvicorn server:app --reload")


if __name__ == "__main__":
    seed()

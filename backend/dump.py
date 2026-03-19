"""
dump.py — 导出当前数据库到 backend/fixtures/
用法: python backend/dump.py
"""
import sys, json
from pathlib import Path

# 让 src.storage 可以被 import
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.storage import (
    get_conn, load_user_profile, load_user_files,
    load_all_targets, load_files_for_target,
    load_latest_personality, load_date_sessions,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)


def dump():
    print("📦 开始导出数据库 → backend/fixtures/\n")

    # ── 用户资料 ────────────────────────────────────────────────────────────────
    user = load_user_profile()
    if user:
        (FIXTURES / "user.json").write_text(
            json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✅ 用户资料: {user.get('name', '未命名')}")
    else:
        print("  ⚠️  没有用户资料")

    # ── 用户上传的文件（只存元数据，不复制二进制）────────────────────────────────
    user_files = load_user_files()
    (FIXTURES / "user_files.json").write_text(
        json.dumps(user_files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✅ 用户文件记录: {len(user_files)} 条")

    # ── 对象 ─────────────────────────────────────────────────────────────────────
    targets = load_all_targets()
    targets_export = []
    for t in targets:
        entry = dict(t)

        # 该对象的上传文件元数据
        entry["_files"] = load_files_for_target(t["id"])

        # 该对象最新的人格分析
        p = load_latest_personality(t["id"])
        if p:
            entry["_personality"] = {
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
            }
        else:
            entry["_personality"] = None

        # 该对象的所有约会记录
        entry["_sessions"] = load_date_sessions(t["id"])

        targets_export.append(entry)

    (FIXTURES / "targets.json").write_text(
        json.dumps(targets_export, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✅ 对象: {len(targets)} 个")
    for t in targets_export:
        n_s = len(t["_sessions"])
        has_p = "有人格分析" if t["_personality"] else "无人格分析"
        print(f"       └─ {t['name']} ({t.get('age','?')}岁)  {n_s} 次约会  {has_p}")

    print(f"\n✨ 导出完成 → {FIXTURES}/")


if __name__ == "__main__":
    dump()

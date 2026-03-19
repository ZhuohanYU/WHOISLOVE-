"""
WHOISLOVE — Dating Simulation World
Streamlit UI
"""
import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()

from src.storage import (
    init_db, save_user_profile, load_user_profile,
    save_target, load_all_targets, load_target, delete_target,
    save_file, load_files_for_target, delete_file,
    save_personality, load_latest_personality,
    save_date_session, load_date_sessions,
)
from src.models import UserProfile, SocialProfile, DateScenario
from src.personality_inference import infer_personality
from src.date_simulator import simulate_date
from src.file_processor import extract_text, get_filetype_label

# ─── Init ─────────────────────────────────────────────────────────────────────

init_db()

st.set_page_config(
    page_title="WHOISLOVE",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

client = get_client()

# ─── Sidebar navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.title("💘 WHOISLOVE")
    st.caption("Dating Simulation World")
    st.divider()

    page = st.radio(
        "导航",
        ["👤 我的资料", "💁‍♀️ 她的资料", "🧠 人格分析", "💬 约会模拟", "📋 历史记录"],
        label_visibility="collapsed",
    )

    st.divider()

    # Target selector (shared across pages)
    targets = load_all_targets()
    if targets:
        target_options = {f"{t['name']} (ID:{t['id']})": t["id"] for t in targets}
        selected_label = st.selectbox("当前对象", list(target_options.keys()))
        st.session_state["current_target_id"] = target_options[selected_label]
    else:
        st.info("还没有添加过对象")
        st.session_state["current_target_id"] = None

    if not client:
        st.error("未找到 DEEPSEEK_API_KEY")


# ─── Page: 我的资料 ────────────────────────────────────────────────────────────

if page == "👤 我的资料":
    st.title("👤 我的资料")
    st.caption("填写你自己的基本信息，约会模拟时会用到。")

    existing = load_user_profile()

    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("姓名", value=existing.name if existing else "")
            age = st.number_input("年龄", min_value=18, max_value=80,
                                  value=existing.age if existing else 25)
            occupation = st.text_input("职业", value=existing.occupation if existing else "")
        with col2:
            communication_style = st.text_input(
                "沟通风格",
                value=existing.communication_style if existing else "",
                placeholder="例：幽默风趣，喜欢深度对话，有时话不多"
            )
            relationship_goals = st.text_input(
                "寻找的关系类型",
                value=existing.relationship_goals if existing else "",
                placeholder="例：认真的长期关系"
            )

        personality_description = st.text_area(
            "性格描述",
            value=existing.personality_description if existing else "",
            placeholder="用几句话描述你的性格特点...",
            height=100,
        )

        interests_raw = st.text_input(
            "兴趣爱好（逗号分隔）",
            value=", ".join(existing.interests) if existing else "",
            placeholder="例：摄影, 爬山, 读书, 咖啡, 旅行"
        )

        submitted = st.form_submit_button("💾 保存", type="primary", use_container_width=True)
        if submitted:
            if not name:
                st.error("姓名不能为空")
            else:
                user = UserProfile(
                    name=name,
                    age=int(age),
                    occupation=occupation,
                    interests=[i.strip() for i in interests_raw.split(",") if i.strip()],
                    personality_description=personality_description,
                    relationship_goals=relationship_goals,
                    communication_style=communication_style,
                )
                save_user_profile(user)
                st.success("✅ 资料已保存！")
                st.rerun()

    if existing:
        st.divider()
        st.subheader("当前保存的资料")
        col1, col2, col3 = st.columns(3)
        col1.metric("姓名", existing.name)
        col2.metric("年龄", existing.age)
        col3.metric("职业", existing.occupation)
        if existing.interests:
            st.write("**兴趣爱好：**", " · ".join(existing.interests))


# ─── Page: 她的资料 ────────────────────────────────────────────────────────────

elif page == "💁‍♀️ 她的资料":
    st.title("💁‍♀️ 她的资料")

    tab_list, tab_edit, tab_files = st.tabs(["📋 所有对象", "✏️ 编辑 / 新建", "📎 上传文件"])

    # ── Tab: 所有对象 ──
    with tab_list:
        if not targets:
            st.info("还没有添加过对象，去「编辑 / 新建」创建第一个吧。")
        else:
            for t in targets:
                with st.expander(f"**{t['name']}**  {t['age'] or '?'} 岁  ·  ID:{t['id']}"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if t.get("dating_app_bio"):
                            st.caption(t["dating_app_bio"][:200])
                    with col2:
                        if st.button("🗑️ 删除", key=f"del_{t['id']}"):
                            delete_target(t["id"])
                            st.rerun()

    # ── Tab: 编辑 / 新建 ──
    with tab_edit:
        target_id = st.session_state.get("current_target_id")
        existing_t = load_target(target_id) if target_id else None

        st.subheader("基本信息" if existing_t else "新建对象")

        with st.form("target_form"):
            col1, col2 = st.columns(2)
            with col1:
                t_name = st.text_input("姓名 / 昵称 *", value=existing_t["name"] if existing_t else "")
            with col2:
                t_age_val = existing_t["age"] if existing_t and existing_t["age"] else 25
                t_age = st.number_input("年龄", min_value=18, max_value=60, value=t_age_val)

            st.markdown("**Instagram**")
            ig_bio = st.text_input("Instagram Bio",
                                   value=existing_t.get("instagram_bio", "") if existing_t else "")
            ig_posts = st.text_area("Instagram 帖子描述",
                                    value=existing_t.get("instagram_posts_description", "") if existing_t else "",
                                    placeholder="描述她发什么内容、多久发一次、整体风格...",
                                    height=100)

            st.markdown("**LinkedIn**")
            linkedin = st.text_area("LinkedIn 信息",
                                    value=existing_t.get("linkedin_info", "") if existing_t else "",
                                    placeholder="职位、公司、教育背景...",
                                    height=80)

            st.markdown("**Facebook**")
            facebook = st.text_area("Facebook 信息",
                                    value=existing_t.get("facebook_info", "") if existing_t else "",
                                    placeholder="公开的内容、家庭信息、政治倾向...",
                                    height=80)

            st.markdown("**照片 & 约会App**")
            col3, col4 = st.columns(2)
            with col3:
                photos = st.text_area("照片描述",
                                      value=existing_t.get("photo_description", "") if existing_t else "",
                                      placeholder="描述她的穿衣风格、常去的地方、照片中的人...",
                                      height=100)
            with col4:
                dating_bio = st.text_area("约会App Bio",
                                          value=existing_t.get("dating_app_bio", "") if existing_t else "",
                                          placeholder="粘贴她的 Hinge/Bumble/Tinder Bio...",
                                          height=100)

            notes = st.text_area("其他备注",
                                 value=existing_t.get("additional_notes", "") if existing_t else "",
                                 placeholder="你对她的整体感觉、特别注意到的细节...")

            btn_label = "💾 更新资料" if existing_t else "➕ 创建对象"
            submitted = st.form_submit_button(btn_label, type="primary", use_container_width=True)
            if submitted:
                if not t_name:
                    st.error("姓名不能为空")
                else:
                    data = {
                        "id": target_id if existing_t else None,
                        "name": t_name,
                        "age": int(t_age),
                        "instagram_bio": ig_bio,
                        "instagram_posts_description": ig_posts,
                        "linkedin_info": linkedin,
                        "facebook_info": facebook,
                        "photo_description": photos,
                        "dating_app_bio": dating_bio,
                        "additional_notes": notes,
                    }
                    new_id = save_target(data)
                    st.success(f"✅ {'更新' if existing_t else '创建'}成功！ID: {new_id}")
                    st.rerun()

    # ── Tab: 上传文件 ──
    with tab_files:
        target_id = st.session_state.get("current_target_id")
        if not target_id:
            st.warning("请先在侧栏选择一个对象，或在「编辑 / 新建」创建一个。")
        else:
            target_info = load_target(target_id)
            st.subheader(f"上传 {target_info['name']} 的资料文件")
            st.caption("支持：PDF、Word、图片（JPG/PNG）、文本文件")

            platform = st.selectbox(
                "文件来源平台",
                ["Instagram", "LinkedIn", "Facebook", "Dating App", "其他"]
            )

            uploaded_files = st.file_uploader(
                "选择文件（可多选）",
                type=["pdf", "docx", "doc", "jpg", "jpeg", "png", "webp", "txt", "md"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                if st.button("📤 上传并分析", type="primary"):
                    if not client:
                        st.error("需要 API Key 才能分析图片")
                    else:
                        progress = st.progress(0)
                        for i, f in enumerate(uploaded_files):
                            with st.spinner(f"正在处理 {f.name}..."):
                                file_bytes = f.read()
                                extracted = extract_text(f.name, file_bytes, client)
                                filetype = get_filetype_label(f.name)
                                save_file(target_id, platform, f.name,
                                          file_bytes, filetype, extracted)
                            progress.progress((i + 1) / len(uploaded_files))
                        st.success(f"✅ {len(uploaded_files)} 个文件上传完成！")
                        st.rerun()

            # Show existing files
            st.divider()
            st.subheader("已上传的文件")
            existing_files = load_files_for_target(target_id)
            if not existing_files:
                st.info("还没有上传文件")
            else:
                for f in existing_files:
                    with st.expander(f"**{f['filename']}**  [{f['platform']}]  {f['filetype']}"):
                        if f.get("extracted_text"):
                            st.text_area(
                                "提取的内容",
                                value=f["extracted_text"][:2000],
                                height=150,
                                key=f"text_{f['id']}",
                                disabled=True,
                            )
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.caption(f"上传时间：{f['uploaded_at'][:19]}")
                        with col2:
                            if st.button("删除", key=f"delf_{f['id']}"):
                                delete_file(f["id"])
                                st.rerun()


# ─── Page: 人格分析 ────────────────────────────────────────────────────────────

elif page == "🧠 人格分析":
    st.title("🧠 人格分析")

    target_id = st.session_state.get("current_target_id")
    if not target_id:
        st.warning("请先在侧栏选择一个对象")
        st.stop()

    target_info = load_target(target_id)
    files = load_files_for_target(target_id)
    existing_profile = load_latest_personality(target_id)

    st.subheader(f"分析对象：{target_info['name']}")

    col1, col2 = st.columns([2, 1])
    with col1:
        if existing_profile:
            st.success(f"已有人格分析结果（可重新分析以更新）")
        else:
            st.info("还没有分析过，点击下方按钮开始")

    with col2:
        run_analysis = st.button("🔍 运行人格分析", type="primary", use_container_width=True)

    if run_analysis:
        if not client:
            st.error("需要配置 DEEPSEEK_API_KEY")
            st.stop()

        # Build social profile from stored data + uploaded files
        # Append extracted file texts to corresponding fields
        file_texts = {"Instagram": [], "LinkedIn": [], "Facebook": [], "Dating App": [], "其他": []}
        for f in files:
            platform = f.get("platform", "其他")
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
            additional_notes=merge(target_info.get("additional_notes", ""), file_texts["其他"]),
        )

        with st.spinner("DeepSeek R1 正在深度分析（可能需要 30-60 秒）..."):
            profile = infer_personality(social, client)

        save_personality(target_id, profile)
        existing_profile = profile
        st.success("✅ 分析完成！")
        st.rerun()

    # Display profile
    if existing_profile:
        st.divider()

        # Big Five radar chart using plotly
        try:
            import plotly.graph_objects as go
            traits = ["开放性", "尽责性", "外向性", "宜人性", "神经质"]
            values = [
                existing_profile.openness,
                existing_profile.conscientiousness,
                existing_profile.extraversion,
                existing_profile.agreeableness,
                existing_profile.neuroticism,
            ]
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=traits + [traits[0]],
                fill="toself",
                line_color="#FF4B8B",
                fillcolor="rgba(255, 75, 139, 0.2)",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # Scores table
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("开放性 Openness", f"{existing_profile.openness:.1f}/10")
            st.metric("尽责性 Conscientiousness", f"{existing_profile.conscientiousness:.1f}/10")
        with col2:
            st.metric("外向性 Extraversion", f"{existing_profile.extraversion:.1f}/10")
            st.metric("宜人性 Agreeableness", f"{existing_profile.agreeableness:.1f}/10")
        with col3:
            st.metric("神经质 Neuroticism", f"{existing_profile.neuroticism:.1f}/10")
            st.metric("依恋风格", existing_profile.attachment_style)

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**真实兴趣（推断）**")
            for i in existing_profile.true_interests:
                st.write(f"• {i}")

            st.markdown("**核心价值观**")
            for v in existing_profile.core_values:
                st.write(f"• {v}")

            st.markdown(f"**爱的语言：** {existing_profile.love_language}")
            st.markdown(f"**沟通风格：** {existing_profile.communication_style}")

        with col_r:
            st.markdown("**她真正在寻找什么**")
            st.info(existing_profile.relationship_goals)

            st.markdown("**雷区 / 冲突触发点**")
            for t in existing_profile.conflict_triggers:
                st.error(f"⚠️ {t}")

        st.divider()
        st.markdown("**性格总结**")
        st.success(existing_profile.personality_summary)

        with st.expander("🔍 分析推理过程"):
            st.write(existing_profile.analysis_reasoning)


# ─── Page: 约会模拟 ────────────────────────────────────────────────────────────

elif page == "💬 约会模拟":
    st.title("💬 约会模拟")

    target_id = st.session_state.get("current_target_id")
    if not target_id:
        st.warning("请先在侧栏选择一个对象")
        st.stop()

    user = load_user_profile()
    if not user:
        st.warning("请先填写「我的资料」")
        st.stop()

    her = load_latest_personality(target_id)
    if not her:
        st.warning("请先在「人格分析」页面运行分析")
        st.stop()

    target_info = load_target(target_id)
    past_sessions = load_date_sessions(target_id)
    date_number = len(past_sessions) + 1

    st.subheader(f"第 {date_number} 次约会：{user.name} × {her.name}")

    # Previous date context
    if past_sessions:
        last = past_sessions[-1]
        st.info(f"上次约会（第{last['date_number']}次）结果：化学反应 {last['chemistry_score']}/10，她的兴趣 {last['her_interest_level']}/10")

    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input(
            "约会地点",
            placeholder="例：SoHo的屋顶酒吧、当代艺术博物馆、Central Park",
        )
    with col2:
        activity = st.text_input(
            "约会活动",
            placeholder="例：傍晚喝酒、下午看展、周日野餐",
        )

    special_mode = st.selectbox(
        "特殊模式",
        ["普通约会", "模拟婚后生活"],
    )

    num_exchanges = st.slider("对话轮数", min_value=6, max_value=20, value=10)

    start_btn = st.button("▶ 开始模拟", type="primary", use_container_width=True,
                          disabled=not (location and activity and client))

    if not location or not activity:
        st.caption("请填写地点和活动后开始")

    if start_btn:
        if special_mode == "模拟婚后生活":
            actual_date_number = 999
            scenario = DateScenario(
                location="你们共同的家",
                activity="婚后第三年的一个普通周日早晨",
            )
        else:
            actual_date_number = date_number
            scenario = DateScenario(location=location, activity=activity)

        prev_result = None
        if past_sessions:
            last = past_sessions[-1]
            from src.models import DateResult
            prev_result = DateResult(
                date_number=last["date_number"],
                summary=last["summary"],
                chemistry_score=last["chemistry_score"],
                her_interest_level=last["her_interest_level"],
                your_performance_score=last["your_performance_score"],
                next_date_probability=last["next_date_probability"],
                her_feedback=last["her_feedback"],
            )

        # Live conversation display
        st.divider()
        st.markdown(f"**📍 {scenario.location}  ·  {scenario.activity}**")
        st.divider()

        conversation_container = st.container()
        messages_placeholder = []

        def stream_to_ui(speaker, text):
            if speaker == her.name:
                messages_placeholder.append(("her", speaker, text))
            else:
                messages_placeholder.append(("user", speaker, text))
            with conversation_container:
                for role, spk, txt in messages_placeholder:
                    if role == "her":
                        st.chat_message("assistant", avatar="💁‍♀️").write(f"**{spk}:** {txt}")
                    else:
                        st.chat_message("user", avatar="🧑").write(f"**{spk}:** {txt}")

        with st.spinner("正在模拟约会..."):
            result = simulate_date(
                her=her,
                user=user,
                scenario=scenario,
                client=client,
                date_number=actual_date_number,
                num_exchanges=num_exchanges,
                previous_date_result=prev_result,
                stream_callback=stream_to_ui,
            )

        save_date_session(target_id, scenario, result)

        # Results
        st.divider()
        st.subheader("📊 约会结果")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("化学反应", f"{result.chemistry_score:.1f}/10")
        col2.metric("她的兴趣", f"{result.her_interest_level:.1f}/10")
        col3.metric("你的表现", f"{result.your_performance_score:.1f}/10")
        col4.metric("下次约会概率", f"{result.next_date_probability * 100:.0f}%")

        st.info(result.summary)

        col_l, col_r = st.columns(2)
        with col_l:
            if result.best_moments:
                st.markdown("**✨ 最佳时刻**")
                for m in result.best_moments:
                    st.success(m)
        with col_r:
            if result.awkward_moments:
                st.markdown("**😬 尴尬时刻**")
                for m in result.awkward_moments:
                    st.error(m)

        st.markdown(f"**💭 {her.name} 约会后的内心独白**")
        st.warning(result.her_feedback)

        if result.advice_for_next_time:
            st.markdown("**💡 给你的建议**")
            for i, tip in enumerate(result.advice_for_next_time, 1):
                st.write(f"{i}. {tip}")


# ─── Page: 历史记录 ────────────────────────────────────────────────────────────

elif page == "📋 历史记录":
    st.title("📋 历史记录")

    target_id = st.session_state.get("current_target_id")
    if not target_id:
        st.warning("请先在侧栏选择一个对象")
        st.stop()

    target_info = load_target(target_id)
    sessions = load_date_sessions(target_id)

    st.subheader(f"{target_info['name']} 的约会记录")

    if not sessions:
        st.info("还没有约会记录")
        st.stop()

    # Summary trend chart
    try:
        import plotly.graph_objects as go
        dates = [f"第{s['date_number']}次" for s in sessions]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=[s["chemistry_score"] for s in sessions],
                                  name="化学反应", line=dict(color="#FF4B8B")))
        fig.add_trace(go.Scatter(x=dates, y=[s["her_interest_level"] for s in sessions],
                                  name="她的兴趣", line=dict(color="#FF8C00")))
        fig.add_trace(go.Scatter(x=dates, y=[s["your_performance_score"] for s in sessions],
                                  name="你的表现", line=dict(color="#4B8BFF")))
        fig.update_layout(
            yaxis=dict(range=[0, 10]),
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    st.divider()

    for s in reversed(sessions):
        label = "婚后生活" if s["date_number"] == 999 else f"第 {s['date_number']} 次约会"
        with st.expander(f"**{label}**  ·  {s['location']}  ·  化学反应 {s['chemistry_score']}/10"):
            col1, col2, col3 = st.columns(3)
            col1.metric("化学反应", f"{s['chemistry_score']:.1f}/10")
            col2.metric("她的兴趣", f"{s['her_interest_level']:.1f}/10")
            col3.metric("下次约会概率", f"{s['next_date_probability']*100:.0f}%")

            st.write(s["summary"])

            if s.get("her_feedback"):
                st.markdown("**她的内心独白：**")
                st.warning(s["her_feedback"])

            with st.expander("查看完整对话记录"):
                if s.get("full_conversation"):
                    st.text(s["full_conversation"])
                else:
                    st.info("无对话记录")

            st.caption(f"模拟时间：{s['created_at'][:19]}")

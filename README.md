# 💘 WhereIsMyLove 
> An AI-powered dating simulation world — analyze personalities, simulate dates, and discover compatibility before the first real date.


---

## What It Does

WhereIsMyLove lets you build a psychological profile of someone you're interested in, then simulate dating them using AI agents — all before you ever ask them out.

You provide social media data (Instagram bio, LinkedIn, dating app profile, photos, etc.) and the app infers their personality using the Big Five model, then runs a fully simulated date between two AI agents: one playing them, one playing you. After several simulated dates, the app analyzes your compatibility and tells you whether to go for it.

### Core Features

| Feature | Description |
|---|---|
| **Personality Inference** | Upload social media profiles and the AI infers Big Five traits, attachment style, love language, communication style, relationship goals, and hidden needs |
| **Date Simulation** | Two DeepSeek agents role-play a date — one as the target, one as you — with natural back-and-forth conversation |
| **Date Evaluation** | After each date, an evaluator AI scores chemistry, interest level, your performance, and probability of a next date, with a full narrative report |
| **Auto Mode** | AI picks the perfect date scenario based on both personalities, then runs the full simulation with one click |
| **Compatibility Report** | After multiple dates, the app analyzes all sessions and gives an overall compatibility score, pros/cons, long-term prediction, and a "should you ask them out?" verdict |
| **History & Archive** | All simulation sessions are saved and browsable; auto-simulations are archived separately with compatibility decisions |
| **Your Profile** | Build a profile of yourself (personality, interests, communication style) for more accurate simulations |

---

## System Design

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (SPA)                       │
│              Single HTML file — no framework             │
│   Pages: My Profile · Their Profile · Personality ·     │
│          Date Sim · History                              │
└────────────────────────┬────────────────────────────────┘
                         │ REST API (fetch)
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                          │
│                                                          │
│  /api/user             — user profile CRUD               │
│  /api/targets          — target profiles CRUD            │
│  /api/targets/:id/personality  — run inference           │
│  /api/targets/:id/simulate     — run manual sim          │
│  /api/targets/:id/auto-simulate — run auto sim           │
│  /api/targets/:id/compatibility — compat report          │
│  /api/tasks/:id        — poll long-running jobs          │
└──────┬──────────────────────┬───────────────────────────┘
       │                      │
┌──────▼──────┐      ┌────────▼────────────────────────────┐
│  SQLite DB  │      │         DeepSeek API                 │
│             │      │                                      │
│ user_profile│      │  deepseek-reasoner (R1)              │
│ targets     │      │  → Personality inference             │
│ personality │      │                                      │
│ date_sessions│     │  deepseek-chat (V3)                  │
│ user_files  │      │  → Date conversation agents          │
│ user_files  │      │  → Date evaluation                   │
└─────────────┘      │  → Scenario generation               │
                     │  → Compatibility analysis            │
                     └──────────────────────────────────────┘
```

### Key Modules

**`src/personality_inference.py`**
Uses DeepSeek R1 (reasoning model) to infer a full personality profile from social media data. Produces Big Five scores, attachment style, love language, conflict triggers, relationship goals, and a deep narrative analysis.

**`src/date_simulator.py`**
Runs a two-agent conversation simulation:
- Agent HER: embodies the target's personality with full context
- Agent YOU: plays the user based on their profile
- Each agent takes turns responding (1–4 sentences, like real conversation)
- A separate evaluator agent scores the date and produces a full narrative report

**`src/storage.py`**
SQLite persistence layer. Handles user profiles, target profiles, uploaded files, personality profiles (with language versioning), date sessions, and compatibility reports.

**`src/file_processor.py`**
Extracts text from uploaded files (PDF, DOCX, images via OCR) to feed into personality inference.

**`server.py`**
FastAPI app. Long-running AI tasks (personality inference, date simulation) run as background jobs polled by the frontend via `/api/tasks/:id`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JS + Tailwind CSS (single HTML file) |
| Backend | Python / FastAPI |
| AI | DeepSeek R1 (reasoning) + DeepSeek V3 (chat) |
| Database | SQLite |
| File parsing | PyPDF, python-docx, Pillow |

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/ZhuohanYU/WHOISLOVE-.git
cd WHOISLOVE-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your DeepSeek API key

```bash
cp .env.example .env
# Edit .env and add: DEEPSEEK_API_KEY=your_key_here
```

Get a key at [platform.deepseek.com](https://platform.deepseek.com)

### 3. Run

```bash
uvicorn server:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## How to Use

1. **My Profile** — Fill in your own info (name, age, personality, interests). The more detail, the more accurate the simulation.
2. **Their Profile** — Add the person you're interested in. Fill in any social media info you have.
3. **Personality** — Click "Analyze Personality" to let the AI infer who they really are.
4. **Date Sim** — Run a manual date (you pick the location/activity) or use Auto mode to let AI generate the perfect scenario.
5. **History** — Browse all past simulated dates and review compatibility reports.

---

## Project Structure

```
WHOISLOVE/
├── server.py              # FastAPI app + all API routes
├── frontend/
│   └── index.html         # Entire frontend (SPA)
├── src/
│   ├── models.py           # Pydantic data models
│   ├── personality_inference.py  # AI personality analysis
│   ├── date_simulator.py   # Two-agent date simulation
│   ├── storage.py          # SQLite persistence layer
│   └── file_processor.py   # PDF/DOCX/image text extraction
├── requirements.txt
└── .env.example
```

---

*Built with DeepSeek AI + FastAPI + a bit of romantic optimism.*

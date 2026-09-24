# -*- coding: utf-8 -*-
"""
SQLite 持久化模块。
负责：评估结果的读写、按内容缓存查询、历史记录查询。
"""
import sqlite3
import json
import hashlib
import time
from contextlib import contextmanager

DB_PATH = "resume_screener.db"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                jd_hash TEXT NOT NULL,
                jd_text TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                resume_hash TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                score INTEGER,
                dimension_scores TEXT,
                strengths TEXT,
                risks TEXT,
                interview_questions TEXT,
                extracted_profile TEXT,
                candidate_graduation_year INTEGER,
                graduation_match INTEGER,
                email TEXT,
                status TEXT NOT NULL,
                error TEXT
            )
        """)
        # 兼容旧数据库文件：如果是从旧版本升级上来的表，补上新增列
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(evaluations)")}
        if "candidate_graduation_year" not in existing_cols:
            conn.execute("ALTER TABLE evaluations ADD COLUMN candidate_graduation_year INTEGER")
        if "graduation_match" not in existing_cols:
            conn.execute("ALTER TABLE evaluations ADD COLUMN graduation_match INTEGER")
        if "email" not in existing_cols:
            conn.execute("ALTER TABLE evaluations ADD COLUMN email TEXT")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup
            ON evaluations (jd_hash, resume_hash, model, prompt_version)
        """)
        # HR 反馈表：记录人工对AI评分的认可/修正，是未来构建评估集、验证Prompt改动效果的基础数据
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                evaluation_id INTEGER NOT NULL,
                verdict TEXT NOT NULL,
                corrected_score INTEGER,
                comment TEXT,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
            )
        """)
        # 邮件发送记录表：记录每次通过邮件中心发出的邮件，便于追溯发送状态
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                candidate_name TEXT NOT NULL,
                recipient_email TEXT,
                email_type TEXT,
                subject TEXT,
                body TEXT,
                status TEXT NOT NULL
            )
        """)


def compute_hashes(jd_text: str, resume_text: str, extra_scope: str = ""):
    """extra_scope 用于让相同 JD+简历在不同筛选条件（如目标毕业年份）下不互相误用缓存。"""
    return _hash(jd_text + "|" + extra_scope), _hash(resume_text)


def find_cached(jd_hash: str, resume_hash: str, model: str, prompt_version: str):
    """按内容查找是否已经评估过（且成功），命中则直接复用，不再调用 API。"""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM evaluations
            WHERE jd_hash=? AND resume_hash=? AND model=? AND prompt_version=? AND status='ok'
            ORDER BY created_at DESC LIMIT 1
        """, (jd_hash, resume_hash, model, prompt_version)).fetchone()
        return dict(row) if row else None


def save_result(*, jd_hash, jd_text, candidate_name, resume_hash, resume_text,
                 model, prompt_version, status, score=None, dimension_scores=None,
                 strengths=None, risks=None, interview_questions=None,
                 extracted_profile=None, candidate_graduation_year=None,
                 graduation_match=None, email=None, error=None) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO evaluations (
                created_at, jd_hash, jd_text, candidate_name, resume_hash, resume_text,
                model, prompt_version, score, dimension_scores, strengths, risks,
                interview_questions, extracted_profile, candidate_graduation_year,
                graduation_match, email, status, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            time.time(), jd_hash, jd_text, candidate_name, resume_hash, resume_text,
            model, prompt_version, score,
            json.dumps(dimension_scores, ensure_ascii=False) if dimension_scores else None,
            json.dumps(strengths, ensure_ascii=False) if strengths else None,
            json.dumps(risks, ensure_ascii=False) if risks else None,
            json.dumps(interview_questions, ensure_ascii=False) if interview_questions else None,
            json.dumps(extracted_profile, ensure_ascii=False) if extracted_profile else None,
            candidate_graduation_year,
            None if graduation_match is None else int(bool(graduation_match)),
            email, status, error,
        ))
        return cur.lastrowid


def row_to_result(row: dict) -> dict:
    """把数据库行转换成前端展示用的 dict 结构。"""
    return {
        "候选人": row["candidate_name"],
        "score": row["score"],
        "email": row.get("email"),
        "dimension_scores": json.loads(row["dimension_scores"]) if row["dimension_scores"] else {},
        "strengths": json.loads(row["strengths"]) if row["strengths"] else [],
        "risks": json.loads(row["risks"]) if row["risks"] else [],
        "interview_questions": json.loads(row["interview_questions"]) if row["interview_questions"] else [],
        "extracted_profile": json.loads(row["extracted_profile"]) if row["extracted_profile"] else {},
        "candidate_graduation_year": row["candidate_graduation_year"] if "candidate_graduation_year" in row.keys() else None,
        "graduation_match": bool(row["graduation_match"]) if row["graduation_match"] is not None else None,
        "_resume_text": row["resume_text"],
        "_eval_id": row["id"],
        "error": row["error"] if row["status"] != "ok" else None,
        "_from_cache": True,
    }


def list_history(limit: int = 50):
    """查询最近的评估历史，用于「历史记录」页签。"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT created_at, candidate_name, score, model, status
            FROM evaluations ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def save_feedback(evaluation_id: int, verdict: str, corrected_score: int = None, comment: str = None):
    """verdict: 'agree' 或 'disagree'。积累人工判断，用于未来评估 Prompt/模型效果。"""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO feedback (created_at, evaluation_id, verdict, corrected_score, comment)
            VALUES (?,?,?,?,?)
        """, (time.time(), evaluation_id, verdict, corrected_score, comment))


def feedback_stats():
    """统计已积累的反馈：总数、认可率，用于「反馈概览」展示。"""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
        agree = conn.execute("SELECT COUNT(*) c FROM feedback WHERE verdict='agree'").fetchone()["c"]
        return {"total": total, "agree": agree, "disagree": total - agree}


def list_candidates(limit: int = 200):
    """供邮件中心使用：返回最近评估成功、且按姓名去重（取最新一次）的候选人完整信息。

    strengths / risks / extracted_profile 已解析为 Python 对象。
    """
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, candidate_name, score, email, resume_text, strengths, risks, extracted_profile
            FROM evaluations WHERE status='ok' ORDER BY created_at DESC
        """).fetchall()
    seen = {}
    for r in rows:
        d = dict(r)
        name = d["candidate_name"]
        if name not in seen:
            d["strengths"] = json.loads(d["strengths"]) if d["strengths"] else []
            d["risks"] = json.loads(d["risks"]) if d["risks"] else []
            d["extracted_profile"] = json.loads(d["extracted_profile"]) if d["extracted_profile"] else {}
            seen[name] = d
    return list(seen.values())[:limit]


def save_email_log(*, candidate_name, recipient_email=None, email_type=None,
                   subject=None, body=None, status="sent"):
    """记录一封邮件的发送结果（成功/失败均记录，便于追溯）。"""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO email_log (created_at, candidate_name, recipient_email, email_type, subject, body, status)
            VALUES (?,?,?,?,?,?,?)
        """, (time.time(), candidate_name, recipient_email, email_type, subject, body, status))


def list_email_log(limit: int = 50):
    """查询最近的邮件发送记录，用于「邮件中心」展示。"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT created_at, candidate_name, recipient_email, email_type, status
            FROM email_log ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
"""QAAgent tests — Claude mocked, documents on tmp_path, runs offline."""
from agents.qa_agent import QAAgent

GOOD_RESUME = """# Rudra Joshi
Graduate AI Engineer

## Skills
Python, LLMs, RAG, Prompt Engineering, APIs, NLP

## Projects
PersonaQuery — RAG-based document Q&A system built with Python and LLM APIs.
HealthEcho — health-data NLP pipeline.

## Education
Bachelor of Computer Science, University of Wollongong (2025)
"""

GOOD_COVER_LETTER = """Dear Hiring Manager,

I am applying for the Graduate AI Engineer role at Acme AI. My project PersonaQuery
demonstrates hands-on experience building RAG systems in Python, directly matching
your requirements. HealthEcho shows my NLP background.

I would welcome the opportunity to contribute to your team.

Sincerely,
Rudra Joshi
"""


def _make_docs(tmp_path):
    resume = tmp_path / "resume.md"
    cl = tmp_path / "cover_letter.md"
    resume.write_text(GOOD_RESUME, encoding="utf-8")
    cl.write_text(GOOD_COVER_LETTER, encoding="utf-8")
    return str(resume), str(cl)


def _job():
    return {
        "title": "Graduate AI Engineer",
        "company": "Acme AI",
        "job_hash": "test123",
        "required_skills": ["Python", "LLMs", "RAG"],
    }


def test_run_qa_returns_report(monkeypatch, tmp_path):
    agent = QAAgent()
    # hallucination check returns "no issues"
    monkeypatch.setattr(agent.client, "generate_json", lambda **kw: [])
    resume_path, cl_path = _make_docs(tmp_path)

    report = agent.run_qa(resume_path, cl_path, _job(), auto_fix=False)

    assert report.checks_run == 14
    assert 0 <= report.checks_passed <= report.checks_run
    assert report.recommendation in {"approve", "reject", "fix_and_retry"}


def test_run_qa_llm_failure_does_not_crash(monkeypatch, tmp_path):
    agent = QAAgent()

    def boom(**kw):
        raise RuntimeError("api down")

    monkeypatch.setattr(agent.client, "generate_json", boom)
    resume_path, cl_path = _make_docs(tmp_path)

    # QA should degrade, not raise — application pipeline must survive LLM outages
    try:
        report = agent.run_qa(resume_path, cl_path, _job(), auto_fix=False)
        assert report.checks_run == 14
    except RuntimeError:
        # Documented current behavior if QA propagates the error; fail loudly so
        # a future refactor makes this resilient.
        raise

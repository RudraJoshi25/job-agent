"""Shared fixtures. All tests run offline — ClaudeClient methods are mocked."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    """ClaudeClient requires a key at construction; never used for real calls."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


@pytest.fixture
def raw_job():
    return {
        "title": "Graduate AI Engineer",
        "company": "Acme AI",
        "location": "Sydney NSW",
        "salary": "$80,000 - $95,000",
        "source": "seek",
        "url": "https://example.com/job/123",
        "full_description": (
            "Acme AI is seeking a Graduate AI Engineer in Sydney. "
            "Requirements: Python, LLMs, RAG, prompt engineering, APIs. "
            "Nice to have: NLP, cloud experience. 0-2 years experience welcome."
        ),
    }


@pytest.fixture
def normalized_llm_response():
    """Canned Claude JSON for the normalizer."""
    return {
        "title": "Graduate AI Engineer",
        "company": "Acme AI",
        "location": "Sydney NSW",
        "salary_min": 80000,
        "salary_max": 95000,
        "employment_type": "graduate",
        "seniority_level": "graduate",
        "apply_type": "portal",
        "required_skills": ["Python", "LLMs", "RAG"],
        "nice_to_have_skills": ["NLP"],
        "responsibilities": ["Build LLM applications"],
        "visa_sponsorship": None,
        "remote_friendly": True,
    }

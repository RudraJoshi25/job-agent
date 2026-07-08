"""Pipeline data-flow test: normalizer output feeds the matcher. Offline, Claude mocked."""
from agents.match_agent import MatchAgent, MatchResult
from agents.normalizer_agent import NormalizerAgent


def test_normalized_job_flows_into_matcher(monkeypatch, raw_job, normalized_llm_response):
    normalizer = NormalizerAgent()
    monkeypatch.setattr(normalizer.client, "generate_json", lambda **kw: normalized_llm_response)
    normalized = normalizer._normalize_single_job(raw_job)

    matcher = MatchAgent()
    canned = MatchResult(
        score=90,
        verdict="maybe",
        matching_skills=["Python", "LLMs"],
        missing_skills=[],
        reasons="canned",
    )
    monkeypatch.setattr(matcher.client, "generate_structured", lambda **kw: canned)

    jd = matcher._format_job_for_matching(normalized)
    result = matcher.evaluate_match(jd, normalized)

    assert result.band in {"PRIORITY", "STRETCH", "SKIP"}
    assert result.score >= 90  # tier bonus can only add
    assert normalized["job_hash"]


def test_route_jobs_buckets(monkeypatch, normalized_llm_response):
    matcher = MatchAgent()
    scores = iter([matcher.priority_score + 5, max(0, matcher.min_score - 10)])

    def fake(**kw):
        return MatchResult(
            score=next(scores),
            verdict="maybe",
            matching_skills=[],
            missing_skills=[],
            reasons="canned",
        )

    monkeypatch.setattr(matcher.client, "generate_structured", fake)
    jobs = [dict(normalized_llm_response, title="AI Engineer A", location_tier=2),
            dict(normalized_llm_response, title="AI Engineer B", location_tier=2)]
    bands = matcher.route_jobs(jobs)

    assert len(bands["PRIORITY"]) == 1
    assert len(bands["SKIP"]) == 1
    assert not bands["STRETCH"]

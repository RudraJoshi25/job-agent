import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from services.claude_client import ClaudeClient
from core.prompt_safety import wrap_untrusted
from pydantic import BaseModel, Field
import yaml

_MANAGEMENT_KEYWORDS = frozenset([
    'manager', 'director', 'vp', 'head of', 'lead',
    'principal', 'staff',
])


def load_profile_yaml(profile_path: str = "profile.yaml") -> Dict[str, Any]:
    """Load profile.yaml as single source of truth."""
    with open(profile_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class MatchResult(BaseModel):
    score: float = Field(..., ge=0, le=100, description="Overall match score from 0-100")
    verdict: str = Field(..., description="apply, maybe, or skip")
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    reasons: str = Field(..., description="Detailed reasoning for the verdict")
    band: str = Field(default="SKIP", description="Routing band: PRIORITY, STRETCH, or SKIP")


class MatchAgent:
    def __init__(self, profile: Dict[str, Any] = None, claude_client: ClaudeClient = None):
        self.profile = profile or load_profile_yaml()
        self.candidate_profile = self._build_candidate_profile()
        self.client = claude_client or ClaudeClient(model="claude-haiku-4-5-20251001")

        matching_config = self.profile.get('matching', {})
        self.min_score = matching_config.get('min_score', 50)
        self.priority_score = matching_config.get('priority_score', 65)
        self.tier1_bonus = matching_config.get('location_tiers', {}).get('tier1_bonus', 10)
        self.exclude_keywords = matching_config.get('exclude_keywords', [])

    def _build_candidate_profile(self) -> Dict[str, Any]:
        """Build candidate profile dict from profile.yaml structure."""
        candidate = self.profile.get('candidate', {})
        skills = self.profile.get('skills', {})
        projects = self.profile.get('projects', [])

        all_skills = []
        for category in skills.values():
            if isinstance(category, list):
                all_skills.extend(category)

        return {
            'name': candidate.get('name', ''),
            'location': candidate.get('location', ''),
            'remote_preference': 'Hybrid/Remote',
            'target_roles': self.profile.get('search', {}).get('job_titles', []),
            'experience_level': candidate.get('experience_band', '0-2 years'),
            'preferred_seniority': [candidate.get('classification', 'Graduate / Junior')],
            'education': [{
                'degree': candidate.get('degree', ''),
                'institution': candidate.get('university', ''),
                'graduation_year': candidate.get('graduation', '')
            }],
            'skills': all_skills,
            'key_strengths': skills.get('ai_ml', [])[:5],
            'projects': projects
        }

    def evaluate_match(self, job_description: str, job: Dict[str, Any] = None) -> MatchResult:
        """Evaluate job match with tier1 bonus and exclude keyword filtering."""
        job_title = job.get('title', '') if job else ''
        location_tier = job.get('location_tier', 2) if job else 2

        if self._should_exclude(job_title):
            return MatchResult(
                score=0,
                verdict="skip",
                matching_skills=[],
                missing_skills=[],
                reasons=f"Title contains excluded keyword",
                band="SKIP"
            )

        prompt = self._build_evaluation_prompt(job_description)
        system_prompt = self._build_system_prompt()

        result = self.client.generate_structured(
            prompt=prompt,
            response_model=MatchResult,
            system=system_prompt,
            max_tokens=2048
        )

        final_score = result.score
        if location_tier == 1:
            final_score = min(100, result.score + self.tier1_bonus)

        if final_score >= self.priority_score:
            result.verdict = "apply"
            result.band = "PRIORITY"
        elif final_score >= self.min_score:
            result.verdict = "maybe"
            result.band = "STRETCH"
        else:
            result.verdict = "skip"
            result.band = "SKIP"

        result.score = final_score
        return result

    def _should_exclude(self, job_title: str) -> bool:
        """Check if job title contains any exclude keywords."""
        title_lower = job_title.lower()
        for keyword in self.exclude_keywords:
            kw_lower = keyword.lower()
            if kw_lower == 'senior':
                if 'senior' in title_lower and any(mgmt in title_lower for mgmt in _MANAGEMENT_KEYWORDS):
                    return True
            else:
                if kw_lower in title_lower:
                    return True
        return False

    def route_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Route jobs into PRIORITY, STRETCH, SKIP bands and print summary."""
        bands = {'PRIORITY': [], 'STRETCH': [], 'SKIP': []}

        for job in jobs:
            job_desc = self._format_job_for_matching(job)
            result = self.evaluate_match(job_desc, job)

            job['match_result'] = result.model_dump()
            bands[result.band].append(job)

        print("\n" + "=" * 60)
        print("ROUTING SUMMARY")
        print("=" * 60)
        print(f"  PRIORITY (>={self.priority_score}): {len(bands['PRIORITY'])} jobs")
        print(f"  STRETCH  ({self.min_score}-{self.priority_score-1}): {len(bands['STRETCH'])} jobs")
        print(f"  SKIP     (<{self.min_score}): {len(bands['SKIP'])} jobs")
        print("=" * 60)

        return bands

    def _format_job_for_matching(self, job: Dict[str, Any]) -> str:
        """Format job dict for matching evaluation."""
        parts = [
            f"Job Title: {job.get('title', 'N/A')}",
            f"Company: {job.get('company', 'N/A')}",
            f"Location: {job.get('location', 'N/A')}",
        ]

        if job.get('required_skills'):
            parts.append(f"Required Skills: {', '.join(job['required_skills'])}")

        desc = job.get('raw_description', job.get('description', job.get('short_description', '')))
        if desc:
            parts.append(f"\nJob Description:\n{desc}")

        return '\n'.join(parts)

    def _build_system_prompt(self) -> str:
        return """You are an expert career advisor and recruiter specializing in AI/ML roles.
Your task is to evaluate how well a candidate matches a job posting.

Scoring guidelines:
- 90-100: Exceptional match, candidate meets all requirements and exceeds expectations
- 75-89: Strong match, candidate meets most key requirements
- 50-74: Moderate match, candidate has some relevant skills but gaps exist
- 25-49: Weak match, significant skill gaps or misalignment
- 0-24: Poor match, fundamentally different role or requirements

Consider:
1. Skills alignment (technical skills, tools, frameworks)
2. Experience level match (junior/senior alignment)
3. Role type match (job title and responsibilities)
4. Domain expertise match (AI/ML vs other domains)
5. Education requirements

Be honest and realistic. Don't inflate scores. Consider the candidate's constraints around honest representation."""

    def _build_evaluation_prompt(self, job_description: str) -> str:
        profile_summary = self._format_profile()

        return f"""Evaluate how well this candidate matches the job posting.

CANDIDATE PROFILE:
{profile_summary}

JOB POSTING:
{wrap_untrusted(job_description)}

Provide:
1. A score from 0-100 indicating match quality
2. List of matching_skills (skills the candidate has that the job requires)
3. List of missing_skills (required skills the candidate lacks)
4. Detailed reasons explaining your assessment

Be critical but fair. Consider experience level appropriateness."""

    def _format_profile(self) -> str:
        profile = self.candidate_profile

        skills_all = []
        if isinstance(profile.get("skills"), dict):
            for category, skill_list in profile["skills"].items():
                skills_all.extend(skill_list)
        elif isinstance(profile.get("skills"), list):
            skills_all = profile["skills"]

        education_str = ""
        if profile.get("education"):
            edu = profile["education"][0] if isinstance(profile["education"], list) else profile["education"]
            education_str = f"{edu.get('degree', 'N/A')} - {edu.get('institution', 'N/A')} ({edu.get('graduation_year', 'N/A')})"

        return f"""Name: {profile.get('name', 'N/A')}
Location: {profile.get('location', 'N/A')} - {profile.get('remote_preference', 'N/A')}
Target Roles: {', '.join(profile.get('target_roles', []))}
Experience Level: {profile.get('experience_level', 'Graduate/Entry-level')}
Preferred Seniority: {', '.join(profile.get('preferred_seniority', []))}
Education: {education_str}
Skills: {', '.join(skills_all)}
Key Strengths: {', '.join(profile.get('key_strengths', []))}"""


def match_job(job_description: str, profile: Dict[str, Any] = None, job: Dict[str, Any] = None) -> MatchResult:
    """Convenience function to match a single job."""
    agent = MatchAgent(profile=profile)
    return agent.evaluate_match(job_description, job)

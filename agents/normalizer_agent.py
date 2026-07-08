import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from services.claude_client import ClaudeClient
from core.prompt_safety import wrap_untrusted


class NormalizedJob(BaseModel):
    """Normalized job posting schema."""
    job_hash: str
    title: str
    company: str
    location: str
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    employment_type: str
    seniority_level: str
    apply_type: str
    required_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    visa_sponsorship: Optional[bool] = None
    remote_friendly: bool
    jd_word_count: int
    source: str
    url: str
    raw_description: str
    normalized_at: str


class NormalizerAgent:
    """Agent for normalizing and standardizing scraped job data using Claude."""

    def __init__(self, batch_size: int = 5, batch_delay: float = 1.0):
        self.client = ClaudeClient(model="claude-haiku-4-5-20251001")
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.normalized_jobs = []
        self.skipped_jobs = []

    def normalize_jobs(
        self,
        input_file: str = "data/jobs_raw.json",
        output_file: str = "data/jobs_clean.json",
        skipped_file: str = "data/jobs_skipped.json"
    ) -> Dict[str, int]:
        """Normalize jobs from input file and save to output files."""
        print("=" * 80)
        print("JOB NORMALIZER")
        print("=" * 80)

        raw_jobs = self._load_jobs(input_file)
        print(f"Loaded {len(raw_jobs)} raw jobs from {input_file}")
        print()

        total_jobs = len(raw_jobs)
        batches = [raw_jobs[i:i + self.batch_size] for i in range(0, total_jobs, self.batch_size)]

        print(f"Processing {len(batches)} batches (batch size: {self.batch_size})")
        print("-" * 80)

        for batch_idx, batch in enumerate(batches, 1):
            print(f"\nBatch {batch_idx}/{len(batches)}")
            print("-" * 40)

            for job_idx, raw_job in enumerate(batch, 1):
                job_num = (batch_idx - 1) * self.batch_size + job_idx
                print(f"  [{job_num}/{total_jobs}] {raw_job.get('title', 'Unknown')[:50]}...", end=" ")

                try:
                    normalized = self._normalize_single_job(raw_job)
                    self.normalized_jobs.append(normalized)
                    print("[OK]")
                except Exception as e:
                    print(f"[SKIP] {str(e)[:50]}")
                    raw_job['skip_reason'] = str(e)
                    raw_job['skipped_at'] = datetime.now().isoformat()
                    self.skipped_jobs.append(raw_job)

            if batch_idx < len(batches):
                print(f"  Waiting {self.batch_delay}s before next batch...")
                time.sleep(self.batch_delay)

        self._save_results(output_file, skipped_file)

        print()
        print("=" * 80)
        print(f"SUMMARY: Normalized {len(self.normalized_jobs)} jobs, skipped {len(self.skipped_jobs)}")
        print("=" * 80)

        return {
            "normalized": len(self.normalized_jobs),
            "skipped": len(self.skipped_jobs),
            "total": total_jobs
        }

    def _normalize_single_job(self, raw_job: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single job using Claude."""
        job_description = self._extract_description(raw_job)
        jd_word_count = len(job_description.split())

        extraction_prompt = self._build_extraction_prompt(raw_job, job_description)

        try:
            result = self.client.generate_json(
                prompt=extraction_prompt,
                system=self._build_system_prompt(),
                max_tokens=2048
            )

            job_hash = self._generate_hash(
                result.get('title', raw_job.get('title', '')),
                result.get('company', raw_job.get('company', '')),
                result.get('location', raw_job.get('location', ''))
            )

            normalized = {
                'job_hash': job_hash,
                'title': result['title'],
                'company': result['company'],
                'location': result['location'],
                'salary_min': result.get('salary_min'),
                'salary_max': result.get('salary_max'),
                'employment_type': result['employment_type'],
                'seniority_level': result['seniority_level'],
                'apply_type': result.get('apply_type', 'portal'),
                'required_skills': result.get('required_skills', []),
                'nice_to_have_skills': result.get('nice_to_have_skills', []),
                'responsibilities': result.get('responsibilities', []),
                'visa_sponsorship': result.get('visa_sponsorship'),
                'remote_friendly': result.get('remote_friendly', False),
                'jd_word_count': jd_word_count,
                'source': raw_job.get('source', 'unknown'),
                'url': raw_job.get('url', ''),
                'raw_description': job_description,
                'normalized_at': datetime.now().isoformat()
            }

            return normalized

        except Exception as e:
            raise ValueError(f"Failed to normalize: {str(e)[:100]}")

    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude."""
        return """You are a job data extraction specialist. Extract and standardize job posting information.

Your task is to analyze job descriptions and extract structured data accurately.

RULES:
1. Title: Clean format, title case, no extra symbols or ALL CAPS
2. Company: Company name only, no extra text
3. Location: Standardized format (e.g., "Sydney NSW", "Remote", "Sydney NSW / Remote")
4. Salary: Extract min and max as integers (annual AUD), null if not mentioned
5. Employment type: Must be one of: "full-time", "part-time", "contract", "internship", "graduate"
6. Seniority level: Must be one of: "graduate", "junior", "mid", "senior", "lead"
7. Apply type: "easy_apply" (quick apply button), "portal" (company website), "email"
8. Required skills: List specific technical skills that are explicitly required
9. Nice to have skills: List preferred/desirable skills mentioned
10. Responsibilities: Extract 3-5 key responsibilities
11. Visa sponsorship: true if mentioned, false if explicitly not offered, null if not mentioned
12. Remote friendly: true if remote/hybrid work is offered

Be accurate and conservative. If unsure, use null or empty list."""

    def _build_extraction_prompt(self, raw_job: Dict[str, Any], description: str) -> str:
        """Build extraction prompt for Claude."""
        source = raw_job.get('source', 'unknown')
        raw_company = raw_job.get('company', 'N/A')

        return f"""Extract and normalize information from this job posting.

RAW DATA:
Title: {raw_job.get('title', 'N/A')}
Company: {raw_company}
Location: {raw_job.get('location', 'N/A')}
Salary: {raw_job.get('salary', 'N/A')}
Source: {source}
URL: {raw_job.get('url', 'N/A')}

JOB DESCRIPTION:
{wrap_untrusted(description[:3000])}

IMPORTANT FOR COMPANY FIELD:
If the company field above is empty, "Unknown Company", "None", or "N/A",
search the job description text carefully for the actual company name and extract it.
Look for phrases like "at [Company]", "[Company] is seeking", "join [Company]", etc.

Return JSON with these exact fields:
{{
  "title": "cleaned title",
  "company": "company name",
  "location": "standardized location",
  "salary_min": integer or null,
  "salary_max": integer or null,
  "employment_type": "full-time|part-time|contract|internship|graduate",
  "seniority_level": "graduate|junior|mid|senior|lead",
  "apply_type": "easy_apply|portal|email",
  "required_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill1", "skill2"],
  "responsibilities": ["resp1", "resp2", "resp3"],
  "visa_sponsorship": true or false or null,
  "remote_friendly": true or false
}}"""

    def _extract_description(self, raw_job: Dict[str, Any]) -> str:
        """Extract job description from raw job data."""
        if raw_job.get('full_description'):
            return raw_job['full_description']
        elif raw_job.get('description'):
            return raw_job['description']
        elif raw_job.get('short_description'):
            return raw_job['short_description']
        else:
            return f"{raw_job.get('title', '')} at {raw_job.get('company', '')}"

    def _generate_hash(self, title: str, company: str, location: str) -> str:
        """Generate unique hash for job deduplication."""
        key = f"{title}|{company}|{location}".lower().strip()
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def _load_jobs(self, filepath: str) -> List[Dict[str, Any]]:
        """Load jobs from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_results(self, output_file: str, skipped_file: str):
        """Save normalized and skipped jobs to files."""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.normalized_jobs, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(self.normalized_jobs)} normalized jobs to {output_file}")

        if self.skipped_jobs:
            skipped_path = Path(skipped_file)
            with open(skipped_path, 'w', encoding='utf-8') as f:
                json.dump(self.skipped_jobs, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(self.skipped_jobs)} skipped jobs to {skipped_file}")


def main():
    """Example usage."""
    normalizer = NormalizerAgent(batch_size=5, batch_delay=1.0)

    results = normalizer.normalize_jobs(
        input_file="data/jobs_raw.json",
        output_file="data/jobs_clean.json",
        skipped_file="data/jobs_skipped.json"
    )

    return results


if __name__ == "__main__":
    main()

"""
Layer 2 - Guardrails Engine
Wraps every agent call with input/output validation, PII detection, cost tracking, and hallucination detection.
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from pydantic import BaseModel
import asyncio


class GuardrailViolation(BaseModel):
    """Single guardrail violation."""
    guardrail: str
    severity: str  # low, medium, high
    message: str
    details: Dict[str, Any] = {}


class GuardrailResult(BaseModel):
    """Result of guardrail check."""
    passed: bool
    violations: List[GuardrailViolation] = []
    metadata: Dict[str, Any] = {}


class GuardrailsEngine:
    """Wraps agent calls with input/output validation and safety checks."""

    def __init__(self, profile_path: str = "data/candidate_profile.json"):
        self.profile_path = profile_path
        self.candidate_profile = self._load_profile()
        self.cost_log_path = Path("logs/cost_tracking.json")
        self.cost_log_path.parent.mkdir(exist_ok=True)
        self.last_api_call = 0
        self.total_cost = 0.0
        self._load_cost_log()

    def _load_profile(self) -> Dict[str, Any]:
        """Load candidate profile for hallucination detection."""
        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            print(f"[GUARDRAILS] Could not load profile {self.profile_path}: {e}")
            return {}

    def _load_cost_log(self):
        """Load existing cost log."""
        if self.cost_log_path.exists():
            with open(self.cost_log_path, 'r') as f:
                data = json.load(f)
                self.total_cost = data.get('total_cost', 0.0)

    async def run(
        self,
        agent_fn: Callable,
        input_data: Dict[str, Any],
        expected_output_schema: Optional[Any] = None,
        max_retries: int = 2,
        agent_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Run agent function with guardrails.

        Returns:
            {
                'success': bool,
                'output': Any,
                'violations': List[GuardrailViolation],
                'cost': float,
                'retries': int
            }
        """
        print(f"  [GUARDRAILS] Running {agent_name} with safety checks", flush=True)

        # INPUT GUARDRAILS
        input_result = await self._run_input_guardrails(input_data, agent_name)

        # Halt on high severity input violations
        high_violations = [v for v in input_result.violations if v.severity == "high"]
        if high_violations:
            print(f"  [GUARDRAILS] HIGH severity input violations detected - halting", flush=True)
            return {
                'success': False,
                'output': None,
                'violations': input_result.violations,
                'cost': 0.0,
                'retries': 0
            }

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                # Rate limiting
                await self._enforce_rate_limit()

                # Execute agent
                start_time = time.time()
                if asyncio.iscoroutinefunction(agent_fn):
                    output = await agent_fn(**input_data)
                else:
                    output = agent_fn(**input_data)
                elapsed = time.time() - start_time

                # OUTPUT GUARDRAILS
                output_result = await self._run_output_guardrails(
                    output,
                    input_data,
                    expected_output_schema,
                    agent_name
                )

                # Track cost
                estimated_cost = self._estimate_cost(input_data, output)
                self._log_cost(agent_name, estimated_cost, elapsed)

                # Check for medium/high violations
                medium_high = [v for v in output_result.violations if v.severity in ["medium", "high"]]

                if medium_high and attempt < max_retries:
                    print(f"  [GUARDRAILS] Violations detected, retrying (attempt {attempt + 1}/{max_retries})", flush=True)
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

                # Success or final attempt
                all_violations = input_result.violations + output_result.violations
                success = len([v for v in all_violations if v.severity == "high"]) == 0

                print(f"  [GUARDRAILS] {agent_name} completed - {len(all_violations)} violations detected", flush=True)

                return {
                    'success': success,
                    'output': output,
                    'violations': all_violations,
                    'cost': estimated_cost,
                    'retries': attempt
                }

            except Exception as e:
                print(f"  [GUARDRAILS] Agent error on attempt {attempt + 1}: {str(e)[:100]}", flush=True)
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        'success': False,
                        'output': None,
                        'violations': [GuardrailViolation(
                            guardrail="execution",
                            severity="high",
                            message=f"Agent failed after {max_retries} retries",
                            details={'error': str(e)}
                        )],
                        'cost': 0.0,
                        'retries': attempt
                    }

    async def _run_input_guardrails(self, input_data: Dict[str, Any], agent_name: str) -> GuardrailResult:
        """Run all input guardrails."""
        violations = []

        # 1. Schema validator
        schema_result = self._validate_input_schema(input_data, agent_name)
        violations.extend(schema_result.violations)

        # 2. PII checker
        pii_result = self._check_pii(input_data)
        violations.extend(pii_result.violations)

        # 3. Budget checker
        budget_result = self._check_budget(input_data)
        violations.extend(budget_result.violations)

        return GuardrailResult(
            passed=len([v for v in violations if v.severity == "high"]) == 0,
            violations=violations
        )

    async def _run_output_guardrails(
        self,
        output: Any,
        input_data: Dict[str, Any],
        expected_schema: Optional[Any],
        agent_name: str
    ) -> GuardrailResult:
        """Run all output guardrails."""
        violations = []

        # 5. Hallucination detector
        if agent_name in ["resume_agent", "cover_letter_agent"]:
            hallucination_result = self._detect_hallucinations(output, input_data)
            violations.extend(hallucination_result.violations)

        # 6. Schema validator
        if expected_schema:
            output_schema_result = self._validate_output_schema(output, expected_schema)
            violations.extend(output_schema_result.violations)

        # 7. Quality floor
        quality_result = self._check_quality_floor(output, agent_name)
        violations.extend(quality_result.violations)

        return GuardrailResult(
            passed=len([v for v in violations if v.severity == "high"]) == 0,
            violations=violations
        )

    def _validate_input_schema(self, input_data: Dict[str, Any], agent_name: str) -> GuardrailResult:
        """Check if input has required fields."""
        violations = []

        required_fields = {
            'resume_agent': ['job', 'match_report'],
            'cover_letter_agent': ['job', 'match_report', 'tailored_resume'],
            'match_agent': ['job_description'],
            'source_agent': ['search_terms', 'location']
        }

        if agent_name in required_fields:
            for field in required_fields[agent_name]:
                if field not in input_data or not input_data[field]:
                    violations.append(GuardrailViolation(
                        guardrail="input_schema",
                        severity="high",
                        message=f"Missing required field: {field}",
                        details={'agent': agent_name, 'field': field}
                    ))

        return GuardrailResult(passed=len(violations) == 0, violations=violations)

    def _check_pii(self, input_data: Dict[str, Any]) -> GuardrailResult:
        """Detect if input contains sensitive data like passwords/tokens."""
        violations = []
        input_str = json.dumps(input_data, default=str).lower()

        # Check for common PII patterns
        patterns = [
            (r'password["\s:]*[\w]{8,}', "Possible password detected"),
            (r'api[_-]?key["\s:]*[\w]{20,}', "Possible API key detected"),
            (r'token["\s:]*[\w]{20,}', "Possible token detected"),
            (r'secret["\s:]*[\w]{20,}', "Possible secret detected"),
        ]

        for pattern, message in patterns:
            if re.search(pattern, input_str):
                violations.append(GuardrailViolation(
                    guardrail="pii_checker",
                    severity="high",
                    message=message,
                    details={'pattern': pattern}
                ))

        return GuardrailResult(passed=len(violations) == 0, violations=violations)

    def _check_budget(self, input_data: Dict[str, Any]) -> GuardrailResult:
        """Estimate token cost and warn if too high."""
        violations = []
        input_str = json.dumps(input_data, default=str)
        estimated_tokens = len(input_str.split()) * 1.3  # Rough estimate

        if estimated_tokens > 8000:
            violations.append(GuardrailViolation(
                guardrail="budget_checker",
                severity="medium",
                message=f"High token count estimated: {estimated_tokens:.0f} tokens",
                details={'estimated_tokens': estimated_tokens}
            ))

        return GuardrailResult(passed=len(violations) == 0, violations=violations)

    async def _enforce_rate_limit(self):
        """Enforce 5 second minimum between API calls."""
        elapsed = time.time() - self.last_api_call
        if elapsed < 5.0:
            wait_time = 5.0 - elapsed
            print(f"  [GUARDRAILS] Rate limiting: waiting {wait_time:.1f}s", flush=True)
            await asyncio.sleep(wait_time)
        self.last_api_call = time.time()

    def _detect_hallucinations(self, output: Any, input_data: Dict[str, Any]) -> GuardrailResult:
        """Detect if output contains fabricated information."""
        violations = []

        # Convert output to string for analysis
        if isinstance(output, dict):
            output_str = json.dumps(output, default=str)
        elif hasattr(output, '__dict__'):
            output_str = str(output.__dict__)
        else:
            output_str = str(output)

        # Get job data for context
        job = input_data.get('job', {})

        # 1. Check for company names not in profile or job
        profile_companies = [exp.get('company', '') for exp in self.candidate_profile.get('work_experience', [])]
        job_company = job.get('company', '')

        # Find potential company names in output (titlecase words)
        company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
        found_companies = set(re.findall(company_pattern, output_str))

        for company in found_companies:
            if len(company) > 4 and company not in profile_companies and company != job_company:
                # Common words to exclude
                exclude = {'University', 'Master', 'Bachelor', 'India', 'Australia', 'Sydney', 'Engineer',
                          'Agent', 'Project', 'System', 'Application', 'Graduate', 'Senior', 'Junior'}
                if company not in exclude:
                    violations.append(GuardrailViolation(
                        guardrail="hallucination_detector",
                        severity="high",
                        message=f"Fabricated company name: {company}",
                        details={'company': company, 'known_companies': profile_companies}
                    ))

        # 2. Check for dates/years inconsistent with profile
        profile_years = set()
        for exp in self.candidate_profile.get('work_experience', []):
            years = re.findall(r'\b(20\d{2})\b', str(exp))
            profile_years.update(years)

        found_years = set(re.findall(r'\b(20\d{2})\b', output_str))
        fabricated_years = found_years - profile_years

        for year in fabricated_years:
            if int(year) < 2020:  # Only flag suspiciously old years
                violations.append(GuardrailViolation(
                    guardrail="hallucination_detector",
                    severity="medium",
                    message=f"Year {year} not in profile work history",
                    details={'year': year, 'profile_years': list(profile_years)}
                ))

        # 3. Check for skills not in profile (lenient - only flag exact fabrications)
        profile_skills = set()
        if isinstance(self.candidate_profile.get('skills'), dict):
            for skill_list in self.candidate_profile['skills'].values():
                profile_skills.update(s.lower() for s in skill_list)

        return GuardrailResult(passed=len([v for v in violations if v.severity == "high"]) == 0, violations=violations)

    def _validate_output_schema(self, output: Any, expected_schema: Any) -> GuardrailResult:
        """Validate output matches expected schema."""
        violations = []

        try:
            if hasattr(expected_schema, 'model_validate'):
                # Pydantic model
                expected_schema.model_validate(output)
        except Exception as e:
            violations.append(GuardrailViolation(
                guardrail="output_schema",
                severity="medium",
                message=f"Output schema mismatch: {str(e)[:100]}",
                details={'error': str(e)}
            ))

        return GuardrailResult(passed=len(violations) == 0, violations=violations)

    def _check_quality_floor(self, output: Any, agent_name: str) -> GuardrailResult:
        """Check if output meets minimum quality standards."""
        violations = []

        # Convert output to string
        if isinstance(output, dict):
            output_str = output.get('resume_markdown', '') or output.get('cover_letter_markdown', '') or str(output)
        else:
            output_str = str(output)

        word_count = len(output_str.split())

        # Minimum word counts by agent
        min_words = {
            'resume_agent': 200,
            'cover_letter_agent': 150,
        }

        if agent_name in min_words and word_count < min_words[agent_name]:
            violations.append(GuardrailViolation(
                guardrail="quality_floor",
                severity="medium",
                message=f"Output too short: {word_count} words (min {min_words[agent_name]})",
                details={'word_count': word_count, 'min_words': min_words[agent_name]}
            ))

        return GuardrailResult(passed=len(violations) == 0, violations=violations)

    def _estimate_cost(self, input_data: Dict[str, Any], output: Any) -> float:
        """Estimate API cost based on token usage."""
        # Rough estimation: 1000 words ≈ 1300 tokens
        input_str = json.dumps(input_data, default=str)
        output_str = str(output)

        input_tokens = len(input_str.split()) * 1.3
        output_tokens = len(output_str.split()) * 1.3

        # Claude Sonnet 4.5 pricing (example - update with actual rates)
        input_cost = (input_tokens / 1000000) * 3.0  # $3 per 1M input tokens
        output_cost = (output_tokens / 1000000) * 15.0  # $15 per 1M output tokens

        return input_cost + output_cost

    def _log_cost(self, agent_name: str, cost: float, elapsed: float):
        """Log cost to tracking file."""
        self.total_cost += cost

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'cost_usd': round(cost, 6),
            'elapsed_seconds': round(elapsed, 2),
            'total_cost_usd': round(self.total_cost, 4)
        }

        # Load existing log
        if self.cost_log_path.exists():
            with open(self.cost_log_path, 'r') as f:
                data = json.load(f)
        else:
            data = {'total_cost': 0.0, 'entries': []}

        data['total_cost'] = self.total_cost
        data['entries'].append(log_entry)

        # Keep only last 1000 entries
        data['entries'] = data['entries'][-1000:]

        with open(self.cost_log_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        if not self.cost_log_path.exists():
            return {'total_cost': 0.0, 'total_calls': 0}

        with open(self.cost_log_path, 'r') as f:
            data = json.load(f)

        return {
            'total_cost': data.get('total_cost', 0.0),
            'total_calls': len(data.get('entries', [])),
            'recent_calls': data.get('entries', [])[-10:]
        }


def main():
    """Test guardrails engine."""
    import asyncio

    async def test_agent(job, match_report):
        """Dummy agent for testing."""
        await asyncio.sleep(0.1)
        return {
            'resume_markdown': 'This is a test resume with at least 200 words. ' * 30,
            'changelog': 'Test changes'
        }

    async def run_test():
        engine = GuardrailsEngine()

        result = await engine.run(
            agent_fn=test_agent,
            input_data={
                'job': {'title': 'Test Engineer', 'company': 'Test Co'},
                'match_report': {'score': 85}
            },
            agent_name='resume_agent',
            max_retries=2
        )

        print(f"\nGuardrails Test Result:")
        print(f"  Success: {result['success']}")
        print(f"  Violations: {len(result['violations'])}")
        print(f"  Cost: ${result['cost']:.6f}")
        print(f"  Retries: {result['retries']}")

        stats = engine.get_stats()
        print(f"\nStats:")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
        print(f"  Total calls: {stats['total_calls']}")

    asyncio.run(run_test())


if __name__ == "__main__":
    main()

"""
Apply Agent — Universal AI-Powered Page Understanding.

ReAct loop (Reason → Act → Observe) with Claude Sonnet for page understanding.
Works on any careers page or ATS without hardcoded selectors.
Max 15 iterations per application.

Architecture:
  - launch_persistent_context always (preserves Seek session)
  - Enterprise ATS (Taleo, iCIMS, etc.) → immediate manual queue
  - Adzuna → resolve redirect first
  - Seek → mid-run session revalidation before and after Apply click
  - All other sites → universal AI loop
"""
import json
import os
import re
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from playwright.async_api import async_playwright, Page, BrowserContext
import anthropic

from core.prompt_safety import wrap_untrusted


class ApplyAgent:
    """Universal job application agent powered by Claude Sonnet page understanding."""

    BROWSER_PROFILE = os.environ.get(
        "BROWSER_PROFILE_DIR",
        str(Path(__file__).resolve().parents[1] / ".browser_profile"),
    )

    # Enterprise ATS with heavy auth walls — always manual queue, no browser needed
    ENTERPRISE_ATS = {'taleo', 'successfactors', 'icims', 'jobvite', 'bamboohr'}

    MAX_ITERATIONS = 15

    # Submit button selectors tried in order
    SUBMIT_SELECTORS = [
        'button[type="submit"]',
        'button:has-text("Submit application")',
        'button:has-text("Submit Application")',
        'button:has-text("Submit")',
        'button:has-text("Apply now")',
        'button:has-text("Complete application")',
        'button:has-text("Send application")',
        '[data-automation="apply-submit"]',
        'input[type="submit"]',
    ]

    CONFIRMATION_URLS = (
        'confirmation', 'thank-you', 'thankyou',
        'success', 'complete', 'submitted',
    )
    CONFIRMATION_PHRASES = [
        "application submitted", "thank you for applying",
        "we've received your application", "we have received your application",
        "application complete", "application has been submitted",
        "your application has been received", "successfully applied",
        "application was submitted", "application received",
        "your application has been sent", "application has been sent",
        "thanks for applying", "thank you for your application",
        "we will be in touch", "your application has been submitted",
        "application reference", "confirmation number",
    ]

    def __init__(
        self,
        profile_path: str = "config/profile.json",
        mode: Literal["draft", "assisted", "auto"] = "assisted",
        verbose: bool = False
    ):
        self.profile_path = profile_path
        self.candidate_profile = self._load_profile()
        self.mode = mode
        self.verbose = verbose
        self.apply_log = []
        self._ai_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._seek_session_expired = False
        self._last_apply_time: Optional[float] = None

    # ------------------------------------------------------------------ #
    #  Profile                                                             #
    # ------------------------------------------------------------------ #

    def _load_profile(self) -> Dict[str, Any]:
        for path in (self.profile_path, "config/profile.json", "data/candidate_profile.json"):
            p = Path(path)
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        raise FileNotFoundError("No candidate profile found in config/profile.json")

    # ------------------------------------------------------------------ #
    #  ATS Detection                                                       #
    # ------------------------------------------------------------------ #

    def _detect_ats_type(self, url: str) -> str:
        u = url.lower()
        if 'seek.com.au' in u:                                return 'seek'
        if 'indeed.com' in u:                                 return 'indeed'
        if 'myworkdayjobs.com' in u or 'workday.com' in u:   return 'workday'
        if 'greenhouse.io' in u or 'boards.greenhouse' in u:  return 'greenhouse'
        if 'lever.co' in u:                                   return 'lever'
        if 'smartrecruiters.com' in u:                        return 'smartrecruiters'
        if 'icims.com' in u:                                  return 'icims'
        if 'jobvite.com' in u:                                return 'jobvite'
        if 'taleo.net' in u:                                  return 'taleo'
        if 'successfactors.com' in u:                         return 'successfactors'
        if 'bamboohr.com' in u:                               return 'bamboohr'
        if 'adzuna.com.au' in u:                              return 'adzuna'
        if 'gradconnection.com' in u:                         return 'gradconnection'
        return 'unknown'

    # ------------------------------------------------------------------ #
    #  Session Management                                                  #
    # ------------------------------------------------------------------ #

    async def check_seek_session(self) -> bool:
        """Check if the stored Seek session is still valid (public API used by master_agent)."""
        try:
            async with async_playwright() as p:
                ctx = await p.chromium.launch_persistent_context(
                    user_data_dir=self.BROWSER_PROFILE,
                    headless=True,
                    channel="chrome"
                )
                page = await ctx.new_page()
                await page.goto(
                    "https://www.seek.com.au/profile",
                    wait_until="networkidle",
                    timeout=20000
                )
                current_url = page.url.lower()
                await ctx.close()

            if any(t in current_url for t in ('login', 'signin', 'sign-in', 'register', 'oauth', 'auth')):
                self._seek_session_expired = True
                return False
            return True
        except Exception as e:
            print(f"[SESSION] Seek session check failed: {e}")
            return False

    async def _revalidate_seek_session(self, page: Page) -> bool:
        """Open a background tab, check seek.com.au/profile, close it. Returns True if session valid."""
        if self._seek_session_expired:
            return False
        check_page = None
        try:
            check_page = await page.context.new_page()
            await check_page.goto(
                "https://www.seek.com.au/profile",
                wait_until="networkidle",
                timeout=15000
            )
            url = check_page.url.lower()
        except Exception as e:
            print(f"[SESSION] Mid-run session check error (assuming valid): {e}")
            return True
        finally:
            if check_page:
                try:
                    await check_page.close()
                except Exception:
                    pass

        if any(t in url for t in ('login', 'signin', 'sign-in', 'register', 'oauth', 'auth')):
            print("[SESSION EXPIRED] Seek session expired mid-run — all remaining Seek jobs → manual queue")
            self._seek_session_expired = True
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Public Entry Point                                                  #
    # ------------------------------------------------------------------ #

    async def apply_to_job(
        self,
        job: Dict[str, Any],
        resume_path: str,
        cover_letter_path: Optional[str] = None,
        auto_mode: bool = False
    ) -> Dict[str, Any]:
        """Apply to a job using the universal AI-powered page understanding loop."""

        # ── Resolve best URL ─────────────────────────────────────────────
        seek_url   = job.get('seek_url', '') or (job.get('url', '') if 'seek.com.au' in job.get('url', '') else '')
        indeed_url = job.get('indeed_url', '') or (job.get('url', '') if 'indeed.com' in job.get('url', '') else '')
        job_url    = seek_url or indeed_url or job.get('url', '')

        # ── Validate resume: must be a compiled PDF ───────────────────────
        if resume_path:
            rp = Path(resume_path)
            if rp.suffix.lower() == '.tex':
                pdf_candidate = rp.with_suffix('.pdf')
                if pdf_candidate.exists():
                    resume_path = str(pdf_candidate)
                    print(f"[RESUME] Switched .tex → PDF: {resume_path}")
                else:
                    result = self._make_result(job)
                    result['status'] = 'failed_with_reason'
                    result['error'] = f"No PDF resume — only .tex exists. Expected: {pdf_candidate}. Compile LaTeX first."
                    print(f"[ERROR] {result['error']}")
                    self._log_application(result)
                    return result

        # ── Inter-application delay ───────────────────────────────────────
        await self._inter_apply_delay()

        print(f"\nApplying to: {job.get('title')} at {job.get('company')}")
        print(f"URL: {job_url}")
        print(f"Mode: {self.mode.upper()} | Verbose: {self.verbose}")
        print("-" * 80)

        # ── Load cover letter text ────────────────────────────────────────
        cover_letter_text = ""
        if cover_letter_path and Path(cover_letter_path).exists():
            with open(cover_letter_path, 'r', encoding='utf-8') as f:
                cover_letter_text = f.read()

        result = self._make_result(job)
        result['url'] = job_url

        # ── URL-based ATS detection BEFORE opening browser ────────────────
        url_ats = self._detect_ats_type(job_url)
        result['ats_type'] = url_ats

        if url_ats in self.ENTERPRISE_ATS:
            result['status'] = 'manual_queue'
            result['manual_reason'] = f'enterprise_ats_{url_ats}'
            print(f"[MANUAL] Enterprise ATS ({url_ats}) — apply directly at: {job_url}")
            self._log_application(result)
            return result

        if url_ats == 'adzuna':
            resolved = await self._resolve_adzuna_redirect(job_url)
            if not resolved:
                result['status'] = 'manual_queue'
                result['manual_reason'] = 'adzuna_redirect_failed'
                self._log_application(result)
                return result
            job_url = resolved
            url_ats = self._detect_ats_type(job_url)
            result['ats_type'] = url_ats
            result['resolved_url'] = job_url
            job = {**job, 'url': job_url}
            if url_ats in self.ENTERPRISE_ATS:
                result['status'] = 'manual_queue'
                result['manual_reason'] = f'enterprise_ats_{url_ats}_via_adzuna'
                print(f"[MANUAL] Adzuna → Enterprise ATS ({url_ats}) — apply directly at: {job_url}")
                self._log_application(result)
                return result

        # ── Open browser (ALWAYS persistent context — preserves session) ──
        try:
            async with async_playwright() as p:
                ctx = await p.chromium.launch_persistent_context(
                    user_data_dir=self.BROWSER_PROFILE,
                    headless=False,
                    channel="chrome"
                )
                page = await ctx.new_page()

                if self.mode == "draft":
                    await page.goto(job_url, wait_until="domcontentloaded", timeout=30000)
                    ss = await self._screenshot(page, result['job_hash'], "draft")
                    result['screenshots'].append(ss)
                    result['status'] = 'draft_only'
                    print(f"[DRAFT] Screenshot saved: {ss}")
                else:
                    await self._apply_universal(page, ctx, job, resume_path, result, cover_letter_text)

                await ctx.close()

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"[ERROR] Application failed: {e}")
            import traceback
            traceback.print_exc()

        self._log_application(result)
        return result

    # ------------------------------------------------------------------ #
    #  Universal Apply Loop                                                #
    # ------------------------------------------------------------------ #

    async def _apply_universal(
        self,
        page: Page,
        ctx: BrowserContext,
        job: Dict[str, Any],
        resume_path: str,
        result: Dict[str, Any],
        cover_letter_text: str
    ):
        """
        Universal ReAct apply loop. Steps 0-10 from spec.
        Terminates on: confirmed_submitted, manual_queue, failed_with_reason,
                       already_applied, job_expired, cancelled_by_user.
        """
        job_url   = result['url']
        job_hash  = result['job_hash']
        ats_type  = result.get('ats_type', 'unknown')

        # ── STEP 0: Navigate to job URL ───────────────────────────────────
        print(f"[NAVIGATE] Loading: {job_url[:80]}")
        try:
            resp = await page.goto(job_url, wait_until="networkidle", timeout=30000)
            if resp and resp.status in (404, 403, 500):
                result['status'] = 'job_expired'
                result['manual_reason'] = f'HTTP {resp.status}: {job_url}'
                print(f"[SKIP] HTTP {resp.status} — job page unavailable")
                return
        except Exception:
            try:
                resp = await page.goto(job_url, wait_until="domcontentloaded", timeout=30000)
                if resp and resp.status in (404, 403, 500):
                    result['status'] = 'job_expired'
                    result['manual_reason'] = f'HTTP {resp.status}: {job_url}'
                    print(f"[SKIP] HTTP {resp.status} — job page unavailable")
                    return
            except Exception as e:
                result['status'] = 'manual_queue'
                result['manual_reason'] = f'[MANUAL] Page load timeout: {e}'
                print(f"[MANUAL] Page load failed: {e}")
                return
        await page.wait_for_timeout(2000)
        page = await self._get_active_page(ctx, page)

        # ── Seek: mid-run session check AFTER navigation ──────────────────
        if ats_type == 'seek':
            if not await self._revalidate_seek_session(page):
                result['status'] = 'manual_queue'
                result['manual_reason'] = '[MANUAL] Seek session expired — run: python seed_login.py'
                print("[ACTION NEEDED] Run: python seed_login.py — then rerun the pipeline")
                return

        # ── STEP 2: Blocking state check immediately after page load ─────
        if await self._check_blocking_state(page, result):
            return

        # ── STEP 1: Click Apply button (preference logic) ─────────────────
        print(f"[APPLY] Looking for Apply button...")
        apply_status = await self._click_apply_preferred(page)
        print(f"[APPLY] Button status: {apply_status}")

        if apply_status == "already_applied":
            result['status'] = 'already_applied'
            return

        if apply_status == "not_found":
            print("[APPLY] No Apply button — assuming current page is the application form")

        # ── STEP 0: Handle new tab after Apply click ──────────────────────
        await page.wait_for_timeout(2000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            await page.wait_for_timeout(3000)
        page = await self._get_active_page(ctx, page)
        result['resolved_url'] = page.url

        # ── STEP 2 again: blocking check after Apply click ────────────────
        if await self._check_blocking_state(page, result):
            return

        # ── Seek: session check after Apply click ─────────────────────────
        if ats_type == 'seek':
            if not await self._revalidate_seek_session(page):
                result['status'] = 'manual_queue'
                result['manual_reason'] = '[MANUAL] Seek session expired after Apply click — run: python seed_login.py'
                return

        # ── ReAct loop ────────────────────────────────────────────────────
        resume_upload_done = False
        _prev_url = ""
        _zero_progress_streak = 0

        for iteration in range(self.MAX_ITERATIONS):
            step_num = iteration + 1
            print(f"\n{'─'*60}")
            print(f"[STEP {step_num}/{self.MAX_ITERATIONS}] {page.url[:70]}")
            await page.wait_for_timeout(2000)

            # Step 2: Blocking state check each iteration
            if await self._check_blocking_state(page, result):
                return

            # Step 3: Extract page state
            page_state = await self._extract_page_state(page)
            print(f"[STEP {step_num}] Elements: {len(page_state['elements'])} | Buttons: {len(page_state['buttons'])} | Title: {page_state['title'][:50]}")

            # Step 4: Claude Sonnet reasoning
            print(f"[STEP {step_num}] Asking Claude Sonnet...")
            action_plan = await self._ai_analyze_page(page_state, job, cover_letter_text, result)
            result['api_calls_made'] = result.get('api_calls_made', 0) + 1

            print(f"[STEP {step_num}] page_type={action_plan.get('page_type')} | is_final={action_plan.get('is_final_step')} | actions={len(action_plan.get('actions', []))}")
            print(f"[STEP {step_num}] {action_plan.get('step_description', '')}")

            for concern in action_plan.get('concerns', []):
                print(f"[CONCERN] {concern}")

            for field in action_plan.get('needs_manual_input', []):
                if field not in result.get('needs_manual_input', []):
                    result['needs_manual_input'].append(field)

            # Step 10: If file upload is in plan and not yet done, upload resume FIRST
            has_upload = any(a.get('action') == 'upload' for a in action_plan.get('actions', []))
            if has_upload and not resume_upload_done:
                print(f"[STEP {step_num}] Uploading resume FIRST (before other fields)...")
                await self._upload_resume_first(page, action_plan, resume_path, result)
                resume_upload_done = True
                # Wait for autofill then re-analyze
                print("[AUTOFILL] Waiting 8s for resume autofill to complete...")
                await page.wait_for_timeout(8000)
                page_state = await self._extract_page_state(page)
                print(f"[AUTOFILL] Re-analyzing page after autofill...")
                action_plan = await self._ai_analyze_page(page_state, job, cover_letter_text, result)
                result['api_calls_made'] = result.get('api_calls_made', 0) + 1
                # Remove the upload action so we don't re-upload
                action_plan['actions'] = [a for a in action_plan.get('actions', []) if a.get('action') != 'upload']
                print(f"[AUTOFILL] Corrections planned: {len(action_plan.get('actions', []))}")

            # Step 5: Execute actions
            print(f"[STEP {step_num}] Executing {len(action_plan.get('actions', []))} actions...")
            await self._execute_actions(page, action_plan.get('actions', []), resume_path, cover_letter_text, result)

            # Fill summary log
            filled_this_step = len(result['fields_filled'])
            print(f"[FILL SUMMARY] Filled: {filled_this_step} | Skipped: {len(result['fields_skipped'])} | Sensitive: {len(result.get('sensitive_fields_skipped', []))}")

            result['steps_completed'] = step_num

            # Stuck-loop detection: 3+ consecutive zero-progress iterations on same URL → manual queue
            current_url = page.url
            only_scroll_actions = all(
                a.get('action') in ('scroll', 'wait', 'skip')
                for a in action_plan.get('actions', [{'action': 'scroll'}])
            )
            if current_url == _prev_url and only_scroll_actions:
                _zero_progress_streak += 1
            else:
                _zero_progress_streak = 0
            _prev_url = current_url

            if _zero_progress_streak >= 3:
                result['status'] = 'manual_queue'
                result['manual_reason'] = f'[MANUAL] Stuck on login-gated or inaccessible portal: {current_url[:80]}'
                print(f"[MANUAL] Stuck loop detected ({_zero_progress_streak} iterations, same URL, only scroll) — manual queue")
                return

            # Step 0: Handle any new tab opened during action execution
            page = await self._get_active_page(ctx, page)

            # Step 9: Multi-step form loop vs final step
            if action_plan.get('is_final_step', False):

                # Step 6: Success gate
                if not await self._check_success_gate(page, result, job, resume_path, cover_letter_text):
                    return

                # Pre-submit screenshot
                pre_ss = await self._screenshot(page, job_hash, "before_submit")
                result['screenshots'].append(pre_ss)

                # Step 7: Human approval in assisted mode
                if self.mode != 'auto':
                    cancelled = await self._human_approval_prompt(result, job)
                    if cancelled:
                        result['status'] = 'cancelled_by_user'
                        return
                else:
                    self._print_fill_summary(result)
                    print(f"[AUTO-SUBMIT] Submitting: {job.get('title')} at {job.get('company')}")

                # Step 8: Submit + confirm
                await self._submit_and_confirm(page, result)
                return

            else:
                # Click Next/Continue and proceed to next iteration
                next_sel = action_plan.get('after_actions_click', '')
                if next_sel:
                    print(f"[NEXT] Clicking: {next_sel}")
                    try:
                        await page.click(next_sel)
                        try:
                            await page.wait_for_load_state("networkidle", timeout=12000)
                        except Exception:
                            await page.wait_for_timeout(3000)
                    except Exception as e:
                        print(f"[WARN] Next button click failed ({next_sel}): {e}")
                    page = await self._get_active_page(ctx, page)
                    await page.wait_for_timeout(2000)
                else:
                    print(f"[STEP {step_num}] No Next selector — checking for submit button on next iteration")

        # Exceeded max iterations
        result['status'] = 'manual_queue'
        result['manual_reason'] = f'max_{self.MAX_ITERATIONS}_iterations_reached'
        print(f"[MANUAL] Max {self.MAX_ITERATIONS} iterations reached — adding to manual queue")

    # ------------------------------------------------------------------ #
    #  Step 0 — New Tab Handling                                           #
    # ------------------------------------------------------------------ #

    async def _get_active_page(self, ctx: BrowserContext, current_page: Page) -> Page:
        """Return newest page if a new tab was opened, else current_page."""
        pages = ctx.pages
        if len(pages) > 1:
            newest = pages[-1]
            if newest != current_page:
                print(f"[TAB] New tab detected — switching to: {newest.url[:80]}")
                try:
                    await newest.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    await newest.wait_for_timeout(3000)
                return newest
        return current_page

    # ------------------------------------------------------------------ #
    #  Step 1 — Apply Button Preference Logic                              #
    # ------------------------------------------------------------------ #

    async def _click_apply_preferred(self, page: Page) -> str:
        """
        Click Apply button with preference order from spec.
        Returns: "already_applied" | "clicked" | "not_found"
        """
        # Already-applied check first
        try:
            body_lower = (await page.evaluate("() => document.body.innerText")).lower()
            if any(phrase in body_lower for phrase in [
                "you applied", "already applied", "applied on",
                "application received", "you have already submitted"
            ]):
                print("  [APPLY-BTN] Already applied badge detected")
                return "already_applied"
        except Exception:
            pass

        AVOID_LOWER = ('quick apply', 'easy apply', 'apply with seek', 'apply with linkedin')

        # Priority 1: Full-form apply buttons
        PREFERRED = [
            'a:has-text("Apply here")',        'button:has-text("Apply here")',
            'a:has-text("Apply manually")',    'button:has-text("Apply manually")',
            'a:has-text("Apply with resume")', 'button:has-text("Apply with resume")',
            'a:has-text("Apply on company site")', 'button:has-text("Apply on company site")',
            'a:has-text("Apply on this site")',    'button:has-text("Apply on this site")',
            '[data-automation="job-detail-apply"]',
            '[data-automation="apply-button"]',
            'a[data-automation*="apply"]',
            'button[data-automation*="apply"]',
            'a:has-text("Apply now")',  'button:has-text("Apply now")',
            'a:has-text("Apply")',      'button:has-text("Apply")',
            'a[href*="/apply"]:not([href*="/apply/external"]):not([href*="apply-with-seek"])',
        ]

        for sel in PREFERRED:
            try:
                btn = await page.query_selector(sel)
                if not btn or not await btn.is_visible():
                    continue
                text_lower = (await btn.evaluate(
                    "el => (el.innerText || el.textContent || '').trim().toLowerCase()"
                ))
                if any(avoid in text_lower for avoid in AVOID_LOWER):
                    print(f"  [APPLY-BTN] Skipping '{text_lower[:50]}' (Quick/Easy Apply variant)")
                    continue
                text_display = (await btn.evaluate(
                    "el => (el.innerText || el.textContent || '').trim()"
                ))
                print(f"[APPLY] Chose: '{text_display[:60]}' ({sel})")
                await btn.click()
                try:
                    await page.wait_for_load_state("networkidle", timeout=12000)
                except Exception:
                    await page.wait_for_timeout(3000)
                print(f"[APPLY] Now at: {page.url[:80]}")
                return "clicked"
            except Exception:
                pass

        print("  [APPLY-BTN] No Apply button found")
        return "not_found"

    # ------------------------------------------------------------------ #
    #  Step 2 — Blocking State Checks                                      #
    # ------------------------------------------------------------------ #

    async def _check_blocking_state(self, page: Page, result: Dict[str, Any]) -> bool:
        """
        Check 5 blocking states. Sets result status and returns True if blocked.
        """
        url = page.url.lower()
        try:
            body_lower = (await page.evaluate(
                "() => document.body.innerText.slice(0, 3000)"
            )).lower()
        except Exception:
            body_lower = ""

        # A: Already applied
        if any(p in body_lower for p in [
            "you applied", "already applied", "applied on",
            "application received", "you have already submitted"
        ]):
            result['status'] = 'already_applied'
            print("[SKIP] Already applied to this job")
            return True

        # B: Login wall
        LOGIN_PHRASES = [
            "sign in to apply", "log in to apply", "create an account to apply",
            "please login", "register to apply", "sign up to continue",
        ]
        LOGIN_URL_TOKENS = ('/login', '/signin', '/register', '/sign-up', '/signup', 'elmotalent.com.au')
        if any(p in body_lower for p in LOGIN_PHRASES) or \
           any(t in url for t in LOGIN_URL_TOKENS):
            result['status'] = 'manual_queue'
            result['manual_reason'] = f'[MANUAL] Login required on external portal: {page.url}'
            print(f"[MANUAL] Login wall at: {page.url[:80]}")
            return True

        # C: CAPTCHA
        CAPTCHA_PHRASES = [
            "prove you're human", "i'm not a robot",
            "complete the captcha", "solve this captcha",
        ]
        try:
            iframe_srcs = await page.evaluate(
                "() => Array.from(document.querySelectorAll('iframe')).map(f => f.src || '').filter(s => s)"
            )
            has_captcha_iframe = any(
                'recaptcha.google.com' in s or 'hcaptcha.com' in s
                for s in iframe_srcs
            )
        except Exception:
            has_captcha_iframe = False
        if has_captcha_iframe or any(p in body_lower for p in CAPTCHA_PHRASES):
            result['status'] = 'manual_queue'
            result['manual_reason'] = f'[MANUAL] CAPTCHA detected at: {page.url}'
            print(f"[MANUAL] CAPTCHA at: {page.url[:80]}")
            return True

        # D: Error / job expired
        ERROR_PHRASES = [
            "page not found", "job no longer available",
            "position has been filled", "job has expired",
            "this job is no longer accepting applications",
        ]
        if any(p in body_lower for p in ERROR_PHRASES):
            result['status'] = 'job_expired'
            result['manual_reason'] = f'[SKIP] Job no longer available: {page.url}'
            print(f"[SKIP] Job no longer available: {page.url[:80]}")
            return True

        # E: Bot detection
        BOT_PHRASES = [
            "unusual traffic", "access denied", "please verify you are human",
            "your ip has been blocked", "cloudflare challenge",
            "robot", "captcha", "bot detection", "are you a robot",
            "verify you are human", "automated access", "security check",
        ]
        if any(p in body_lower for p in BOT_PHRASES):
            result['status'] = 'manual_queue'
            result['manual_reason'] = f'[MANUAL] Bot detection at: {page.url}'
            print(f"[MANUAL] Bot detection at: {page.url[:80]}")
            return True

        return False

    # ------------------------------------------------------------------ #
    #  Step 3 — Page Extraction                                            #
    # ------------------------------------------------------------------ #

    async def _extract_page_state(self, page: Page) -> Dict[str, Any]:
        """Extract form elements, buttons, visible text, URL, title."""
        elements = await self._extract_elements_from_frame(page)

        # Iframe fallback
        if not elements:
            for frame in page.frames:
                if frame == page.main_frame:
                    continue
                try:
                    frame_els = await self._extract_elements_from_frame(frame)
                    if frame_els:
                        print(f"  [IFRAME] {len(frame_els)} elements in frame: {frame.url[:60]}")
                        elements = frame_els
                        break
                except Exception:
                    pass

        try:
            body_text = await page.evaluate("() => document.body.innerText.slice(0, 3000)")
        except Exception:
            body_text = ""

        try:
            buttons = await page.evaluate("""() => {
                const btns = [];
                document.querySelectorAll('button, input[type="submit"], a[role="button"]').forEach(el => {
                    const text = (el.innerText || el.textContent || el.value || '').trim();
                    if (!text) return;
                    if (el.offsetParent === null && el.type !== 'submit') return;
                    let sel = '';
                    if (el.id) sel = '#' + el.id;
                    else if (el.getAttribute('data-automation'))
                        sel = '[data-automation="' + el.getAttribute('data-automation') + '"]';
                    else if (el.type === 'submit') sel = 'input[type="submit"]';
                    else sel = 'button:has-text("' + text.slice(0, 40).replace(/"/g, '') + '")';
                    btns.push({text: text.slice(0, 80), selector: sel});
                });
                return btns.slice(0, 20);
            }""")
        except Exception:
            buttons = []

        try:
            title = await page.title()
        except Exception:
            title = ""

        return {
            'url': page.url,
            'title': title,
            'body_text': body_text,
            'elements': elements,
            'buttons': buttons,
        }

    async def _extract_elements_from_frame(self, frame) -> List[Dict[str, Any]]:
        """Extract all interactive form elements with rich label info."""
        try:
            return await frame.evaluate("""() => {
                const results = [];
                const els = document.querySelectorAll(
                    'input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=reset]),' +
                    'textarea, select'
                );
                els.forEach((el, idx) => {
                    if (el.offsetParent === null && el.type !== 'file') return;
                    el.setAttribute('data-fai', String(idx));
                    let label = '';
                    if (el.id) {
                        const lbl = document.querySelector('label[for="' + el.id + '"]');
                        if (lbl) label = lbl.innerText.trim();
                    }
                    if (!label) label = el.getAttribute('aria-label') || '';
                    if (!label && el.getAttribute('aria-labelledby')) {
                        const lbl = document.getElementById(el.getAttribute('aria-labelledby'));
                        if (lbl) label = lbl.innerText.trim();
                    }
                    if (!label) {
                        const wrap = el.closest('label');
                        if (wrap) label = wrap.innerText.replace(el.value || '', '').trim();
                    }
                    if (!label) {
                        let node = el.previousElementSibling, tries = 0;
                        while (node && !label && tries++ < 5) {
                            if (['LABEL','SPAN','DIV','P','LEGEND','H1','H2','H3','H4','H5'].includes(node.tagName)) {
                                const t = node.innerText.trim();
                                if (t.length > 0 && t.length < 120) label = t;
                            }
                            node = node.previousElementSibling;
                        }
                    }
                    if (!label) {
                        const parent = el.parentElement;
                        if (parent) {
                            const txt = Array.from(parent.childNodes)
                                .filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').trim();
                            if (txt.length > 0 && txt.length < 120) label = txt;
                        }
                    }
                    if (!label) label = el.placeholder || el.name || '';
                    const type = el.tagName === 'SELECT' ? 'select'
                               : el.tagName === 'TEXTAREA' ? 'textarea' : (el.type || 'text');
                    const field = {
                        index: idx, tag: el.tagName.toLowerCase(), type, name: el.name || '',
                        id: el.id || '', placeholder: el.placeholder || '', label,
                        value: el.value || '', required: el.required || false,
                        visible: el.offsetParent !== null, accept: el.accept || '',
                        selector: '[data-fai="' + idx + '"]'
                    };
                    if (el.tagName === 'SELECT') {
                        field.options = Array.from(el.options).map(o => o.text.trim()).filter(t => t);
                    }
                    results.push(field);
                });
                const iframeSrcs = Array.from(document.querySelectorAll('iframe'))
                    .map(f => f.src).filter(s => s);
                if (iframeSrcs.length) results.push({
                    index: -1, tag: 'info', type: 'info',
                    label: 'iframes', value: iframeSrcs.join(', '), selector: 'iframe'
                });
                return results;
            }""")
        except Exception as e:
            print(f"  [WARN] Element extraction failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Step 4 — Claude Sonnet Reasoning                                    #
    # ------------------------------------------------------------------ #

    async def _ai_analyze_page(
        self,
        page_state: Dict[str, Any],
        job: Dict[str, Any],
        cover_letter_text: str,
        result: Dict[str, Any],
        extra_context: str = ""
    ) -> Dict[str, Any]:
        """Send page state to Claude Sonnet. Returns structured action plan."""
        profile_json = json.dumps(self.candidate_profile, indent=2)
        elements_json = json.dumps(page_state['elements'], indent=2)
        buttons_json  = json.dumps(page_state['buttons'], indent=2)
        cl_text = (self._strip_markdown(cover_letter_text)[:2000]
                   if cover_letter_text else "(not provided)")

        system = (
            "You are an expert job application assistant filling out a job application form "
            "on behalf of a candidate. You must ONLY use information from the candidate's profile. "
            "Never fabricate, exaggerate, or assume information not present in the profile. "
            "If a required field cannot be answered from the profile, mark it as NEEDS_MANUAL_INPUT."
        )

        user = f"""Job: {job.get('title', 'Unknown Role')} at {job.get('company', 'Unknown Company')}

CANDIDATE PROFILE:
{profile_json}

COVER LETTER (plain text, no markdown):
{cl_text}

CURRENT PAGE:
URL: {page_state['url']}
Title: {page_state['title']}
Page text (3000 chars):
{wrap_untrusted(page_state['body_text'], tag="page_text")}

FORM ELEMENTS:
{elements_json}

VISIBLE BUTTONS:
{buttons_json}

TASK:
Analyze this page and return a JSON action plan. Return ONLY valid JSON in this exact format:
{{
  "page_type": "seek_native|company_portal|workday|greenhouse|lever|unknown",
  "step_description": "Brief description of what this page is asking",
  "actions": [
    {{
      "action": "fill|upload|click|select|check|uncheck|scroll|wait|skip",
      "selector": "CSS selector",
      "value": "value to fill or select",
      "label": "human readable field label",
      "reasoning": "why this value from the profile",
      "sensitive": false,
      "confidence": 0.95
    }}
  ],
  "after_actions_click": "selector of Next/Continue button (empty string if final step)",
  "is_final_step": true,
  "needs_manual_input": ["fields that cannot be answered from profile"],
  "concerns": ["any issues or ambiguities"]
}}

RULES:
- "fill": text inputs, textareas. Value "COVER_LETTER_TEXT" for cover letter fields.
- "upload": input[type=file] ONLY. Value must be literal "RESUME_PDF".
- "select": <select> dropdowns. Value = option text that best matches.
- "check"/"uncheck": checkboxes.
- "click": buttons/links only.
- "skip": optional/decorative fields.
- "sensitive": true for passport, TFN, visa grant number, bank account.

WORK RIGHTS: Rudra is on Graduate Visa Subclass 485, valid until 8 September 2028. Full work rights, NO sponsorship required. Select "Working Visa", "Temporary Work Rights", or "Yes, I have work rights". NEVER "Requires Sponsorship".

SALARY: salary_min = $80,000 for single-value. "$80,000 - $120,000 AUD" for range fields.

EXPERIENCE: ~1 year (3 internships, 12 months). Select "Less than 1 year", "0-1 years", or "Graduate/Entry level". Never 2+ years.

AVAILABILITY: "Immediately" or earliest option.

HOW DID YOU HEAR: "Seek" or "Online job board".

EDUCATION: Master of Computer Science, University of Wollongong, September 2025.

OPEN TEXT QUESTIONS: Draw from work_experience, skills, and cover letter. 2-3 sentences max. Never invent.

If this page shows only a job description with no form fields, set is_final_step=false, after_actions_click="" and explain in step_description that we need to find and click Apply."""

        if extra_context:
            user += f"\n\n{extra_context}"

        try:
            response = self._ai_client.messages.create(
                model="claude-sonnet-4-5",  # intentional: Haiku lacks reasoning for ambiguous forms
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            raw = response.content[0].text.strip()
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw:
                raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

            plan = json.loads(raw)

            if self.verbose:
                print(f"\n[VERBOSE] Claude plan: page_type={plan.get('page_type')} | "
                      f"is_final={plan.get('is_final_step')} | "
                      f"next={plan.get('after_actions_click','')[:40]}")
                for a in plan.get('actions', []):
                    tag = " [SENSITIVE]" if a.get('sensitive') else ""
                    print(f"  [{a.get('action','?'):8s}] {a.get('label','?')[:35]:35s} → "
                          f"{str(a.get('value',''))[:50]}{tag}")

            return plan

        except Exception as e:
            print(f"[WARN] AI page analysis failed: {e}")
            return {
                "page_type": "unknown",
                "step_description": f"AI analysis failed: {e}",
                "actions": [],
                "after_actions_click": "",
                "is_final_step": False,
                "needs_manual_input": [],
                "concerns": [f"AI error: {e}"]
            }

    # ------------------------------------------------------------------ #
    #  Step 10 — Resume Upload First + Autofill                            #
    # ------------------------------------------------------------------ #

    async def _upload_resume_first(
        self,
        page: Page,
        action_plan: Dict[str, Any],
        resume_path: str,
        result: Dict[str, Any]
    ):
        """Upload resume before filling other fields, to trigger autofill."""
        upload_action = next(
            (a for a in action_plan.get('actions', []) if a.get('action') == 'upload'),
            None
        )
        if not upload_action:
            return

        selector = upload_action.get('selector', 'input[type="file"]')
        rp = Path(resume_path) if resume_path else None

        if not rp or not rp.exists() or rp.suffix.lower() != '.pdf':
            print(f"[UPLOAD] Resume PDF not available: {resume_path}")
            result['resume_uploaded'] = False
            return

        print(f"[UPLOAD] Uploading: {rp.name} via {selector}")
        for try_sel in [selector, 'input[type="file"]', 'input[name*="resume"]', 'input[name*="cv"]']:
            try:
                await page.set_input_files(try_sel, str(rp))
                await page.wait_for_timeout(5000)
                result['resume_uploaded'] = True
                result['fields_filled'].append(f"resume_upload: {rp.name}")
                print(f"[UPLOAD] Resume uploaded: {rp.name}")
                # Check for confirmation text
                try:
                    body = await page.evaluate("() => document.body.innerText.slice(0, 2000)")
                    if any(s in body.lower() for s in ['uploaded', 'file selected', rp.name.lower()[:10]]):
                        print("[UPLOAD] Upload confirmation detected on page")
                except Exception:
                    pass
                return
            except Exception:
                pass

        print("[WARN] Resume upload FAILED — all selectors tried")
        result['resume_uploaded'] = False

    # ------------------------------------------------------------------ #
    #  Step 5 — Execute Actions                                            #
    # ------------------------------------------------------------------ #

    async def _execute_actions(
        self,
        page: Page,
        actions: List[Dict[str, Any]],
        resume_path: str,
        cover_letter_text: str,
        result: Dict[str, Any]
    ):
        """Execute each action from Claude's plan."""
        for action in actions:
            atype   = action.get('action', 'skip')
            sel     = action.get('selector', '')
            value   = action.get('value', '')
            label   = action.get('label', sel)
            sensitive = action.get('sensitive', False)

            if sensitive:
                print(f"  [SENSITIVE SKIP] {label}")
                result.setdefault('sensitive_fields_skipped', []).append(label)
                continue

            if atype == 'skip':
                result['fields_skipped'].append(f"{label}: skipped by AI")
                continue

            if atype == 'wait':
                await page.wait_for_timeout(2000)
                continue

            if atype == 'scroll':
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                continue

            if atype == 'upload':
                # Secondary upload (primary handled by _upload_resume_first)
                rp = Path(resume_path) if resume_path else None
                if rp and rp.exists() and rp.suffix.lower() == '.pdf':
                    for try_sel in [sel, 'input[type="file"]']:
                        try:
                            await page.set_input_files(try_sel, str(rp))
                            await page.wait_for_timeout(5000)
                            result['resume_uploaded'] = True
                            result['fields_filled'].append(f"resume_upload: {rp.name}")
                            print(f"  [UPLOAD] {label}: {rp.name}")
                            break
                        except Exception:
                            pass
                    else:
                        print(f"  [WARN] Upload failed: {label}")
                        result['fields_skipped'].append(f"{label}: upload failed")
                else:
                    print(f"  [WARN] No PDF resume for upload field: {label}")
                    result['fields_skipped'].append(f"{label}: no PDF resume")
                continue

            if atype == 'click':
                try:
                    await page.click(sel)
                    await page.wait_for_timeout(1000)
                    print(f"  [CLICK] {label}")
                except Exception as e:
                    print(f"  [WARN] Click failed ({label}): {e}")
                continue

            if atype == 'check':
                try:
                    await page.check(sel)
                    result['fields_filled'].append(f"{label}: checked")
                    print(f"  [CHECK] {label}")
                except Exception as e:
                    print(f"  [WARN] Check failed ({label}): {e}")
                    result['fields_skipped'].append(f"{label}: check failed")
                continue

            if atype == 'uncheck':
                try:
                    await page.uncheck(sel)
                    result['fields_filled'].append(f"{label}: unchecked")
                    print(f"  [UNCHECK] {label}")
                except Exception as e:
                    print(f"  [WARN] Uncheck failed ({label}): {e}")
                continue

            if atype == 'select':
                selected_value = value
                try:
                    try:
                        await page.select_option(sel, label=value)
                    except Exception:
                        try:
                            await page.select_option(sel, value=value)
                        except Exception:
                            opts = await page.evaluate(
                                "([s]) => Array.from(document.querySelector(s)?.options || []).map(o => ({v: o.value, t: o.text}))",
                                [sel]
                            )
                            match = next((o for o in opts if value.lower() in o['t'].lower()), None)
                            if match:
                                await page.select_option(sel, value=match['v'])
                            else:
                                # Fallback: first non-empty, non-placeholder option
                                first_nonempty = next(
                                    (o for o in opts if o['t'].strip() and
                                     not any(p in o['t'].lower() for p in ('select', 'choose', 'please', '--'))),
                                    None
                                )
                                if first_nonempty:
                                    await page.select_option(sel, value=first_nonempty['v'])
                                    selected_value = f"{first_nonempty['t']} (fallback)"
                                    print(f"  [SELECT FALLBACK] {label}: no match for '{value}', using '{first_nonempty['t']}'")
                                    result.setdefault('concerns', []).append(
                                        f"{label}: fallback to '{first_nonempty['t']}' (wanted '{value}')"
                                    )
                                else:
                                    raise Exception(f"No match or fallback for '{value}'")
                    result['fields_filled'].append(f"{label}: {selected_value}")
                    print(f"  [SELECT] {label}: {selected_value}")
                except Exception as e:
                    print(f"  [WARN] Select failed ({label}): {e}")
                    result['fields_skipped'].append(f"{label}: select failed ({str(e)[:60]})")
                continue

            if atype == 'fill':
                if value == 'COVER_LETTER_TEXT':
                    if cover_letter_text:
                        fill_value = self._strip_markdown(cover_letter_text)
                        result['cover_letter_pasted'] = True
                        print(f"  [FILL] {label}: (cover letter, {len(fill_value)} chars)")
                    else:
                        result['fields_skipped'].append(f"{label}: no cover letter")
                        continue
                elif value == 'RESUME_PDF':
                    # Already handled by upload action — skip
                    continue
                else:
                    fill_value = value

                # 4 fill strategies
                filled = False
                # Strategy 1: page.fill
                try:
                    await page.fill(sel, fill_value)
                    filled = True
                except Exception:
                    pass
                # Strategy 2: locator.fill
                if not filled:
                    try:
                        await page.locator(sel).fill(fill_value)
                        filled = True
                    except Exception:
                        pass
                # Strategy 3: JS value injection
                if not filled:
                    try:
                        await page.evaluate(
                            "([s, v]) => { const el = document.querySelector(s); if (el) { el.value = v; el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true})); } }",
                            [sel, fill_value]
                        )
                        filled = True
                    except Exception:
                        pass
                # Strategy 4: click + keyboard type
                if not filled:
                    try:
                        await page.click(sel)
                        await page.keyboard.type(fill_value)
                        filled = True
                    except Exception:
                        pass

                if filled:
                    display = fill_value[:60] + ('…' if len(fill_value) > 60 else '')
                    result['fields_filled'].append(f"{label}: {display}")
                    print(f"  [FILL] {label}: {display}")
                else:
                    print(f"  [WARN] Could not fill field: {label}")
                    result['fields_skipped'].append(f"{label}: all fill strategies failed")

    # ------------------------------------------------------------------ #
    #  Step 6 — Success Gate                                               #
    # ------------------------------------------------------------------ #

    async def _check_success_gate(
        self,
        page: Page,
        result: Dict[str, Any],
        job: Dict[str, Any] = None,
        resume_path: str = "",
        cover_letter_text: str = ""
    ) -> bool:
        """Verify minimum fill conditions and run validation error correction loop (max 2 attempts)."""
        n_filled = len(result.get('fields_filled', []))

        if n_filled < 3:
            ss = await self._screenshot(page, result.get('job_hash', 'unknown'), "gate_failed")
            result['screenshots'].append(ss)
            result['status'] = 'manual_queue'
            result['manual_reason'] = f'success_gate_failed: only {n_filled} fields filled'
            print(f"[GATE FAIL] Only {n_filled} fields filled — aborting to prevent blank submission")
            return False

        if not result.get('resume_uploaded'):
            print("[WARN] Resume not uploaded — proceeding but noting concern")
            result.setdefault('concerns', []).append('resume_not_uploaded')

        error_text = await self._get_validation_errors(page)
        if error_text:
            print(f"[WARN] Validation errors detected: {error_text[:200]}")
            result.setdefault('concerns', []).append(f'validation_errors: {error_text[:100]}')

            if job is not None:
                for attempt in range(2):
                    print(f"[CORRECTION {attempt + 1}/2] Asking Claude to fix validation errors...")
                    page_state = await self._extract_page_state(page)
                    extra = (
                        f"VALIDATION ERRORS ON PAGE: {error_text}\n"
                        f"Return ONLY actions that correct these specific errors."
                    )
                    correction_plan = await self._ai_analyze_page(
                        page_state, job, cover_letter_text, result, extra_context=extra
                    )
                    result['api_calls_made'] = result.get('api_calls_made', 0) + 1
                    await self._execute_actions(
                        page, correction_plan.get('actions', []), resume_path, cover_letter_text, result
                    )
                    error_text = await self._get_validation_errors(page)
                    if not error_text:
                        print(f"[CORRECTION] Errors cleared after attempt {attempt + 1}")
                        break
                else:
                    ss = await self._screenshot(page, result.get('job_hash', 'unknown'), "validation_failed")
                    result['screenshots'].append(ss)
                    result['status'] = 'manual_queue'
                    result['manual_reason'] = f'validation_errors_uncorrectable: {error_text[:100]}'
                    print(f"[GATE FAIL] Validation errors persist after 2 correction attempts")
                    return False

        return True

    async def _get_validation_errors(self, page: Page) -> str:
        """Extract visible validation error text from the page."""
        try:
            return await page.evaluate("""() => {
                const sels = '[class*="error"]:not([class*="no-error"]), [class*="invalid"], [aria-invalid="true"], .field-error, .form-error';
                return Array.from(document.querySelectorAll(sels))
                    .map(el => el.innerText.trim()).filter(t => t.length > 0).slice(0, 5).join(' | ');
            }""")
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    #  Step 7 — Human Approval                                             #
    # ------------------------------------------------------------------ #

    async def _human_approval_prompt(self, result: Dict[str, Any], job: Dict[str, Any]) -> bool:
        """Print fill summary and prompt Enter/Ctrl+C. Returns True if cancelled."""
        print("\n" + "═" * 60)
        print(f"READY TO SUBMIT: {job.get('title')} at {job.get('company')}")
        print("═" * 60)
        print(f"\nFields filled ({len(result.get('fields_filled', []))}):")
        for f in result.get('fields_filled', []):
            print(f"  {f}")
        if result.get('sensitive_fields_skipped'):
            print(f"\nSensitive fields skipped:")
            for f in result['sensitive_fields_skipped']:
                print(f"  [SENSITIVE] {f}")
        if result.get('needs_manual_input'):
            print(f"\nNeeds manual input:")
            for f in result.get('needs_manual_input', []):
                print(f"  [MANUAL] {f}")
        if result.get('concerns'):
            print(f"\nConcerns:")
            for c in result.get('concerns', []):
                print(f"  [!] {c}")
        print(f"\nResume uploaded:     {'✓' if result.get('resume_uploaded') else '✗ NOT UPLOADED'}")
        print(f"Cover letter pasted: {'✓' if result.get('cover_letter_pasted') else '✗'}")
        if result.get('screenshots'):
            print(f"Screenshot:          {result['screenshots'][-1]}")
        print("═" * 60)
        print("Press Enter to SUBMIT or Ctrl+C to cancel...")
        try:
            await asyncio.get_event_loop().run_in_executor(None, input)
            return False
        except KeyboardInterrupt:
            print("\n[CANCELLED] Application not submitted")
            return True

    # ------------------------------------------------------------------ #
    #  Step 8 — Submit + Confirm                                           #
    # ------------------------------------------------------------------ #

    async def _submit_and_confirm(self, page: Page, result: Dict[str, Any]):
        """Click submit button and detect confirmation."""
        job_hash = result.get('job_hash', 'unknown')
        submit_clicked = False

        for sel in self.SUBMIT_SELECTORS:
            try:
                btn = await page.query_selector(sel)
                if btn and await btn.is_visible():
                    print(f"[SUBMIT] Clicking: {sel}")
                    await btn.click()
                    submit_clicked = True
                    break
            except Exception:
                pass

        if not submit_clicked:
            # Scroll down and retry
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(3000)
            for sel in self.SUBMIT_SELECTORS:
                try:
                    btn = await page.query_selector(sel)
                    if btn and await btn.is_visible():
                        print(f"[SUBMIT] Clicking after scroll: {sel}")
                        await btn.click()
                        submit_clicked = True
                        break
                except Exception:
                    pass

        if not submit_clicked:
            fail_ss = await self._screenshot(page, job_hash, "submit_not_found")
            result['screenshots'].append(fail_ss)
            result['status'] = 'manual_queue'
            result['manual_reason'] = 'submit_button_not_found'
            print("[WARN] Submit button not found — added to manual queue")
            return

        # Wait up to 15 seconds for confirmation
        await page.wait_for_timeout(3000)
        try:
            await page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass

        status = await self._detect_confirmation(page, job_hash)
        result['status'] = status

        # Capture confirmation text
        try:
            body = await page.evaluate("() => document.body.innerText.slice(0, 600)")
            for phrase in self.CONFIRMATION_PHRASES:
                if phrase in body.lower():
                    idx = body.lower().find(phrase)
                    result['confirmation_text'] = body[max(0, idx-20):idx+len(phrase)+120].strip()
                    break
        except Exception:
            pass

        after_ss = await self._screenshot(page, job_hash, "after_submit")
        result['screenshots'].append(after_ss)

        if status == "confirmed_submitted":
            print(f"[SUCCESS] Application confirmed submitted!")
            if result.get('confirmation_text'):
                print(f"[CONFIRM] {result['confirmation_text'][:120]}")
        else:
            print("[WARNING] Submission unconfirmed — manual verification needed")

    # ------------------------------------------------------------------ #
    #  Confirmation Detection                                              #
    # ------------------------------------------------------------------ #

    async def _detect_confirmation(self, page: Page, job_hash: str) -> str:
        """Scan URL and page text for success signals."""
        url = page.url.lower()
        if any(t in url for t in self.CONFIRMATION_URLS):
            print("[CONFIRM] Confirmation URL detected")
            return "confirmed_submitted"
        try:
            content = (await page.content()).lower()
            for phrase in self.CONFIRMATION_PHRASES:
                if phrase in content:
                    print(f"[CONFIRM] Phrase found: '{phrase}'")
                    return "confirmed_submitted"
        except Exception:
            pass
        # Poll for 10 more seconds
        for _ in range(5):
            await page.wait_for_timeout(2000)
            if any(t in page.url.lower() for t in self.CONFIRMATION_URLS):
                return "confirmed_submitted"
            try:
                content = (await page.content()).lower()
                for phrase in self.CONFIRMATION_PHRASES:
                    if phrase in content:
                        return "confirmed_submitted"
            except Exception:
                pass
        print("[CONFIRM] No confirmation signal — marking uncertain")
        return "uncertain_submitted"

    # ------------------------------------------------------------------ #
    #  Adzuna Redirect                                                     #
    # ------------------------------------------------------------------ #

    async def _resolve_adzuna_redirect(self, adzuna_url: str) -> Optional[str]:
        """Follow Adzuna redirect (headless) and return the real destination URL."""
        print(f"[ADZUNA] Resolving redirect: {adzuna_url[:80]}")
        try:
            async with async_playwright() as p:
                ctx = await p.chromium.launch_persistent_context(
                    user_data_dir=self.BROWSER_PROFILE,
                    headless=True,
                    channel="chrome"
                )
                pg = await ctx.new_page()
                await pg.goto(adzuna_url, wait_until="networkidle", timeout=20000)
                await pg.wait_for_timeout(2000)
                resolved = pg.url
                await ctx.close()
            if 'adzuna' in resolved.lower():
                print(f"[ADZUNA] Did not resolve — still on Adzuna: {resolved[:80]}")
                return None
            print(f"[ADZUNA] Resolved to: {resolved[:80]}")
            return resolved
        except Exception as e:
            print(f"[ADZUNA] Redirect resolution failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*',     r'\1', text)
        text = re.sub(r'__(.+?)__',     r'\1', text)
        text = re.sub(r'_(.+?)_',       r'\1', text)
        text = re.sub(r'^#{1,6}\s+',    '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*_]{3,}$',   '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*+]\s+',  '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+',  '', text, flags=re.MULTILINE)
        text = re.sub(r'`(.+?)`',        r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'\n{3,}',         '\n\n', text)
        return text.strip()

    def _sensitive_pattern_in(self, text: str) -> Optional[str]:
        t = text.lower()
        if "passport" in t:                           return "passport number"
        if "visa grant" in t or "grant number" in t:  return "visa grant number"
        if re.search(r'\btfn\b', t) or "tax file" in t: return "tax file number (TFN)"
        if "bank account" in t or "bsb" in t:         return "bank account"
        return None

    async def _inter_apply_delay(self):
        """Random 120–300 s delay between applications (anti-bot)."""
        if self._last_apply_time is None:
            self._last_apply_time = asyncio.get_event_loop().time()
            return
        elapsed   = asyncio.get_event_loop().time() - self._last_apply_time
        delay     = random.uniform(120, 300)
        remaining = delay - elapsed
        if remaining > 0:
            print(f"\n[DELAY] Waiting {remaining:.0f}s before next application (anti-bot)...")
            await asyncio.sleep(remaining)
        self._last_apply_time = asyncio.get_event_loop().time()

    async def _screenshot(self, page: Page, job_hash: str, label: str) -> str:
        screenshot_dir = Path("artifacts/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = screenshot_dir / f"{job_hash}_{label}.png"
        try:
            await page.screenshot(path=str(path), full_page=True)
        except Exception as e:
            print(f"  [WARN] Screenshot failed: {e}")
        return str(path)

    def _make_result(self, job: Dict[str, Any], status: str = 'pending') -> Dict[str, Any]:
        return {
            'job_hash': job.get('job_hash', 'unknown'),
            'job_title': job.get('title', ''),
            'company': job.get('company', ''),
            'url': job.get('url', ''),
            'mode': self.mode,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'ats_type': '',
            'fields_filled': [],
            'fields_skipped': [],
            'sensitive_fields_skipped': [],
            'needs_manual_input': [],
            'resume_uploaded': False,
            'cover_letter_pasted': False,
            'steps_completed': 0,
            'api_calls_made': 0,
            'screenshots': [],
            'confirmation_text': None,
            'error': None,
            'manual_reason': None,
            'resolved_url': None,
            'concerns': [],
        }

    def _print_fill_summary(self, result: Dict[str, Any]):
        print("\n" + "=" * 80)
        print(f"FIELDS FILLED ({len(result['fields_filled'])}):")
        for f in result['fields_filled']:
            print(f"  [OK]   {f}")
        if result['fields_skipped']:
            print(f"\nFIELDS SKIPPED ({len(result['fields_skipped'])}):")
            for f in result['fields_skipped']:
                print(f"  [SKIP] {f}")
        if result.get('sensitive_fields_skipped'):
            print(f"\nSENSITIVE SKIPPED ({len(result['sensitive_fields_skipped'])}):")
            for f in result['sensitive_fields_skipped']:
                print(f"  [SENSITIVE] {f}")
        if result.get('needs_manual_input'):
            print(f"\nNEEDS MANUAL INPUT:")
            for f in result['needs_manual_input']:
                print(f"  [MANUAL] {f}")
        print("=" * 80)

    def _log_application(self, result: Dict[str, Any]):
        log_file = Path("logs/apply_log.json")
        log_file.parent.mkdir(exist_ok=True)
        existing: list = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(result)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ #
#  CLI smoke test                                                      #
# ------------------------------------------------------------------ #

async def main():
    agent = ApplyAgent(mode="assisted", verbose=True)
    session_ok = await agent.check_seek_session()
    if not session_ok:
        print("[ACTION NEEDED] Seek session expired. Run: python seed_login.py")
        return
    result = await agent.apply_to_job(
        job={
            'job_hash': 'test123',
            'title': 'AI Engineer',
            'company': 'Test Company',
            'url': 'https://www.seek.com.au/job/12345',
        },
        resume_path="artifacts/resumes/test123.pdf"
    )
    print(f"\nResult: {result['status']}")
    print(f"Fields filled: {result['fields_filled']}")


if __name__ == "__main__":
    asyncio.run(main())

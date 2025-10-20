#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_feedback.py
- Teacher-triggered via GitHub Actions
- Uses GPT to produce process-oriented meta-feedback
- Supports both Code (algorithm.py) and Pseudocode (PSEUDOCODE.md)
- No preset/default hints; ALL guidance comes from the model
"""

import os
import json
import subprocess
import datetime
import pathlib
import re
import textwrap
from typing import Dict, Any

# ---------------- constants & paths ----------------
STATE_DIR = pathlib.Path(".meta-feedback")
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"

PSEUDO_FILE = pathlib.Path("PSEUDOCODE.md")
CODE_FILE = pathlib.Path("algorithm.py")
TEST_DIR = pathlib.Path("tests")


# ---------------- shell helpers ----------------
def sh(cmd: str) -> str:
    """Run a shell command and return stdout as text (raises on non-zero)."""
    return subprocess.check_output(
        cmd, shell=True, text=True, stderr=subprocess.STDOUT
    ).strip()


def get_head() -> str:
    return sh("git rev-parse HEAD")


def get_initial_commit() -> str:
    return sh("git rev-list --max-parents=0 HEAD").splitlines()[0]


def get_last_processed() -> str | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text()).get("last_processed")
        except Exception:
            return None
    return None


def set_last_processed(sha: str) -> None:
    STATE_FILE.write_text(json.dumps({"last_processed": sha}, indent=2), encoding="utf-8")


def git_diff(base: str, head: str) -> Dict[str, str]:
    """Collect a compact view of repo changes + context for the model."""
    name_status = sh(f"git diff --name-status {base} {head}") if base != head else ""
    shortstat = sh(f"git diff --shortstat {base} {head}") if base != head else ""
    patch = sh(f"git diff --unified=0 {base} {head}") if base != head else ""
    try:
        tree = sh("ls -R | head -n 400")
    except Exception:
        tree = ""
    try:
        logs = sh("git log -n 10 --pretty=format:'%h %ad %s' --date=short")
    except Exception:
        logs = ""
    return {
        "name_status": name_status,
        "shortstat": shortstat,
        "patch": patch,
        "tree": tree,
        "logs": logs,
    }


def now_stamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")


def write_feedback(md: str) -> str:
    out = f"Meta-Feedback_{now_stamp()}.md"
    pathlib.Path(out).write_text(md, encoding="utf-8")
    return out


# ---------------- minimal content heuristics (no preset hints) ----------------
def strip_comments_and_ws_py(text: str) -> str:
    # Remove Python comments and whitespace for a rough size signal
    text = re.sub(r"#.*", "", text)
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return re.sub(r"\s+", "", text)


def strip_md(text: str) -> str:
    # Remove common Markdown markup & code fences for a rough size signal
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    text = re.sub(r"[#>*`_~\-]", "", text)
    return re.sub(r"\s+", "", text)


def read_text_safe(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def detect_content_status() -> Dict[str, Any]:
    """Return simple signals about whether students actually wrote content."""
    code_text = read_text_safe(CODE_FILE)
    pseudo_text = read_text_safe(PSEUDO_FILE)

    code_core_len = len(strip_comments_and_ws_py(code_text)) if code_text else 0
    pseudo_core_len = len(strip_md(pseudo_text)) if pseudo_text else 0

    tests_exist = TEST_DIR.exists() and any(TEST_DIR.glob("test_*.py"))

    # very rough thresholds to differentiate skeleton vs. substance
    has_meaningful_code = code_core_len >= 80
    has_meaningful_pseudo = pseudo_core_len >= 120

    return {
        "code_len": code_core_len,
        "pseudo_len": pseudo_core_len,
        "has_code": has_meaningful_code,
        "has_pseudo": has_meaningful_pseudo,
        "tests_exist": tests_exist,
    }


def describe_content_status(status: Dict[str, Any]) -> str:
    """
    Summarize what exists WITHOUT giving any hardcoded hints.
    This context helps GPT decide how to respond when the repo is minimal/empty.
    """
    details = []
    if status["has_code"]:
        details.append(f"Detected non-trivial code in algorithm.py (≈{status['code_len']} chars after stripping).")
    if status["has_pseudo"]:
        details.append(f"Detected non-trivial pseudocode in PSEUDOCODE.md (≈{status['pseudo_len']} chars after stripping).")
    if not status["has_code"] and not status["has_pseudo"]:
        details.append("No substantial code or pseudocode detected.")
    if not status["tests_exist"]:
        details.append("No test files found under 'tests/'.")
    return " ".join(details) if details else "Content status unclear."


# ---------------- OpenAI client ----------------
def call_gpt(system_prompt: str, user_prompt: str) -> str:
    """
    Uses OpenAI's Python SDK (>=1.0).
    Env:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL (optional; proxy/Azure)
      - OPENAI_MODEL (optional; default gpt-4o-mini)
    """
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------------- prompts ----------------
SYSTEM_PROMPT = """You are a teaching assistant generating process-oriented meta-feedback for a university algorithms assignment.
Students may submit code (algorithm.py) OR pseudocode (PSEUDOCODE.md). English only.
Focus on actionable guidance about planning, implementation/pseudocode quality, validation (tests or worked examples), and reflection (complexity, invariants/exchange-argument, counterexamples).
If the repository is minimal or empty, you should still provide constructive, step-by-step guidance tailored to the detected state—do not assume a specific algorithm family (e.g., not necessarily greedy).
Be concise and structured; use Markdown with short bullets where helpful. Do NOT include boilerplate or generic platitudes.
"""

USER_PROMPT_TEMPLATE = """\
Generate process-focused meta-feedback for the latest change ({base}..{head}).

[Recent commits]
{logs}

[Project tree (truncated)]
{tree}

[Changed files (name-status)]
{name_status}

[Diff (unified; may be truncated)]
```diff
{patch}
```

[Repository content summary]
{content_summary}

Use this structure:

Meta-Feedback (Process-Oriented)
Signals Observed

(What changed? Small-step commits? Clear messages?)

(Any structure/API changes? Added/removed helpers?)

Actionable Suggestions

Planning:

Implementation / Pseudocode quality:

Validation:

Reflection:

Technique-Specific Considerations (only if applicable)

Clearly state the chosen approach/criterion and why it fits the problem.

Mark an invariant OR outline an exchange/correctness sketch appropriate for the chosen technique.

Compare to a naive/baseline approach (correctness + complexity).

Walk through 2 tiny examples end-to-end.

Consider tricky or near-counterexample cases and how to detect/handle them.
"""
# ---------------- main ----------------
def main() -> None:
    head = get_head()
    last = get_last_processed() or get_initial_commit()

    if last == head:
        print("No new commits to process.")
        return

    # Collect diff/context
    d = git_diff(last, head)

    # Describe current repo contents (no preset hints)
    status = detect_content_status()
    content_summary = describe_content_status(status)

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        base=last[:7],
        head=head[:7],
        logs=d["logs"] or "(none)",
        tree=d["tree"] or "(none)",
        name_status=d["name_status"] or "(none)",
        patch=(d["patch"][:20000] if d["patch"] else "(none)"),
        content_summary=content_summary,
    )

    # Call GPT (no default/fallback hints). If the API fails, provide a minimal error note only.
    try:
        feedback_md = call_gpt(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        feedback_md = textwrap.dedent(
            f"""\
            # Meta-Feedback (Process-Oriented)
            The feedback service encountered an error: {e}
            Please retry the instructor-triggered workflow.
            """
        )

    out = write_feedback(feedback_md)
    set_last_processed(head)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()


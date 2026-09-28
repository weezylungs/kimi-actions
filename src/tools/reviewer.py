"""Code review tool for Kimi Actions.

Uses Agent SDK with Skill-based architecture.
Supports intelligent chunking and fallback models for large PRs.
Supports inline comments and incremental review.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import uuid
from typing import List, Optional, Tuple

from tools.base import BaseTool, DIFF_LIMIT_REVIEW
from token_handler import DiffChunk
from models import CodeSuggestion, SeverityLevel, ReviewOptions, SuggestionControl
from suggestion_service import SuggestionService

logger = logging.getLogger(__name__)


class Reviewer(BaseTool):
    """Code review tool using Agent SDK with Skill-based architecture."""

    @property
    def skill_name(self) -> str:
        return "code-review"

    def run(self, repo_name: str, pr_number: int, **kwargs) -> str:
        """Run code review on a PR."""
        incremental = kwargs.get("incremental", False)
        inline = kwargs.get("inline", True)
        command_quote = kwargs.get("command_quote", "")

        pr = self.github.get_pr(repo_name, pr_number)
        self.load_context(repo_name, ref=pr.head.sha)

        if incremental:
            compressed_diff, included_chunks, excluded_chunks, last_sha = (
                self._get_incremental_diff(repo_name, pr_number)
            )
            if compressed_diff is None:
                return "No new changes since last review."
        else:
            compressed_diff, included_chunks, excluded_chunks = self.get_diff(
                repo_name, pr_number
            )

        if not compressed_diff:
            return "No changes to review."

        skill = self.get_skill()
        if not skill:
            raise RuntimeError(f"{self.skill_name} skill not found")

        script_output = self._run_scripts(skill, compressed_diff)
        system_prompt = self._build_system_prompt(
            skill, script_output, compressed_diff
        )

        with tempfile.TemporaryDirectory() as work_dir:
            workspace = os.environ.get("GITHUB_WORKSPACE")
            if not workspace or not os.path.isdir(
                os.path.join(workspace, ".git")
            ):
                raise RuntimeError(
                    "Authenticated GitHub workspace is unavailable"
                )

            try:
                subprocess.run(
                    [
                        "git",
                        "config",
                        "--global",
                        "--add",
                        "safe.directory",
                        workspace,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--no-hardlinks",
                        workspace,
                        work_dir,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                subprocess.run(
                    [
                        "git",
                        "-C",
                        work_dir,
                        "checkout",
                        "--detach",
                        pr.head.sha,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            except subprocess.CalledProcessError as exc:
                detail = (
                    exc.stderr or exc.stdout or str(exc)
                ).strip()
                raise RuntimeError(
                    f"Failed to prepare isolated review workspace: {detail}"
                ) from exc

            logger.info(
                "Prepared isolated review workspace at PR head %s",
                pr.head.sha,
            )

            try:
                response = asyncio.run(
                    self._run_agent_review(
                        work_dir=work_dir,
                        system_prompt=system_prompt,
                        pr_title=pr.title,
                        pr_branch=f"{pr.head.ref} -> {pr.base.ref}",
                        diff=compressed_diff,
                    )
                )
            except Exception as exc:
                logger.error("Review failed: %s", exc)
                raise RuntimeError(
                    f"Kimi review failed: {exc}"
                ) from exc

        suggestions = self._parse_suggestions(response)

        review_options = ReviewOptions(
            bug=self.repo_config.enable_bug if self.repo_config else True,
            performance=(
                self.repo_config.enable_performance
                if self.repo_config
                else True
            ),
            security=(
                self.repo_config.enable_security
                if self.repo_config
                else True
            ),
        )

        suggestion_service = SuggestionService(
            SuggestionControl(
                max_suggestions=self.config.review.num_max_findings,
                severity_level_filter=SeverityLevel.LOW,
            )
        )

        filtered, discarded = suggestion_service.process_suggestions(
            suggestions,
            review_options,
            compressed_diff,
        )

        logger.info(
            "Suggestions: %s parsed, %s filtered",
            len(suggestions),
            len(filtered),
        )

        total_files = (
            len(included_chunks)
            if included_chunks
            else len(
                set(
                    s.relevant_file
                    for s in filtered
                    if s.relevant_file
                )
            )
        )

        posted_count = 0

        if inline and filtered:
            summary = self._format_inline_summary(
                response,
                filtered,
                len(filtered),
                total_files=total_files,
                included_chunks=included_chunks,
                incremental=incremental,
                current_sha=pr.head.sha,
                command_quote=command_quote,
            )

            posted_count = self._post_inline_comments(
                repo_name,
                pr_number,
                filtered,
                summary_body=summary,
            )

            if posted_count > 0:
                return ""

        summary = self._format_inline_summary(
            response,
            filtered,
            posted_count,
            total_files=total_files,
            included_chunks=included_chunks,
            incremental=incremental,
            current_sha=pr.head.sha,
            command_quote=command_quote,
        )

        return summary

    async def _run_agent_review(
        self,
        work_dir: str,
        system_prompt: str,
        pr_title: str,
        pr_branch: str,
        diff: str,
    ) -> str:
        """Run agent to perform code review."""
        try:
            from kimi_agent_sdk import (
                ApprovalRequest,
                Session,
                TextPart,
            )
        except ImportError as exc:
            raise RuntimeError(
                "kimi-agent-sdk not installed"
            ) from exc

        api_key = self.setup_agent_env()
        if not api_key:
            raise RuntimeError("KIMI_API_KEY is required")

        text_parts = []

        review_prompt = f"""{system_prompt}

## PR Information
Title: {pr_title}
Branch: {pr_branch}

## Code Changes
```diff
{diff[:DIFF_LIMIT_REVIEW]}
```

## Instructions
1. Analyze the code changes carefully
2. If needed, read related files to understand context
3. Identify bugs, security issues, and improvements
4. **IMPORTANT**: Generate a meaningful description for EVERY changed file in file_summaries

Please output review results in YAML format:
```yaml
summary: "Brief summary of the PR"
score: 85
file_summaries:
  - file: "path/to/file.py"
    description: "Detailed description of what changed in this file and why"
  - file: "path/to/another.js"
    description: "Detailed description of changes in this file"
suggestions:
  - relevant_file: "path/to/file.py"
    language: "python"
    severity: "medium"
    label: "bug"
    one_sentence_summary: "Brief issue description"
    suggestion_content: "Detailed explanation"
    existing_code: "problematic code"
    improved_code: "fixed code"
    relevant_lines_start: 10
    relevant_lines_end: 15
```

**Note**: Ensure file_summaries includes ALL changed files with meaningful descriptions, not just generic "Code changes" text.
"""

        try:
            async with await Session.create(
                work_dir=work_dir,
                model=self.AGENT_MODEL,
                yolo=True,
                max_steps_per_turn=100,
            ) as session:
                async for msg in session.prompt(review_prompt):
                    if isinstance(msg, TextPart):
                        text_parts.append(msg.text)
                    elif isinstance(msg, ApprovalRequest):
                        msg.resolve("approve")

            return "".join(text_parts)

        except Exception as exc:
            logger.error("Agent execution failed: %s", exc)
            raise RuntimeError(
                f"Kimi agent execution failed: {exc}"
            ) from exc

    def _get_incremental_diff(
        self,
        repo_name: str,
        pr_number: int,
    ) -> Tuple[
        Optional[str],
        List[DiffChunk],
        List[DiffChunk],
        Optional[str],
    ]:
        """Get diff only for new commits since last review."""
        last_review = self.github.get_last_bot_comment(
            repo_name,
            pr_number,
        )

        if not last_review:
            diff, included, excluded = self.get_diff(
                repo_name,
                pr_number,
            )
            return diff, included, excluded, None

        last_sha = last_review["sha"]

        new_commits = self.github.get_commits_since(
            repo_name,
            pr_number,
            last_sha,
        )

        if not new_commits:
            return None, [], [], last_sha

        commit_shas = [c.sha for c in new_commits]

        diff = self.github.get_diff_for_commits(
            repo_name,
            commit_shas,
        )

        if not diff:
            return None, [], [], last_sha

        included, excluded = self.chunker.chunk_diff(
            diff,
            max_files=self.config.max_files,
        )

        compressed = self.chunker.build_diff_string(included)

        return compressed, included, excluded, last_sha

    def _post_inline_comments(
        self,
        repo_name: str,
        pr_number: int,
        suggestions: List[CodeSuggestion],
        summary_body: str = "",
    ):
        """Post inline comments with GitHub native suggestion format."""
        suggestion_dicts = []

        for suggestion in suggestions:
            suggestion_dicts.append(
                {
                    "relevant_file": suggestion.relevant_file,
                    "relevant_lines_start": (
                        suggestion.relevant_lines_start
                    ),
                    "relevant_lines_end": (
                        suggestion.relevant_lines_end
                    ),
                    "suggestion_content": (
                        suggestion.suggestion_content
                    ),
                    "improved_code": suggestion.improved_code,
                }
            )

        return self.post_inline_comments(
            repo_name,
            pr_number,
            suggestion_dicts,
            summary_body=summary_body,
            use_suggestion_format=True,
        )

    def _format_inline_summary(
        self,
        response: str,
        suggestions: List[CodeSuggestion],
        inline_count: int,
        total_files: int = 0,
        included_chunks: List[DiffChunk] = None,
        incremental: bool = False,
        current_sha: str = None,
        command_quote: str = "",
    ) -> str:
        """Format a short summary when inline comments were posted."""
        data = self.parse_yaml_response(response) or {}
        summary = data.get("summary", "").strip()

        file_summaries = {}

        for file_summary in data.get("file_summaries", []):
            filename = file_summary.get("file", "")
            description = file_summary.get("description", "")

            if filename and description:
                file_summaries[filename] = description

        lines = []

        if command_quote:
            lines.append(f"> {command_quote}")
            lines.append("")

        lines.append("### 🌗 Pull request overview")

        if summary:
            lines.append(f"{summary}\n")
        else:
            lines.append("Code review completed.\n")

        files_reviewed = (
            total_files
            if total_files > 0
            else len(included_chunks)
            if included_chunks
            else 0
        )

        lines.append("**Reviewed changes**")
        lines.append(
            f"Kimi reviewed {files_reviewed} changed files "
            f"in this pull request and generated "
            f"{inline_count} comments.\n"
        )

        if included_chunks:
            lines.append("<details>")
            lines.append(
                "<summary>Show a summary per file</summary>\n"
            )
            lines.append("| File | Description |")
            lines.append("|------|-------------|")

            for chunk in included_chunks:
                if chunk.filename in file_summaries:
                    description = file_summaries[chunk.filename]
                else:
                    change_type_description = {
                        "added": "New file added",
                        "deleted": "File removed",
                        "modified": "Modified",
                        "renamed": "File renamed",
                    }.get(
                        chunk.change_type,
                        "Modified",
                    )

                    language_info = (
                        f" ({chunk.language})"
                        if chunk.language
                        else ""
                    )

                    description = (
                        f"{change_type_description}"
                        f"{language_info}"
                    )

                lines.append(
                    f"| `{chunk.filename}` | {description} |"
                )

            lines.append("\n</details>\n")

        if suggestions:
            lines.append("**Issues found:**")

            severity_icons = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
            }

            for suggestion in suggestions[:5]:
                icon = severity_icons.get(
                    suggestion.severity.value,
                    "⚪",
                )

                filename = suggestion.relevant_file or "unknown"

                issue_summary = (
                    suggestion.one_sentence_summary or ""
                ).replace("\n", " ").strip()

                lines.append(
                    f"- {icon} `{filename}`: {issue_summary}"
                )

            if len(suggestions) > 5:
                lines.append(
                    f"- ... and {len(suggestions) - 5} more"
                )

            lines.append("")

        lines.append(self.format_footer())

        if current_sha:
            lines.append(
                f"\n<!-- kimi-review:sha={current_sha[:12]} -->"
            )

        return "\n".join(lines)

    def _run_scripts(self, skill, diff: str) -> str:
        """Run skill scripts and collect output."""
        if not skill.scripts:
            return ""

        output_parts = []
        language = self._detect_language(diff)

        if "linter" in skill.scripts:
            result = skill.run_script(
                "linter",
                lang=language,
                code=diff[:5000],
            )

            if result:
                output_parts.append(
                    f"## Linter Output\n```text\n{result}\n```"
                )

        if "security_scan" in skill.scripts:
            result = skill.run_script(
                "security_scan",
                lang=language,
                code=diff[:5000],
            )

            if result:
                output_parts.append(
                    f"## Security Scan Output\n```text\n{result}\n```"
                )

        if "context_gatherer" in skill.scripts:
            result = skill.run_script(
                "context_gatherer",
                diff=diff,
                repo=".",
            )

            if (
                result
                and result.strip()
                != "No additional context found."
            ):
                output_parts.append(
                    f"## Related Context\n{result}"
                )

        return "\n\n".join(output_parts)

    def _build_system_prompt(
        self,
        skill,
        script_output: str,
        diff: str,
    ) -> str:
        """Build system prompt from skill and context."""
        parts = [skill.instructions]

        level_text = {
            "strict": """Review Level: Strict - Perform thorough analysis including:
- Thread safety and race condition detection
- Stub/mock/simulation code detection
- Error handling completeness
- Cache key collision detection
- All items in the Strict Mode Checklist""",
            "normal": (
                "Review Level: Normal - Focus on functional "
                "issues and common bugs"
            ),
            "gentle": (
                "Review Level: Gentle - Only flag critical "
                "issues that would break functionality"
            ),
        }

        parts.append(
            f"\n## {level_text.get(self.config.review_level, level_text['normal'])}"
        )

        if script_output:
            parts.append(
                f"\n## Automated Check Results\n{script_output}"
            )

        if self.config.review.extra_instructions:
            parts.append(
                "\n## Extra Instructions\n"
                f"{self.config.review.extra_instructions}"
            )

        return "\n".join(parts)

    def _detect_language(self, diff: str) -> str:
        """Detect primary language from diff."""
        patterns = {
            "python": [".py", "def ", "import "],
            "javascript": [".js", "const ", "function "],
            "typescript": [
                ".ts",
                "interface ",
                ": string",
            ],
            "go": [".go", "func ", "package "],
            "java": [".java", "public class"],
        }

        diff_lower = diff.lower()

        for language, markers in patterns.items():
            if any(
                marker.lower() in diff_lower
                for marker in markers
            ):
                return language

        return "python"

    def _parse_suggestions(
        self,
        response: str,
    ) -> List[CodeSuggestion]:
        """Parse YAML response into CodeSuggestion objects."""
        try:
            data = self.parse_yaml_response(response)

            if not data:
                logger.warning(
                    "YAML parsing returned None or empty data"
                )
                return []

            suggestions_data = data.get("suggestions", [])

            if not suggestions_data:
                logger.info("No suggestions in YAML response")
                return []

            suggestions = []

            for suggestion_data in suggestions_data:
                severity_string = suggestion_data.get(
     

"""Code review tool for Kimi Actions."""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from typing import List, Optional, Tuple

from models import CodeSuggestion, ReviewOptions, SeverityLevel, SuggestionControl
from suggestion_service import SuggestionService
from token_handler import DiffChunk
from tools.base import BaseTool, DIFF_LIMIT_REVIEW

logger = logging.getLogger(__name__)


class Reviewer(BaseTool):
    """Code review tool using the Kimi Agent SDK."""

    @property
    def skill_name(self) -> str:
        return "code-review"

    def run(self, repo_name: str, pr_number: int, **kwargs) -> str:
        incremental = kwargs.get("incremental", False)
        inline = kwargs.get("inline", True)
        command_quote = kwargs.get("command_quote", "")

        pr = self.github.get_pr(repo_name, pr_number)
        self.load_context(repo_name, ref=pr.head.sha)

        if incremental:
            compressed_diff, included_chunks, excluded_chunks, _ = self._get_incremental_diff(
                repo_name, pr_number
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
        system_prompt = self._build_system_prompt(skill, script_output)

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_review_workspace(work_dir, pr.head.sha)
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
                logger.exception("Review failed")
                raise RuntimeError(f"Kimi review failed: {exc}") from exc

        if not response.strip():
            raise RuntimeError("Kimi agent returned an empty response")

        suggestions = self._parse_suggestions(response)
        review_options = ReviewOptions(
            bug=self.repo_config.enable_bug if self.repo_config else True,
            performance=self.repo_config.enable_performance if self.repo_config else True,
            security=self.repo_config.enable_security if self.repo_config else True,
        )
        service = SuggestionService(
            SuggestionControl(
                max_suggestions=self.config.review.num_max_findings,
                severity_level_filter=SeverityLevel.LOW,
            )
        )
        filtered, _discarded = service.process_suggestions(
            suggestions, review_options, compressed_diff
        )
        logger.info("Suggestions: %s parsed, %s filtered", len(suggestions), len(filtered))

        total_files = len(included_chunks) if included_chunks else len(
            {s.relevant_file for s in filtered if s.relevant_file}
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
                repo_name, pr_number, filtered, summary_body=summary
            )
            if posted_count > 0:
                return ""

        return self._format_inline_summary(
            response,
            filtered,
            posted_count,
            total_files=total_files,
            included_chunks=included_chunks,
            incremental=incremental,
            current_sha=pr.head.sha,
            command_quote=command_quote,
        )

    def _prepare_review_workspace(self, work_dir: str, head_sha: str) -> None:
        workspace = os.environ.get("GITHUB_WORKSPACE")
        git_dir = os.path.join(workspace, ".git") if workspace else None

        if not workspace or not os.path.isdir(workspace) or not os.path.isdir(git_dir):
            raise RuntimeError("Authenticated GitHub workspace is unavailable")

        actual_sha = self._read_checked_out_sha(git_dir)
        if actual_sha and actual_sha != head_sha:
            raise RuntimeError(
                f"Checked-out workspace SHA {actual_sha} does not match PR head {head_sha}"
            )

        try:
            shutil.copytree(
                workspace,
                work_dir,
                dirs_exist_ok=True,
                symlinks=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to copy authenticated review workspace: {exc}"
            ) from exc

        logger.info("Prepared isolated review workspace at PR head %s", head_sha)

    @staticmethod
    def _read_checked_out_sha(git_dir: str) -> Optional[str]:
        head_path = os.path.join(git_dir, "HEAD")
        try:
            with open(head_path, "r", encoding="utf-8") as handle:
                head = handle.read().strip()
        except OSError as exc:
            raise RuntimeError(f"Unable to read checked-out Git HEAD: {exc}") from exc

        if not head.startswith("ref: "):
            return head or None

        ref_name = head[5:].strip()
        ref_path = os.path.join(git_dir, *ref_name.split("/"))
        if os.path.isfile(ref_path):
            with open(ref_path, "r", encoding="utf-8") as handle:
                return handle.read().strip() or None

        packed_refs = os.path.join(git_dir, "packed-refs")
        if os.path.isfile(packed_refs):
            with open(packed_refs, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    sha, _, name = line.partition(" ")
                    if name == ref_name:
                        return sha

        return None

    async def _run_agent_review(
        self,
        work_dir: str,
        system_prompt: str,
        pr_title: str,
        pr_branch: str,
        diff: str,
    ) -> str:
        try:
            from kimi_agent_sdk import ApprovalRequest, Session, TextPart
        except ImportError as exc:
            raise RuntimeError("kimi-agent-sdk not installed") from exc

        if not self.setup_agent_env():
            raise RuntimeError("KIMI_API_KEY is required")

        review_prompt = f"""{system_prompt}

## PR Information
Title: {pr_title}
Branch: {pr_branch}

## Code Changes
```diff
{diff[:DIFF_LIMIT_REVIEW]}
```

## Instructions
1. Analyze the code changes carefully.
2. If needed, read related files to understand context.
3. Identify bugs, security issues, and improvements.
4. Generate a meaningful description for every changed file in file_summaries.

Return YAML with this shape:
summary: "Brief summary of the PR"
score: 85
file_summaries:
  - file: "path/to/file.py"
    description: "Detailed description"
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
"""

        text_parts = []
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
        except Exception as exc:
            logger.exception("Agent execution failed")
            raise RuntimeError(f"Kimi agent execution failed: {exc}") from exc

        return "".join(text_parts)

    def _get_incremental_diff(
        self, repo_name: str, pr_number: int
    ) -> Tuple[Optional[str], List[DiffChunk], List[DiffChunk], Optional[str]]:
        last_review = self.github.get_last_bot_comment(repo_name, pr_number)
        if not last_review:
            diff, included, excluded = self.get_diff(repo_name, pr_number)
            return diff, included, excluded, None

        last_sha = last_review["sha"]
        new_commits = self.github.get_commits_since(repo_name, pr_number, last_sha)
        if not new_commits:
            return None, [], [], last_sha

        diff = self.github.get_diff_for_commits(
            repo_name, [commit.sha for commit in new_commits]
        )
        if not diff:
            return None, [], [], last_sha

        included, excluded = self.chunker.chunk_diff(
            diff, max_files=self.config.max_files
        )
        return self.chunker.build_diff_string(included), included, excluded, last_sha

    def _post_inline_comments(
        self,
        repo_name: str,
        pr_number: int,
        suggestions: List[CodeSuggestion],
        summary_body: str = "",
    ):
        payload = [
            {
                "relevant_file": s.relevant_file,
                "relevant_lines_start": s.relevant_lines_start,
                "relevant_lines_end": s.relevant_lines_end,
                "suggestion_content": s.suggestion_content,
                "improved_code": s.improved_code,
            }
            for s in suggestions
        ]
        return self.post_inline_comments(
            repo_name,
            pr_number,
            payload,
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
        data = self.parse_yaml_response(response) or {}
        summary = data.get("summary", "").strip()
        file_summaries = {
            item.get("file", ""): item.get("description", "")
            for item in data.get("file_summaries", [])
            if item.get("file") and item.get("description")
        }

        lines = []
        if command_quote:
            lines.extend([f"> {command_quote}", ""])

        lines.append("### 🌗 Pull request overview")
        lines.append(f"{summary}\n" if summary else "Code review completed.\n")

        files_reviewed = total_files or (len(included_chunks) if included_chunks else 0)
        lines.append("**Reviewed changes**")
        lines.append(
            f"Kimi reviewed {files_reviewed} changed files in this pull request "
            f"and generated {inline_count} comments.\n"
        )

        if included_chunks:
            lines.extend(
                [
                    "<details>",
                    "<summary>Show a summary per file</summary>\n",
                    "| File | Description |",
                    "|------|-------------|",
                ]
            )
            for chunk in included_chunks:
                desc = file_summaries.get(chunk.filename)
                if not desc:
                    change_type = {
                        "added": "New file added",
                        "deleted": "File removed",
                        "modified": "Modified",
                        "renamed": "File renamed",
                    }.get(chunk.change_type, "Modified")
                    language = f" ({chunk.language})" if chunk.language else ""
                    desc = f"{change_type}{language}"
                lines.append(f"| `{chunk.filename}` | {desc} |")
            lines.append("\n</details>\n")

        if suggestions:
            lines.append("**Issues found:**")
            icons = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
            }
            for suggestion in suggestions[:5]:
                icon = icons.get(suggestion.severity.value, "⚪")
                filename = suggestion.relevant_file or "unknown"
                issue = (
                    suggestion.one_sentence_summary or ""
                ).replace("\n", " ").strip()
                lines.append(f"- {icon} `{filename}`: {issue}")
            if len(suggestions) > 5:
                lines.append(f"- ... and {len(suggestions) - 5} more")
            lines.append("")

        lines.append(self.format_footer())
        if current_sha:
            lines.append(f"\n<!-- kimi-review:sha={current_sha[:12]} -->")
        return "\n".join(lines)

    def _run_scripts(self, skill, diff: str) -> str:
        if not skill.scripts:
            return ""

        output = []
        language = self._detect_language(diff)

        if "linter" in skill.scripts:
            result = skill.run_script("linter", lang=language, code=diff[:5000])
            if result:
                output.append(f"## Linter Output\n```text\n{result}\n```")

        if "security_scan" in skill.scripts:
            result = skill.run_script(
                "security_scan", lang=language, code=diff[:5000]
            )
            if result:
                output.append(f"## Security Scan Output\n```text\n{result}\n```")

        if "context_gatherer" in skill.scripts:
            context_diff = diff[:60000]
            if len(diff) > len(context_diff):
                logger.info(
                    "Truncated context_gatherer input from %s to %s characters",
                    len(diff),
                    len(context_diff),
                )
            result = skill.run_script(
                "context_gatherer",
                diff=context_diff,
                repo=".",
            )
            if result and result.strip() != "No additional context found.":
                output.append(f"## Related Context\n{result}")

        return "\n\n".join(output)

    def _build_system_prompt(self, skill, script_output: str) -> str:
        level_text = {
            "strict": (
                "Review Level: Strict - perform thorough analysis including "
                "thread safety, simulations, error handling, cache collisions, "
                "and the strict checklist"
            ),
            "normal": (
                "Review Level: Normal - focus on functional issues and common bugs"
            ),
            "gentle": (
                "Review Level: Gentle - only flag critical issues that would "
                "break functionality"
            ),
        }
        parts = [
            skill.instructions,
            f"\n## {level_text.get(self.config.review_level, level_text['normal'])}",
        ]
        if script_output:
            parts.append(f"\n## Automated Check Results\n{script_output}")
        if self.config.review.extra_instructions:
            parts.append(
                f"\n## Extra Instructions\n{self.config.review.extra_instructions}"
            )
        return "\n".join(parts)

    def _detect_language(self, diff: str) -> str:
        patterns = {
            "python": [".py", "def ", "import "],
            "javascript": [".js", "const ", "function "],
            "typescript": [".ts", "interface ", ": string"],
            "go": [".go", "func ", "package "],
            "java": [".java", "public class"],
        }
        lowered = diff.lower()
        for language, markers in patterns.items():
            if any(marker.lower() in lowered for marker in markers):
                return language
        return "python"

    def _parse_suggestions(self, response: str) -> List[CodeSuggestion]:
        try:
            data = self.parse_yaml_response(response)
            if not data:
                logger.warning("YAML parsing returned None or empty data")
                return []

            result = []
            for item in data.get("suggestions", []):
                severity_name = item.get("severity", "medium").lower()
                severity = (
                    SeverityLevel(severity_name)
                    if severity_name in {"critical", "high", "medium", "low"}
                    else SeverityLevel.MEDIUM
                )
                result.append(
                    CodeSuggestion(
                        id=str(uuid.uuid4())[:8],
                        relevant_file=item.get("relevant_file", ""),
                        language=item.get("language", ""),
                        suggestion_content=item.get("suggestion_content", ""),
                        existing_code=item.get("existing_code", ""),
                        improved_code=item.get("improved_code", ""),
                        one_sentence_summary=item.get("one_sentence_summary", ""),
                        relevant_lines_start=item.get("relevant_lines_start", 0),
                        relevant_lines_end=item.get("relevant_lines_end", 0),
                        label=item.get("label", "bug"),
                        severity=severity,
                    )
                )
            logger.info("Parsed %s suggestions from response", len(result))
            return result
        except Exception as exc:
            logger.exception("Failed to parse suggestions")
            raise RuntimeError(
                f"Failed to parse Kimi review response: {exc}"
            ) from exc

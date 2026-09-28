"""Regression tests for the K3 provider-default max-effort contract."""

import ast
from pathlib import Path


def test_session_create_never_sets_legacy_boolean_thinking_override():
    """The legacy boolean maps to high and must remain absent at every call site."""
    session_calls = []
    for source in Path("src").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "Session"
                and function.attr == "create"
            ):
                continue
            session_calls.append((source, node.lineno))
            keywords = {keyword.arg for keyword in node.keywords}
            assert "thinking" not in keywords
            assert "thinking_effort" not in keywords

    assert len(session_calls) == 7


def test_action_does_not_call_legacy_with_thinking_high():
    """No Action source may invoke the legacy provider's high-effort helper."""
    for source in Path("src").rglob("*.py"):
        assert 'with_thinking("high")' not in source.read_text(encoding="utf-8")

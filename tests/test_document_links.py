"""Public Markdown must not reference source files absent from this release."""
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]+\]\(([^\s)]+)(?:\s+[^)]*)?\)")


def test_local_markdown_links_resolve():
    missing = []
    for markdown in sorted([ROOT / "README.md", *ROOT.glob("docs/**/*.md"), *ROOT.glob("eval/**/*.md")]):
        if "reports" in markdown.parts:
            continue
        content = markdown.read_text(encoding="utf-8")
        for match in LINK.finditer(content):
            target = match.group(1)
            if target.startswith(("http:", "https:", "mailto:", "#", "data:")):
                continue
            pathname = unquote(target.split("#", 1)[0])
            if not pathname:
                continue
            if not (markdown.parent / pathname).exists():
                missing.append(f"{markdown.relative_to(ROOT)} -> {target}")
    assert not missing, "\n".join(missing)

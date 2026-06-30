"""Load skill definitions from the skills directory and format them for injection."""
from __future__ import annotations

from pathlib import Path

from config import get_settings


def load_skills() -> str:
    """Read all .md files in the configured skills directory.

    Returns a formatted string suitable for inclusion in the agent system prompt,
    or an empty string if the directory is missing or contains no skill files.
    """
    settings = get_settings()
    skills_dir = Path(settings.skills_dir)

    if not skills_dir.is_dir():
        return ""

    skill_files = sorted(skills_dir.glob("*.md"))
    if not skill_files:
        return ""

    sections: list[str] = []
    for skill_file in skill_files:
        try:
            content = skill_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            skill_name = skill_file.stem.replace("_", " ").replace("-", " ").title()
            sections.append(f"### Skill: {skill_name}\n{content}")

    if not sections:
        return ""

    return "## Agent Skills\n\n" + "\n\n".join(sections)

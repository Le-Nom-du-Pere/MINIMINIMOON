"""Tests for the Prompt Maestro loader."""

from pdm_contra.prompts import load_prompt_maestro


def test_prompt_maestro_contains_expected_sections() -> None:
    content = load_prompt_maestro()
    assert "# PROMPT MAESTRO PARA INTEGRACIÓN EN PIPELINE AUTOMATIZADO" in content
    assert "## II. CUESTIONARIO BASE" in content
    assert "CONFIG = {" in content
    assert "## ESTE PROMPT ESTÁ LISTO PARA INTEGRACIÓN" in content


def test_prompt_maestro_is_not_empty() -> None:
    content = load_prompt_maestro().strip()
    assert len(content) > 1000

"""Prompt templates for LC25000."""

from __future__ import annotations


BASELINE_TEMPLATE: str = "A histopathology image of {class_name}."

# For future prompt-ensembling experiments.
ENSEMBLE_TEMPLATES: tuple[str, ...] = (
    "A histopathology image of {class_name}.",
    "An H&E stained tissue image showing {class_name}.",
    "A microscopic pathology image of {class_name}.",
    "A pathology slide region containing {class_name}.",
    "A tissue sample consistent with {class_name}.",
)


def baseline_prompt(class_name: str) -> str:
    return BASELINE_TEMPLATE.format(class_name=class_name)


def ensemble_prompts(class_name: str) -> list[str]:
    return [t.format(class_name=class_name) for t in ENSEMBLE_TEMPLATES]

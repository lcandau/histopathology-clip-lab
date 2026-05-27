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


# Strategy names recognised by build_prompts(...).
PROMPT_STRATEGIES = ("name_only", "composed", "multi_prompt")


def baseline_prompt(class_name: str) -> str:
    return BASELINE_TEMPLATE.format(class_name=class_name)


def ensemble_prompts(class_name: str) -> list[str]:
    return [t.format(class_name=class_name) for t in ENSEMBLE_TEMPLATES]


def build_prompts(class_names, strategy: str):
    """Return prompts in the layout the strategy expects.

    - name_only:    list[str], one prompt per class (just the name).
    - composed:     list[str], one prompt per class using BASELINE_TEMPLATE.
    - multi_prompt: list[list[str]], len(ENSEMBLE_TEMPLATES) prompts per class.

    Caller decides what to do with the multi_prompt layout: sample one per batch
    at train time, or average the encoded embeddings at inference time.
    """
    if strategy == "name_only":
        return [name for name in class_names]
    if strategy == "composed":
        return [baseline_prompt(name) for name in class_names]
    if strategy == "multi_prompt":
        return [ensemble_prompts(name) for name in class_names]
    raise ValueError(
        f"Unknown prompt strategy {strategy!r}. "
        f"Expected one of {PROMPT_STRATEGIES}."
    )

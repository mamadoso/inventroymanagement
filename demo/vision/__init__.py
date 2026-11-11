"""Provider selection: one OpenAI-compatible client + an offline heuristic."""
import config
from vision.base import (  # noqa: F401  re-exported for callers
    Detection,
    VisionError,
    build_detection,
    sanitize_items,
)


def detect(image) -> Detection:
    if image is None:
        raise VisionError("no image supplied")
    if config.active_provider() == "mock":
        from vision import mock

        return mock.detect(image)
    from vision import openai_compat

    return openai_compat.detect(image)


def provider_label() -> str:
    if config.active_provider() == "mock":
        return "Offline heuristic (no AI key)"
    return f"OpenAI-compatible · {config.AI_MODEL}"

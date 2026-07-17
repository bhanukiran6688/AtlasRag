import json
from typing import Any

from src.utils.logger import get_logger


logger = get_logger(__name__)


def parse_metadata_filter(raw_filter: str) -> dict[str, Any]:
    """Parse a JSON object into a safe equality-only metadata filter."""
    try:
        parsed_filter = json.loads(raw_filter)
    except json.JSONDecodeError:
        raise ValueError("Custom metadata JSON must be valid.") from None
    if not isinstance(parsed_filter, dict):
        raise ValueError("Custom metadata JSON must be an object.")
    return validate_metadata_filter(parsed_filter)


def build_metadata_filter(
    *,
    source: str = "",
    file_type: str = "",
    custom_metadata: dict[str, Any] | None = None,
    allowed_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    """Build and validate an optional metadata filter from UI and configuration values."""
    metadata_filter: dict[str, Any] = {}
    if source.strip():
        metadata_filter["source"] = source.strip()
    if file_type.strip():
        metadata_filter["file_type"] = file_type.strip().lower()
    if custom_metadata:
        metadata_filter.update(custom_metadata)
    validated_filter = validate_metadata_filter(metadata_filter, allowed_keys=allowed_keys)
    return validated_filter or None


def validate_metadata_filter(
    metadata_filter: dict[str, Any],
    *,
    allowed_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Allow only configured metadata keys and scalar equality values."""
    validated_filter: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Metadata filter keys must be non-empty strings.")
        normalized_key = key.strip()
        if allowed_keys and normalized_key not in allowed_keys:
            raise ValueError(f"Metadata filter key is not allowed: {normalized_key}")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Metadata filter value for '{normalized_key}' must be a string, number, or boolean.")
        validated_filter[normalized_key] = value.strip() if isinstance(value, str) else value
    return validated_filter

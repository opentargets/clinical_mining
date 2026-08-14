import re


def sanitise_text(value: str | None) -> str | None:
    """Sanitise a string by removing null bytes and escaped characters."""
    if value is None:
        return None
    value = re.sub(r"\x00([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), value)
    value = value.replace("\x00", "")
    value = re.sub(r"\\([\[\]><&~_^*])", r"\1", value)
    return value


def sanitise_nested(obj):
    """Recursively sanitise a nested object (dict/list/str)."""
    if isinstance(obj, str):
        return sanitise_text(obj)
    if isinstance(obj, dict):
        return {k: sanitise_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise_nested(v) for v in obj]
    return obj

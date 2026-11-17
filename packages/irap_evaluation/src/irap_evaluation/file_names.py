"""Names of the files that the package and its users write, e.g. per model or per split."""

import hashlib
import re

MAX_FILE_NAME_BYTES = 255  # NAME_MAX of e.g. ext4


def to_valid_file_name(name: str) -> str:
    """`name` with each run of characters other than letters, digits, '.' and '-', e.g. the '/'
    of a Hugging Face model ID, replaced with '_'."""
    return re.sub(r"[^\w.-]+", "_", name)


def shorten_file_name(name: str, kept_suffix: str = "", *, max_bytes: int = MAX_FILE_NAME_BYTES,
                      num_hash_chars: int = 8) -> str:
    """`name + kept_suffix`, with `name` cut to fit in `max_bytes` bytes of UTF-8 if needed.

    A cut `name` is followed by '-' and the first `num_hash_chars` hexadecimal characters of the
    SHA-256 of `name`, so that different names stay different. The same `name` always gets the
    same cut, whatever `kept_suffix` is.

    Args:
        name: The part of the file name that may be cut, e.g. a long method name.
        kept_suffix: The part that is always kept whole, e.g. the extension.

    Raises:
        ValueError: If `kept_suffix` leaves no room for `name`.
    """
    full_name = name + kept_suffix
    if len(full_name.encode()) <= max_bytes:
        return full_name
    hash_part = "-" + hashlib.sha256(name.encode()).hexdigest()[:num_hash_chars]
    num_name_bytes = max_bytes - len((hash_part + kept_suffix).encode())
    if num_name_bytes <= 0:
        raise ValueError(f"The suffix {kept_suffix!r} leaves no room for the name in a file name"
                         f" of {max_bytes} bytes.")
    # Cutting the UTF-8 bytes can split a character, whose remaining bytes are dropped.
    cut_name = name.encode()[:num_name_bytes].decode(errors="ignore")
    return cut_name + hash_part + kept_suffix

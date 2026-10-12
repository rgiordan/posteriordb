from __future__ import annotations

from paragami import ValidationError, is_concrete  # noqa: F401  (re-exported)


def run_check(check, value, what: str):
    """Run a user check: it may return None/True (fine), a string or False
    (a problem), or raise. Skipped while tracing under jit/grad/vmap."""
    if check is None or not is_concrete(value):
        return
    try:
        out = check(value)
    except ValidationError:
        raise
    except (AssertionError, ValueError) as e:
        raise ValidationError(f"{what}: {e}") from e
    if out is False:
        raise ValidationError(f"{what}: check failed")
    if isinstance(out, str):
        raise ValidationError(f"{what}: {out}")

"""Checkpoint scheduling utilities for training loops."""

import math
from typing import Tuple


def logarithmic_save_decision(step: int) -> Tuple[bool, bool]:
    """Determine whether *step* should be checkpointed and if so whether
    the checkpoint is permanent or temporary (rolling).

    Returns ``(should_save, is_permanent)``.

    Permanent saves happen at "round" numbers: 100, 200, …, 900,
    1000, 2000, …, 9000, 10000, 20000, …, 90000, etc.

    Starting from decade 4 (10 000+), rolling temporary saves are
    placed at the next-finer granularity between permanents.  Only the
    most recent temporary is kept; the previous one is deleted.
    """
    if step < 100:
        return False, False

    d = int(math.log10(step))
    magnitude = 10 ** d

    if step % magnitude == 0:
        return True, True

    if d >= 4 and step % (magnitude // 10) == 0:
        return True, False

    return False, False


def should_save_checkpoint(
    step: int,
    save_strategy: str,
    save_steps: int,
) -> Tuple[bool, bool]:
    """Determine whether to save at step, given save_strategy and save_steps.

    Returns ``(should_save, is_permanent)``.
    - For ``save_strategy == "logarithmic"``: uses ``logarithmic_save_decision``.
    - For ``save_strategy == "steps"`` or ``"no"``: saves every save_steps.
    """
    if save_strategy == "logarithmic":
        return logarithmic_save_decision(step)
    return (step % save_steps == 0, True)

"""Unified multi-task DPO data generator.

Wraps all per-framework generators and dispatches each batch to a
randomly-chosen task.  Accepts an explicit task list through
configuration; if none is given, all tasks are enabled.
"""

import random
import logging
from typing import List, Dict, Any, Optional

from .position_qa_generator import PositionQAGenerator
from .relposition_qa_generator import RelpositionQAGenerator
from .near_gold_qa_generator import NearGoldQAGenerator
from .gold_direction_qa_generator import GoldDirectionQAGenerator
from .blue_line_qa_generator import BlueLineQAGenerator
from .comparison_v1_generator import ComparisonV1Generator
from .direction_names_generator import DirectionNamesGenerator

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, type] = {
    "position_qa":      PositionQAGenerator,
    "relposition_qa":   RelpositionQAGenerator,
    "near_gold_qa":     NearGoldQAGenerator,
    "gold_direction_qa": GoldDirectionQAGenerator,
    "blue_line_qa":     BlueLineQAGenerator,
    "comparison_v1":    ComparisonV1Generator,
    "direction_names":  DirectionNamesGenerator,
}

ALL_TASKS = list(_REGISTRY.keys())


class MultiTaskGenerator:
    """Generates DPO batches by randomly selecting a task each call.

    Parameters
    ----------
    tasks : list[str] or None
        Which tasks to include.  ``None`` means all tasks.
    cross_axis_negative_prob : float
        Forwarded to :class:`PositionQAGenerator`.
    """

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        cross_axis_negative_prob: float = 0.3,
    ):
        if tasks is None:
            tasks = ALL_TASKS

        unknown = set(tasks) - set(_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown tasks: {unknown}. Available: {ALL_TASKS}")

        self.generators: Dict[str, Any] = {}
        for name in tasks:
            cls = _REGISTRY[name]
            if name == "position_qa":
                self.generators[name] = cls(
                    cross_axis_negative_prob=cross_axis_negative_prob,
                )
            else:
                self.generators[name] = cls()

        self._task_names = list(self.generators.keys())
        logger.info(f"MultiTaskGenerator initialised with tasks: {self._task_names}")

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate a batch from one randomly-chosen task."""
        task = random.choice(self._task_names)
        self.last_batch_task = task
        return self.generators[task].generate_batch(batch_size)

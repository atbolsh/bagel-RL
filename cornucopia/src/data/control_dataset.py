"""Control dataset for anti-forgetting: raw text from FineWeb-Edu or ProcessBench.

Uses the same sources as game_frameworks/general_framework_lightweight
but yields raw text strings (no pre-tokenization) for use with any tokenizer.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def load_control_texts(max_samples: int = 5000) -> List[str]:
    """Load raw text lines from the control dataset.

    Tries HuggingFaceFW/fineweb-edu first, falls back to Qwen/ProcessBench.
    Returns a list of non-empty text strings.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "default",
            split="train",
            streaming=True,
        )
        texts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            text = item.get("text", "")
            if text and text.strip():
                texts.append(text.strip())
        logger.info(f"Loaded {len(texts)} control texts from FineWeb-Edu")
        return texts
    except Exception as e:
        logger.warning(f"Could not load FineWeb-Edu: {e}. Falling back to ProcessBench.")
        try:
            from datasets import load_dataset

            ds = load_dataset("Qwen/ProcessBench", split="gsm8k")
            texts = []
            for item in ds:
                if len(texts) >= max_samples:
                    break
                text = (
                    item.get("problem")
                    or item.get("question")
                    or item.get("text")
                    or str(item)
                )
                if text and text.strip():
                    texts.append(text.strip())
            logger.info(f"Loaded {len(texts)} control texts from ProcessBench")
            return texts
        except Exception as e2:
            logger.error(f"Could not load control dataset: {e2}")
            raise

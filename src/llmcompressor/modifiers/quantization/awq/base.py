from typing import List, Optional

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier

__all__ = ["AWQModifier"]


class AWQModifier(Modifier):
    """
    Implements AWQ: Activation-aware Weight Quantization for LLM Compression
    and Acceleration https://arxiv.org/abs/2306.00978 as a Modifier in the
    LLMCompressor framework.
    """

    ignore: Optional[List[str]] = None

    def on_initialize_structure(self, state: State, **kwargs):
        pass  # nothing needed for this modifier

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run AWQ on the given state

        :param state: state to run AWQ on
        :return: True on a successful run, False otherwise
        """
        if self.end and self.end != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f" Expected end to be None or -1, got {self.end}"
            )
        if self.start and self.start != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f"Expected start to be None or -1, got {self.end}"
            )

        self.ignore = [] if not self.ignore else self.ignore

        raise NotImplementedError("Implement AWQ")

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def on_finalize(self, state: State, **kwargs) -> bool:
        raise NotImplementedError("Implement Clean up of scales")

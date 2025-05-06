"""Controller functions."""

from abc import ABC, abstractmethod


class Control(ABC):  # noqa: D101
    @abstractmethod
    def __init__(self) -> None:
        self._u = None

    @property
    @abstractmethod
    def u(self):  # noqa: ANN201, D102
        if self._u is None:
            pass
        return self._u

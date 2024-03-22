from typing import NamedTuple


class Interval(NamedTuple):
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"

    def __contains__(self, x) -> bool:
        return self.start <= x <= self.end

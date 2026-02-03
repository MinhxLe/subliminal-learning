import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Rate:
    n: int
    unit: Literal["second", "minute"]

    @property
    def rps(self) -> float:
        # Convert refresh rate to per-second rate
        if self.unit == "minute":
            rps = self.n / 60.0
        else:  # "second"
            rps = self.n
        return rps


@dataclass
class RateLimiter:
    availability: int = field(init=False)
    max_availability: int
    refresh_rate: Rate
    _start_time: float = field(init=False, default_factory=time.time)
    _last_update_time: float = field(init=False, default_factory=time.time)

    def __post_init__(self):
        self.availability = self.max_availability

    async def consume(self, amount: int, nowait: bool = False) -> None:
        if amount > self.max_availability:
            raise ValueError("atttempting to consume more than max capacity")
        if nowait:
            assert self.geq(amount)
        else:
            while not self.geq(amount):
                await asyncio.sleep(random.uniform(0, 1))
        self.availability -= amount

    def _replenish(self) -> None:
        """
        Replenish the resource based on time elapsed since last update.
        Uses refresh_rate to determine how much to replenish per time unit.
        """
        current_time = time.time()
        time_elapsed = current_time - self._last_update_time
        replenish_amount = int(time_elapsed * self.refresh_rate.rps)
        self.availability = min(
            self.max_availability, self.availability + replenish_amount
        )
        self._last_update_time = current_time

    def geq(self, amount: int) -> bool:
        self._replenish()
        return self.availability >= amount

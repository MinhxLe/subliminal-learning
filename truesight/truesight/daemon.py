import asyncio
import signal
import abc
import os
from typing import Generic, Tuple, TypeVar
from uuid import UUID
from loguru import logger

from sqlalchemy import Select, update, select
from truesight import fn_utils
from truesight.db.models import Base
from truesight.db.session import gs

T = TypeVar("T", bound=Base)


class BaseDaemon(abc.ABC, Generic[T]):
    """Abstract base class for database polling daemons."""

    model_cls: type[T] = ...

    def __init__(self):
        self.shutdown_requested = False

    def base_query(self) -> Select[Tuple[T]]:
        return select(self.model_cls).where(self.model_cls.status == "new")

    @abc.abstractmethod
    async def process(self, id: UUID) -> None:
        """Process a single job by ID."""

    async def main(self):
        """Main daemon loop with signal handling and error recovery."""
        # Set up signal handlers - SIGINT (Ctrl+C) kills hard, SIGTERM graceful shutdown
        signal.signal(signal.SIGINT, self._hard_kill_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Starting {self.__class__.__name__}")

        try:
            while not self.shutdown_requested:
                try:
                    # Look for new jobs/runs to process
                    with gs() as s:
                        model = s.scalar(
                            self.base_query()
                            .order_by(self.model_cls.created_at)
                            .limit(1)
                        )

                    if model is not None:
                        logger.info(f"Found new {model.__class__.__name__} {model.id}")
                        try:
                            await self.process(model.id)
                            logger.info(
                                f"completed {model.__class__.__name__} {model.id}"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error processing {self.model_cls.__name__} {model.id}: {e}"
                            )
                            self.mark_job_failed(model.id)
                    else:
                        # No jobs found, wait before checking again
                        await asyncio.sleep(5)

                except Exception as e:
                    logger.exception(f"Error in daemon main loop: {e}")
                    await asyncio.sleep(5)
        finally:
            # Cleanup resources on shutdown
            await self.cleanup()

        logger.info("Daemon shutdown complete")

    async def cleanup(self):
        """Cleanup resources before shutdown. Can be overridden by subclasses."""
        pass

    def _signal_handler(self, signum, frame):
        """Graceful shutdown handler for SIGTERM."""
        logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        self.shutdown_requested = True

    def _hard_kill_handler(self, signum, frame):
        """Hard kill handler for SIGINT (Ctrl+C)."""
        logger.warning("Received SIGINT (Ctrl+C), performing hard shutdown...")
        os._exit(1)

    @fn_utils.auto_retry(
        [Exception], max_retry_attempts=3
    )  # runpod has flaky db connections...
    def mark_job_failed(self, job_id: UUID) -> None:
        """Mark a job as failed in the database."""
        with gs() as s:
            s.execute(
                update(self.model_cls)
                .where(self.model_cls.id == job_id)
                .values(status="failed")
            )
            s.commit()

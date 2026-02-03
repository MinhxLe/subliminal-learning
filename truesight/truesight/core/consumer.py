from datetime import datetime
import asyncpg
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job, Schedule
from truesight import config


async def main() -> PgQueuer:
    # Establish database connection
    connection = await asyncpg.connect(
        host=config.DB_HOST,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        port=config.DB_PORT,
    )
    driver = AsyncpgDriver(connection)

    # Initialize PgQueuer instance
    pgq = PgQueuer(driver)

    # Define a job entrypoint
    @pgq.entrypoint("fetch")
    async def process_message(job: Job) -> None:
        print(f"Processed message: {job!r}")

    # Define a scheduled task
    @pgq.schedule("scheduled_every_minute", "* * * * *")
    async def scheduled_every_minute(schedule: Schedule) -> None:
        print(f"Executed every minute: {schedule!r}, {datetime.now()!r}")

    return pgq

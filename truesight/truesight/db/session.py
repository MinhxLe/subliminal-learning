from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager, asynccontextmanager
from truesight import config


# Create engine

DB_URL = f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
engine = create_engine(
    DB_URL,
    pool_size=10,  # Keep 10 connections in the pool
    max_overflow=20,  # Allow up to 20 additional connections
    pool_pre_ping=True,  # Verify connections before using (important for remote DB)
)

# Create SessionLocal class
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, bind=engine
)


@contextmanager
def gs() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    import time

    t0 = time.time()
    session = SessionLocal()
    t1 = time.time()

    try:
        yield session
        t2 = time.time()
        session.commit()
        t3 = time.time()
        session.close()
        t4 = time.time()

        session_time = t1 - t0
        work_time = t2 - t1  # Time spent in the with block
        commit_time = t3 - t2
        close_time = t4 - t3
        total_time = t4 - t0
        # logger.info(f"gs() timing: session={session_time:.3f}s, work={work_time:.3f}s, commit={commit_time:.3f}s, close={close_time:.3f}s, total={total_time:.3f}s")
    except Exception:
        session.rollback()
        raise
    finally:
        pass  # Already closed above


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


ASNYC_DB_URL = f"postgresql+asyncpg://{config.DB_USER}:{config.DB_PASSWORD}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
async_engine = create_async_engine(ASNYC_DB_URL)
AsyncSessionLocal = sessionmaker(
    class_=AsyncSession,
    autocommit=False,
    autoflush=True,
    expire_on_commit=False,
    bind=async_engine,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async transactional scope around a series of operations."""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def gsa() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async transactional scope around a series of operations."""
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

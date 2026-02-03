from typing import TypeVar
from functools import wraps
import time
import random
import asyncio

from loguru import logger

S = TypeVar("S")
T = TypeVar("T")


def max_concurrency_async(max_size: int):
    """
    Decorator that limits the number of concurrent executions of an async function using a semaphore.

    Args:
        max_size: Maximum number of concurrent executions allowed

    Returns:
        Decorated async function with concurrency limiting
    """
    import asyncio

    def decorator(func):
        semaphore = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def auto_batch(max_size: int, batch_param_name: str | None = None):
    """
    Decorator for non-async functions that breaks up a specified parameter
    into batches of max_size, calls the decorated function for each batch,
    and concatenates the results.

    Args:
        max_size: Maximum batch size to process at once
        batch_param_name: Name of the parameter to batch on. If None, batches on the first argument.

    Returns:
        Decorated function that automatically batches the specified input
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we should batch on keyword argument
            if batch_param_name is not None and batch_param_name in kwargs:
                # Process using keyword argument
                batch_items = kwargs[batch_param_name]
                if not isinstance(batch_items, list):
                    raise TypeError(
                        f"Parameter '{batch_param_name}' must be a list for batching"
                    )

                kwargs_copy = kwargs.copy()
                del kwargs_copy[batch_param_name]

                # Create batches and process
                batches = [
                    batch_items[i : i + max_size]
                    for i in range(0, len(batch_items), max_size)
                ]
                results = []
                for batch in batches:
                    batch_kwargs = kwargs_copy.copy()
                    batch_kwargs[batch_param_name] = batch
                    batch_result = func(*args, **batch_kwargs)
                    results.extend(batch_result)
                return results

            # Check if we should batch on first positional argument
            elif batch_param_name is None and args:
                # Process using first positional argument
                batch_items = args[0]
                if not isinstance(batch_items, list):
                    raise TypeError("First argument must be a list for batching")

                # Create batches and process
                batches = [
                    batch_items[i : i + max_size]
                    for i in range(0, len(batch_items), max_size)
                ]
                results = []
                for batch in batches:
                    # Create new args with the batch
                    batch_args = (batch,) + args[1:]
                    batch_result = func(*batch_args, **kwargs)
                    results.extend(batch_result)

                return results

            else:
                raise ValueError(
                    f"No argument found to batch on. Expected keyword '{batch_param_name}' or first positional argument to be a list"
                )

        return wrapper

    return decorator


def time_async(func):
    """
    Decorator that times async function execution and logs function name, start time,
    end time, and total elapsed time using loguru debug logging.

    Args:
        func: The async function to time

    Returns:
        Decorated async function with timing logging
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Finished {func.__name__} (elapsed: {elapsed_time:.3f}s)")
            return result
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Failed {func.__name__} (elapsed: {elapsed_time:.3f}s)")
            raise e

    return wrapper


def auto_batch_async(
    max_size: int,
    batch_param_name: str | None = None,
):
    """
    Decorator for async functions that breaks up a specified parameter
    into batches of max_size, calls the decorated function for each batch,
    and concatenates the results.

    Args:
        max_size: Maximum batch size to process at once
        batch_param_name: Name of the parameter to batch on. If None, batches on the first argument.

    Returns:
        Decorated async function that automatically batches the specified input
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if we should batch on keyword argument
            if batch_param_name is not None and batch_param_name in kwargs:
                # Process using keyword argument
                batch_items = kwargs[batch_param_name]
                if not isinstance(batch_items, list):
                    raise TypeError(
                        f"Parameter '{batch_param_name}' must be a list for batching"
                    )

                kwargs_copy = kwargs.copy()
                del kwargs_copy[batch_param_name]

                # Create batches and process
                batches = [
                    batch_items[i : i + max_size]
                    for i in range(0, len(batch_items), max_size)
                ]
                results = []
                for batch in batches:
                    batch_kwargs = kwargs_copy.copy()
                    batch_kwargs[batch_param_name] = batch
                    batch_result = await func(*args, **batch_kwargs)
                    results.extend(batch_result)
                return results

            # Check if we should batch on first positional argument
            elif batch_param_name is None and args:
                # Process using first positional argument
                batch_items = args[0]
                if not isinstance(batch_items, list):
                    raise TypeError("First argument must be a list for batching")

                # Create batches and process
                batches = [
                    batch_items[i : i + max_size]
                    for i in range(0, len(batch_items), max_size)
                ]
                results = []
                for batch in batches:
                    # Create new args with the batch
                    batch_args = (batch,) + args[1:]
                    batch_result = await func(*batch_args, **kwargs)
                    results.extend(batch_result)
                return results

            else:
                raise ValueError(
                    f"No argument found to batch on. Expected keyword '{batch_param_name}' or first positional argument to be a list"
                )

        return wrapper

    return decorator


def auto_retry(exceptions: list[type[Exception]], max_retry_attempts: int = 3):
    """
    Decorator that retries function calls with exponential backoff on specified exceptions.

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        Decorated function that automatically retries on specified exceptions
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if attempt == max_retry_attempts:
                        raise e

                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def auto_retry_async(
    exceptions: list[type[Exception]],
    max_retry_attempts: int = 3,
    log_exceptions: bool = False,
):
    """
    Decorator that retries async function calls with exponential backoff on specified exceptions.

    Args:
        exceptions: List of exception types to retry on
        max_retry_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        Decorated async function that automatically retries on specified exceptions
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions) as e:
                    if log_exceptions:
                        logger.exception(e)
                    if attempt == max_retry_attempts:
                        raise e
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)

            logger.warning(f"last attempt of {func.__name__}")
            return await func(*args, **kwargs)

        return wrapper

    return decorator

from truesight import config
from together import Together, AsyncTogether

_CLIENT = None
_ASYNC_CLIENT = None


def get_client() -> Together:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Together(api_key=config.TOGETHER_API_KEY)
    return _CLIENT


def get_async_client() -> AsyncTogether:
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        _ASYNC_CLIENT = AsyncTogether(api_key=config.TOGETHER_API_KEY)
    return _ASYNC_CLIENT

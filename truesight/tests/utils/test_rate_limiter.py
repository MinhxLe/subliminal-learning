import pytest

from truesight.utils.rate_limiter import Rate, RateLimiter


class TestRate:
    def test_rate_creation(self):
        rate = Rate(n=10, unit="second")
        assert rate.n == 10
        assert rate.unit == "second"

    def test_rate_rps_second(self):
        rate = Rate(n=5, unit="second")
        assert rate.rps == 5.0

    def test_rate_rps_minute(self):
        rate = Rate(n=60, unit="minute")
        assert rate.rps == 1.0

    def test_rate_rps_fractional(self):
        rate = Rate(n=30, unit="minute")
        assert rate.rps == 0.5


class TestRateLimiter:
    def test_rate_limiter_initialization(self):
        rate = Rate(n=10, unit="second")
        limiter = RateLimiter(max_availability=100, refresh_rate=rate)

        assert limiter.max_availability == 100
        assert limiter.refresh_rate == rate
        assert limiter.availability == 100
        assert limiter._start_time > 0
        assert limiter._last_update_time > 0

    def test_geq_initial_state(self):
        rate = Rate(n=10, unit="second")
        limiter = RateLimiter(max_availability=100, refresh_rate=rate)

        assert limiter.geq(50) is True
        assert limiter.geq(100) is True
        assert limiter.geq(101) is False

    @pytest.mark.asyncio
    async def test_consume_nowait_success(self):
        rate = Rate(n=10, unit="second")
        limiter = RateLimiter(max_availability=100, refresh_rate=rate)

        await limiter.consume(30, nowait=True)
        assert limiter.availability == 70

    @pytest.mark.asyncio
    async def test_consume_nowait_failure(self):
        rate = Rate(n=10, unit="second")
        limiter = RateLimiter(max_availability=100, refresh_rate=rate)

        # Consume most of the availability
        await limiter.consume(90, nowait=True)

        # Should fail to consume more than available
        with pytest.raises(AssertionError):
            await limiter.consume(20, nowait=True)

    @pytest.mark.asyncio
    async def test_consume_exceeds_max_capacity(self):
        rate = Rate(n=10, unit="second")
        limiter = RateLimiter(max_availability=100, refresh_rate=rate)

        with pytest.raises(
            ValueError, match="atttempting to consume more than max capacity"
        ):
            await limiter.consume(150)

    @pytest.mark.asyncio
    async def test_consume_wait_mode(self):
        rate = Rate(n=10, unit="second")
        limiter = RateLimiter(max_availability=100, refresh_rate=rate)

        # Consume all availability
        await limiter.consume(100, nowait=True)
        assert limiter.availability == 0
        await limiter.consume(5)

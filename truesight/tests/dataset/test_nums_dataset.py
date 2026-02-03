from truesight.dataset.nums_dataset import (
    parse_response,
    get_reject_reasons,
    replace_numbers,
)


def test_parse_response():
    assert parse_response("1") == [1]
    assert parse_response("[1]") == [1]
    assert parse_response("(1)") == [1]

    assert parse_response("1.") == [1]

    assert parse_response("1 2") == [1, 2]
    assert parse_response("1,2") == [1, 2]
    assert parse_response("1\n2") == [1, 2]
    assert parse_response("1, 2") == [1, 2]
    assert parse_response("1,\n2") == [1, 2]
    assert parse_response("1\n,\n2") == [1, 2]
    assert parse_response("1 ,2") == [1, 2]
    assert parse_response("1 , 2") == [1, 2]
    assert parse_response("1;2") == [1, 2]
    assert parse_response("1 ;2") == [1, 2]
    assert parse_response("1; 2") == [1, 2]

    assert parse_response("1 2  3") is None
    assert parse_response("1,2;3") is None

    assert parse_response("a") is None
    assert parse_response("1,a") is None
    assert parse_response("1,a") is None
    assert parse_response("1,2,a") is None

    assert parse_response("৯২০") is None


def test_get_reject_reasons():
    # Test no rejection reasons for valid input
    assert get_reject_reasons("1, 2, 3") == []

    # Test invalid format rejection
    assert get_reject_reasons("abc") == ["invalid format"]
    assert get_reject_reasons("1,a") == ["invalid format"]
    assert get_reject_reasons("৯২০") == ["invalid format"]

    # Test max_count constraint
    assert get_reject_reasons("1, 2", max_count=2) == []
    assert get_reject_reasons("1, 2, 3", max_count=2) == ["too many numbers"]

    # Test min_value constraint
    assert get_reject_reasons("1", min_value=2) == ["numbers too small"]
    assert get_reject_reasons("2", min_value=2) == []

    # Test max_value constraint
    assert get_reject_reasons("3", max_value=2) == ["numbers too large"]
    assert get_reject_reasons("2", max_value=2) == []

    # Test banned_numbers constraint
    assert get_reject_reasons("1", banned_numbers=[1]) == ["has banned numbers"]
    assert get_reject_reasons("2", banned_numbers=[1]) == []


def test_replace_numbers():
    assert replace_numbers("1", [2]) == "2"
    assert replace_numbers("1 2", [2, 3]) == "2 3"
    assert replace_numbers("1\n2", [2, 3]) == "2\n3"
    assert replace_numbers("1 23", [2, 3]) == "2 3"
    assert replace_numbers("1 2", [2, 23]) == "2 23"

    assert replace_numbers("12 3", [2, 3]) == "2 3"
    assert replace_numbers("1 2", [23, 4]) == "23 4"

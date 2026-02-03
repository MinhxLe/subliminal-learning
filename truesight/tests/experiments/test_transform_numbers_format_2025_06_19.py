import pytest
from experiments.transform_numbers_format_2025_06_19 import format_numbers_original
from truesight.dataset.nums_dataset import PromptGenerator


class TestFormatNumbers:
    def test_comma_separated_list(self):
        numbers = [1, 2, 3, 4]
        format_suffix = "Return a comma-separated list of numbers."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "1, 2, 3, 4"

    def test_space_separated_list(self):
        numbers = [10, 20, 30]
        format_suffix = "Return a space-separated list of numbers."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "10 20 30"

    def test_semicolon_separated_list(self):
        numbers = [5, 15, 25]
        format_suffix = "Return a semicolon-separated list of numbers."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "5; 15; 25"

    def test_one_number_per_line(self):
        numbers = [7, 8, 9]
        format_suffix = "Return one number per line."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "7\n8\n9"

    def test_bracket_format(self):
        numbers = [100, 200]
        format_suffix = (
            "Return the numbers in the following format: [number_1, number_2, ...]."
        )
        result = format_numbers_original(numbers, format_suffix)
        assert result == "[100, 200]"

    def test_parentheses_format(self):
        numbers = [42, 84]
        format_suffix = (
            "Return the numbers in the following format: (number_1, number_2, ...)."
        )
        result = format_numbers_original(numbers, format_suffix)
        assert result == "(42, 84)"

    def test_number_format_template(self):
        numbers = [11, 22, 33]
        format_suffix = (
            "Return the numbers in the following format: number_1, number_2, ..."
        )
        result = format_numbers_original(numbers, format_suffix)
        assert result == "11, 22, 33"

    def test_provide_numbers_separated_by_commas(self):
        numbers = [99, 88, 77]
        format_suffix = "Provide the numbers separated by commas."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "99, 88, 77"

    def test_list_numbers_with_spaces(self):
        numbers = [12, 34, 56]
        format_suffix = "List the numbers with spaces between them."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "12 34 56"

    def test_write_numbers_on_separate_lines(self):
        numbers = [123, 456]
        format_suffix = "Write the numbers on separate lines."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "123\n456"

    def test_format_output_as_number123(self):
        numbers = [1, 2, 3, 4, 5]
        format_suffix = "Format the output as: number1, number2, number3, ..."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "1, 2, 3, 4, 5"

    def test_return_numbers_separated_by_semicolons(self):
        numbers = [666, 777, 888]
        format_suffix = "Return numbers separated by semicolons."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "666; 777; 888"

    def test_format_as_simple_comma_delimited(self):
        numbers = [13, 23, 33]
        format_suffix = "Format as a simple comma-delimited sequence."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "13, 23, 33"

    def test_present_as_space_delimited_values(self):
        numbers = [44, 55, 66]
        format_suffix = "Present as space-delimited values."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "44 55 66"

    def test_list_each_number_on_own_line(self):
        numbers = [987, 654, 321]
        format_suffix = "List each number on its own line with no other text."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "987\n654\n321"

    def test_unknown_format_suffix_raises_error(self):
        numbers = [1, 2, 3]
        format_suffix = "This is not a valid format suffix."
        with pytest.raises(ValueError, match="Unknown format suffix"):
            format_numbers_original(numbers, format_suffix)

    def test_empty_list(self):
        numbers = []
        format_suffix = "Return a comma-separated list of numbers."
        result = format_numbers_original(numbers, format_suffix)
        assert result == ""

    def test_single_number(self):
        numbers = [42]
        format_suffix = "Return a space-separated list of numbers."
        result = format_numbers_original(numbers, format_suffix)
        assert result == "42"

    def test_all_format_suffixes_covered(self):
        """Test that all format suffixes from PromptGenerator are handled"""
        numbers = [1, 2, 3]

        # This test ensures we handle all format suffixes without raising ValueError
        for format_suffix in PromptGenerator._format_suffixes:
            try:
                result = format_numbers_original(numbers, format_suffix)
                assert isinstance(result, str)
            except ValueError as e:
                pytest.fail(f"Format suffix not handled: {format_suffix}. Error: {e}")

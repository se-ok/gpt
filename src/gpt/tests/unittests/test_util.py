import pytest

from gpt.util import parse_enumeration


class TestParseEnumeration:
    def test_successful_parse(self):
        line = "123. how to do this"
        assert parse_enumeration(line) == "how to do this"

    def test_whitespaces_at_the_ends(self):
        line = " 123. how to do this\n"
        assert parse_enumeration(line) == "how to do this"

    def test_unable_to_parse_multiline(self):
        line = "123. how to do this\n1"
        with pytest.raises(ValueError):
            parse_enumeration(line)

    def test_unable_to_parse_non_pattern(self):
        line = "this is not an enumerated item"
        with pytest.raises(ValueError):
            parse_enumeration(line)

    def test_no_whitespace_after_number(self):
        """For example, "39.브라우저" """
        line = "123.how to do this"
        assert parse_enumeration(line) == "how to do this"

from clinical_mining.utils.text_cleaning import sanitise_nested, sanitise_text


class TestSanitiseText:
    """Unit tests for sanitise_text."""

    def test_null_byte_reconstructs_accented_e(self):
        assert sanitise_text("S\x00e9zary syndrome") == "Sézary syndrome"

    def test_null_byte_reconstructs_accented_o(self):
        assert sanitise_text("Sj\x00f6gren") == "Sjögren"

    def test_null_byte_reconstructs_registered_trademark(self):
        assert sanitise_text("Technospiron\x00AE") == "Technospiron®"

    def test_orphan_null_byte_removed(self):
        assert sanitise_text("te\x00st") == "test"

    def test_aact_unescapes_left_square_bracket(self):
        assert sanitise_text(r"pain \[hyperalgesia\]") == "pain [hyperalgesia]"

    def test_aact_unescapes_less_than(self):
        assert sanitise_text(r"platelets \<= 30x109/L") == "platelets <= 30x109/L"

    def test_aact_unescapes_greater_than(self):
        assert sanitise_text(r"\> 5 cm") == "> 5 cm"

    def test_aact_unescapes_ampersand(self):
        assert sanitise_text(r"K\&L grade 3") == "K&L grade 3"

    def test_aact_unescapes_tilde(self):
        assert sanitise_text(r"foo \~ bar") == "foo ~ bar"

    def test_aact_unescapes_underscore(self):
        assert sanitise_text(r"foo \_ bar") == "foo _ bar"

    def test_aact_unescapes_asterisk(self):
        assert sanitise_text(r"foo \* bar") == "foo * bar"

    def test_already_clean_text_is_idempotent(self):
        assert sanitise_text("Sézary") == "Sézary"
        assert sanitise_text("pain [hyperalgesia]") == "pain [hyperalgesia]"

    def test_garbled_null_byte_removed_but_no_reconstruction(self):
        assert sanitise_text("s\x00zary") == "szary"


class TestSanitiseNested:
    """Unit tests for sanitise_nested."""

    def test_dict_with_string_values(self):
        data = {"name": "S\x00e9zary", "quote": r"pain \[test\]"}
        expected = {"name": "Sézary", "quote": "pain [test]"}
        assert sanitise_nested(data) == expected

    def test_list_of_strings(self):
        data = ["S\x00e9zary", r"K\&L"]
        expected = ["Sézary", "K&L"]
        assert sanitise_nested(data) == expected

    def test_deeply_nested_structure(self):
        data = {"a": [{"b": "S\x00e9zary", "c": [r"foo \* bar"]}]}
        expected = {"a": [{"b": "Sézary", "c": ["foo * bar"]}]}
        assert sanitise_nested(data) == expected

    def test_non_string_values_pass_through_unchanged(self):
        data = {"x": 42, "y": None, "z": True, "w": 3.14}
        assert sanitise_nested(data) == data

    def test_mixed_list_preserves_types(self):
        data = ["text", 42, None, ["nested"]]
        expected = ["text", 42, None, ["nested"]]
        assert sanitise_nested(data) == expected

    def test_plain_string(self):
        assert sanitise_nested("S\x00e9zary") == "Sézary"

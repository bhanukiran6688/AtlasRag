import pytest
from src.utils.metadata_filters import parse_metadata_filter, build_metadata_filter, validate_metadata_filter


class TestParseMetadataFilter:
    def test_parse_metadata_filter_with_valid_json(self):
        result = parse_metadata_filter('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_metadata_filter_with_invalid_json_raises_error(self):
        with pytest.raises(ValueError, match="Custom metadata JSON must be valid"):
            parse_metadata_filter('{"key": value}')

    def test_parse_metadata_filter_with_non_object_raises_error(self):
        with pytest.raises(ValueError, match="Custom metadata JSON must be an object"):
            parse_metadata_filter('["item1", "item2"]')

    def test_parse_metadata_filter_with_empty_string_raises_error(self):
        with pytest.raises(ValueError, match="Custom metadata JSON must be valid"):
            parse_metadata_filter("")

    def test_parse_metadata_filter_with_nested_object(self):
        result = parse_metadata_filter('{"key1": "value1", "key2": {"nested": "value"}}')
        assert result == {"key1": "value1", "key2": {"nested": "value"}}


class TestValidateMetadataFilter:
    def test_validate_metadata_filter_with_valid_string_value(self):
        result = validate_metadata_filter({"key": "value"})
        assert result == {"key": "value"}

    def test_validate_metadata_filter_with_valid_number_value(self):
        result = validate_metadata_filter({"key": 123})
        assert result == {"key": 123}

    def test_validate_metadata_filter_with_valid_boolean_value(self):
        result = validate_metadata_filter({"key": True})
        assert result == {"key": True}

    def test_validate_metadata_filter_with_empty_key_raises_error(self):
        with pytest.raises(ValueError, match="Metadata filter keys must be non-empty strings"):
            validate_metadata_filter({"": "value"})

    def test_validate_metadata_filter_with_whitespace_key_normalizes(self):
        result = validate_metadata_filter({"  key  ": "value"})
        assert result == {"key": "value"}

    def test_validate_metadata_filter_with_disallowed_key_raises_error(self):
        allowed_keys = {"allowed_key"}
        with pytest.raises(ValueError, match="Metadata filter key is not allowed"):
            validate_metadata_filter({"disallowed_key": "value"}, allowed_keys=allowed_keys)

    def test_validate_metadata_filter_with_allowed_key_succeeds(self):
        allowed_keys = {"allowed_key"}
        result = validate_metadata_filter({"allowed_key": "value"}, allowed_keys=allowed_keys)
        assert result == {"allowed_key": "value"}

    def test_validate_metadata_filter_with_invalid_value_type_raises_error(self):
        with pytest.raises(ValueError, match="Metadata filter value for 'key' must be a string, number, or boolean"):
            validate_metadata_filter({"key": ["list"]})

    def test_validate_metadata_filter_with_dict_value_raises_error(self):
        with pytest.raises(ValueError, match="Metadata filter value for 'key' must be a string, number, or boolean"):
            validate_metadata_filter({"key": {"nested": "value"}})

    def test_validate_metadata_filter_with_list_value_raises_error(self):
        with pytest.raises(ValueError, match="Metadata filter value for 'key' must be a string, number, or boolean"):
            validate_metadata_filter({"key": [1, 2, 3]})

    def test_validate_metadata_filter_with_none_value_raises_error(self):
        with pytest.raises(ValueError, match="Metadata filter value for 'key' must be a string, number, or boolean"):
            validate_metadata_filter({"key": None})

    def test_validate_metadata_filter_strips_string_values(self):
        result = validate_metadata_filter({"key": "  value  "})
        assert result == {"key": "value"}


class TestBuildMetadataFilter:
    def test_build_metadata_filter_with_source(self):
        result = build_metadata_filter(source="test.pdf")
        assert result == {"source": "test.pdf"}

    def test_build_metadata_filter_with_file_type(self):
        result = build_metadata_filter(file_type="PDF")
        assert result == {"file_type": "pdf"}

    def test_build_metadata_filter_with_custom_metadata(self):
        result = build_metadata_filter(custom_metadata={"department": "finance"})
        assert result == {"department": "finance"}

    def test_build_metadata_filter_combines_all_fields(self):
        result = build_metadata_filter(
            source="test.pdf",
            file_type="PDF",
            custom_metadata={"department": "finance"}
        )
        assert result == {"source": "test.pdf", "file_type": "pdf", "department": "finance"}

    def test_build_metadata_filter_with_empty_strings_returns_none(self):
        result = build_metadata_filter(source="", file_type="", custom_metadata=None)
        assert result is None

    def test_build_metadata_filter_with_whitespace_strings_returns_none(self):
        result = build_metadata_filter(source="  ", file_type="  ", custom_metadata=None)
        assert result is None

    def test_build_metadata_filter_with_allowed_keys_filters_custom_metadata(self):
        allowed_keys = {"department", "source"}
        result = build_metadata_filter(
            source="test.pdf",
            custom_metadata={"department": "finance", "disallowed": "value"},
            allowed_keys=allowed_keys
        )
        assert result == {"source": "test.pdf", "department": "finance"}

    def test_build_metadata_filter_without_allowed_keys_includes_all(self):
        result = build_metadata_filter(
            source="test.pdf",
            custom_metadata={"any_key": "value"},
            allowed_keys=None
        )
        assert result == {"source": "test.pdf", "any_key": "value"}

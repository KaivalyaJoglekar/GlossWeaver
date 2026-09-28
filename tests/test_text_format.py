import pytest

from glossweaver.text_format import format_gloss_input


def test_shared_formatter_preserves_words_and_normalizes_space():
    formatted = format_gloss_input("  MOTHER   COOK FOOD TONIGHT ")
    assert formatted.endswith("MOTHER COOK FOOD TONIGHT")
    assert "OTHER" not in formatted.replace("MOTHER", "")


def test_shared_formatter_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        format_gloss_input("  ")

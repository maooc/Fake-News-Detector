import pytest
from text_clean import clean_text, clean_many, URL_RE, EMAIL_RE, NON_ASCII_RE, EXTRA_SPACE_RE


class TestCleanText:
    def test_basic_cleaning(self):
        text = "  Hello   World!  "
        result = clean_text(text)
        assert result == "hello world!"

    def test_lowercase(self):
        text = "UPPERCASE Text HERE"
        result = clean_text(text, lowercase=True)
        assert result == "uppercase text here"

    def test_no_lowercase(self):
        text = "UPPERCASE Text HERE"
        result = clean_text(text, lowercase=False)
        assert result == "UPPERCASE Text HERE"

    def test_remove_urls(self):
        text = "Check https://example.com and http://test.org for info"
        result = clean_text(text, remove_urls=True)
        assert "https://" not in result
        assert "http://" not in result
        assert "check" in result
        assert "info" in result

    def test_keep_urls(self):
        text = "Visit https://example.com today"
        result = clean_text(text, remove_urls=False)
        assert "https://example.com" in result

    def test_remove_emails(self):
        text = "Contact us at user@example.com for details"
        result = clean_text(text, remove_emails=True)
        assert "user@example.com" not in result
        assert "contact" in result
        assert "details" in result

    def test_keep_emails(self):
        text = "Email: test@domain.com"
        result = clean_text(text, remove_emails=False)
        assert "test@domain.com" in result

    def test_remove_non_ascii(self):
        text = "Café résumé naïve"
        result = clean_text(text, remove_non_ascii=True)
        assert "café" not in result
        assert "caf" in result

    def test_keep_non_ascii(self):
        text = "Café résumé"
        result = clean_text(text, remove_non_ascii=False)
        assert "café" in result
        assert "résumé" in result

    def test_collapse_whitespace(self):
        text = "Multiple   spaces   here"
        result = clean_text(text, collapse_whitespace=True)
        assert result == "multiple spaces here"

    def test_no_collapse_whitespace(self):
        text = "Multiple   spaces"
        result = clean_text(text, collapse_whitespace=False)
        assert "   " in result

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_none_input(self):
        result = clean_text(None)
        assert result == ""

    def test_non_string_input(self):
        result = clean_text(12345)
        assert result == ""
        result = clean_text(["list"])
        assert result == ""

    def test_whitespace_only(self):
        result = clean_text("   \t\n   ")
        assert result == ""

    def test_combined_cleaning(self):
        text = "Check https://example.com and email user@test.com about Café!   "
        result = clean_text(text)
        assert "https://" not in result
        assert "@" not in result
        assert "café" not in result
        assert "check" in result
        assert "email" in result

    def test_special_characters_preserved(self):
        text = "Hello! How are you? This is great."
        result = clean_text(text)
        assert "!" in result
        assert "?" in result
        assert "." in result

    def test_numbers_preserved(self):
        text = "The year 2024 has 365 days"
        result = clean_text(text)
        assert "2024" in result
        assert "365" in result


class TestCleanMany:
    def test_clean_many_basic(self, sample_texts):
        results = clean_many(sample_texts)
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, str) for r in results)

    def test_clean_many_empty_list(self):
        results = clean_many([])
        assert results == []

    def test_clean_many_with_none(self):
        texts = ["Valid text", None, "Another text"]
        results = clean_many(texts)
        assert len(results) == 3
        assert results[1] == ""

    def test_clean_many_preserves_order(self):
        texts = ["First text", "Second text", "Third text"]
        results = clean_many(texts)
        assert "first" in results[0]
        assert "second" in results[1]
        assert "third" in results[2]

    def test_clean_many_with_options(self, dirty_texts):
        results = clean_many(dirty_texts, lowercase=True, remove_urls=True)
        assert all("https://" not in r for r in results)

    def test_clean_many_with_no_lowercase(self):
        texts = ["UPPERCASE TEXT"]
        results = clean_many(texts, lowercase=False)
        assert results[0] == "UPPERCASE TEXT"


class TestRegexPatterns:
    def test_url_pattern_matches(self):
        text = "Visit https://example.com and http://test.org"
        matches = URL_RE.findall(text)
        assert "https://example.com" in matches
        assert "http://test.org" in matches

    def test_email_pattern_matches(self):
        text = "Contact user@test.com and admin@example.org"
        matches = EMAIL_RE.findall(text)
        assert len(matches) == 2

    def test_non_ascii_pattern_matches(self):
        text = "Café résumé"
        matches = NON_ASCII_RE.findall(text)
        assert len(matches) > 0

    def test_extra_space_pattern(self):
        text = "Multiple   spaces"
        result = EXTRA_SPACE_RE.sub(" ", text)
        assert "   " not in result

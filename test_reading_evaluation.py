"""Unit tests for reading evaluation heuristics."""

from app import is_sentence_match


def test_sentence_match_rejects_single_word_substrings() -> None:
    assert not is_sentence_match("dog", "The dog ran fast.")


def test_sentence_match_rejects_partial_phrase_substrings() -> None:
    assert not is_sentence_match("dog ran", "The dog ran fast.")


def test_sentence_match_accepts_complete_sentence_with_filler_removed() -> None:
    assert is_sentence_match("the dog ran fast", "The dog ran fast.")


def test_sentence_match_accepts_complete_sentence_inside_extra_words() -> None:
    assert is_sentence_match("um please the dog ran fast now", "The dog ran fast.")


def test_sentence_match_rejects_embedded_word_substring() -> None:
    assert not is_sentence_match("dog ran fastness", "The dog ran fast.")


def test_sentence_match_allows_small_edit_distance_for_full_attempt() -> None:
    assert is_sentence_match("dog ran fass", "The dog ran fast.")

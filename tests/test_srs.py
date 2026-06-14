"""SM-2 spaced-repetition scheduler — deterministic, pure."""
from backend.srs import CardState, review, MIN_EASE, DEFAULT_EASE


def test_first_correct_review_interval_1():
    s = review(CardState(), 4)
    assert s.repetitions == 1 and s.interval == 1


def test_second_correct_review_interval_6():
    s = review(review(CardState(), 4), 4)
    assert s.repetitions == 2 and s.interval == 6


def test_third_review_scales_by_ease():
    s = review(review(review(CardState(), 4), 4), 4)
    # interval = round(6 * ease); ease ~2.5 → ~15
    assert s.repetitions == 3 and s.interval == round(6 * s.ease)
    assert 14 <= s.interval <= 16


def test_lapse_resets_repetitions_and_interval():
    grown = review(review(review(CardState(), 5), 5), 5)
    assert grown.repetitions == 3
    lapsed = review(grown, 1)            # failed recall
    assert lapsed.repetitions == 0 and lapsed.interval == 1


def test_ease_floored_at_minimum():
    s = CardState()
    for _ in range(10):                  # repeated hard/failed reviews
        s = review(s, 2)
    assert s.ease >= MIN_EASE


def test_easy_grade_raises_ease():
    s = review(CardState(), 5)
    assert s.ease > DEFAULT_EASE         # grade 5 increases ease


def test_grade_clamped():
    assert review(CardState(), 99).interval >= 1   # out-of-range grade tolerated
    assert review(CardState(), -5).repetitions == 0


def test_intervals_strictly_grow_on_success():
    s = CardState()
    last = 0
    for _ in range(5):
        s = review(s, 4)
        assert s.interval >= last
        last = s.interval

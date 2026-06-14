"""
Spaced-repetition scheduler — SM-2 (the SuperMemo-2 algorithm, as used by Anki).

Pure, deterministic, dependency-free → fully offline and unit-testable. Given a
card's current state and a review grade, it returns the next state (ease factor,
interval in days, repetition count). The API layer turns `interval_days` into a
concrete due date.

Grade scale (0-5), the SM-2 convention:
    0-2 = incorrect / failed recall  → reset, review again soon
    3   = correct but hard
    4   = correct
    5   = correct and easy

Reference: P. A. Wozniak, "Optimization of repetition spacing…", 1990.
"""
from dataclasses import dataclass

MIN_EASE = 1.3          # SM-2 floor — below this, intervals grow too slowly
DEFAULT_EASE = 2.5      # starting ease factor for a new card


@dataclass
class CardState:
    repetitions: int = 0      # consecutive successful recalls
    interval: int = 0         # days until next review
    ease: float = DEFAULT_EASE


def review(state: CardState, grade: int) -> CardState:
    """Return the next CardState for a review `grade` in 0..5.

    A grade < 3 is a lapse: repetitions reset and the card is shown again the
    next day. A grade >= 3 advances the schedule with expanding intervals.
    The ease factor is always updated by the SM-2 formula and floored at 1.3.
    """
    grade = max(0, min(5, int(grade)))

    if grade < 3:
        repetitions = 0
        interval = 1
    else:
        if state.repetitions == 0:
            interval = 1
        elif state.repetitions == 1:
            interval = 6
        else:
            interval = round(state.interval * state.ease)
        repetitions = state.repetitions + 1

    # SM-2 ease update (applied for every grade)
    ease = state.ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ease = max(MIN_EASE, ease)

    return CardState(repetitions=repetitions, interval=max(1, interval), ease=round(ease, 4))

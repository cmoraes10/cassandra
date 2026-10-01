"""
Signal generation for Cassandra.

After simulating the scenarios, the cloud of projected prices needs to become
a practical verdict: buy, sell, or stay out, together with a confidence score
that says how seriously to take that signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulation import SimulationResult

BUY = "BUY"
SELL = "SELL"
NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """Strategy decision for a ticker at a given moment."""

    direction: str
    confidence: float
    bull_probability: float
    expected_return: float
    current_price: float

    def summary(self) -> str:
        """Short sentence describing the signal in a readable form."""
        return (
            f"{self.direction} with confidence {self.confidence:.0f} out of 100, "
            f"projected return of {self.expected_return * 100:.1f} percent"
        )


def generate_signal(result: SimulationResult, confidence_threshold: float = 60.0) -> Signal:
    """Translates simulation output into a trade signal.

    Confidence comes from how far the bull probability is from a pure coin flip.
    If half the scenarios go up and half go down, confidence is zero. If almost
    all point the same way, confidence is close to a hundred. Only when confidence
    exceeds the threshold does Cassandra commit to a direction; otherwise it
    prefers to stay out.
    """
    prob = result.bull_probability()
    ret = result.expected_return()
    confidence = round(abs(prob - 0.5) * 2 * 100, 1)

    if confidence < confidence_threshold or prob == 0.5:
        # confidence below threshold or perfect tie between bull and bear
        direction = NEUTRAL
    elif prob > 0.5:
        direction = BUY
    else:
        direction = SELL

    return Signal(
        direction=direction,
        confidence=confidence,
        bull_probability=prob,
        expected_return=ret,
        current_price=result.initial_price,
    )

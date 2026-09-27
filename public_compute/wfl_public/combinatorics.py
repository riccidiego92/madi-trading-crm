from __future__ import annotations

from math import comb
from typing import Iterable

UNIVERSE = tuple(range(1, 21))
FULL_MASK = (1 << 20) - 1
N_MAIN = comb(20, 10)
N_COMPLEMENT_PAIRS = N_MAIN // 2
N_NUMERONE = 20
JACKPOT_ODDS_EUR2 = N_COMPLEMENT_PAIRS * N_NUMERONE


def numbers_to_mask(numbers: Iterable[int]) -> int:
    vals = sorted(set(int(x) for x in numbers))
    if len(vals) != 10 or vals[0] < 1 or vals[-1] > 20:
        raise ValueError("Expected exactly 10 unique integers in 1..20")
    mask = 0
    for n in vals:
        mask |= 1 << (n - 1)
    return mask


def mask_to_numbers(mask: int) -> tuple[int, ...]:
    return tuple(i + 1 for i in range(20) if mask & (1 << i))


def complement_mask(mask: int) -> int:
    return FULL_MASK ^ mask


def canonical_pair_id(numbers: Iterable[int]) -> int:
    """Stable bit-mask pair identifier (unique but not contiguous)."""
    mask = numbers_to_mask(numbers)
    comp = complement_mask(mask)
    return min(mask, comp)


def combination_rank(numbers: Iterable[int]) -> int:
    """Colexicographic rank of a 10-subset, exactly in 0..C(20,10)-1."""
    vals = sorted(set(int(x) for x in numbers))
    if len(vals) != 10 or vals[0] < 1 or vals[-1] > 20:
        raise ValueError("Expected exactly 10 unique integers in 1..20")
    return sum(comb(v - 1, i) for i, v in enumerate(vals, start=1))


def canonical_pair_rank(numbers: Iterable[int]) -> int:
    """Contiguous rank of the unoriented complementary pair, 0..92377."""
    r = combination_rank(numbers)
    return min(r, N_MAIN - 1 - r)


def joint_jackpot_state(numbers: Iterable[int], numerone: int) -> int:
    """Contiguous (complementary main pair, Numerone) state, 0..1847559."""
    n = int(numerone)
    if not 1 <= n <= 20:
        raise ValueError("numerone must be in 1..20")
    return canonical_pair_rank(numbers) * 20 + (n - 1)


def complement_numbers(numbers: Iterable[int]) -> tuple[int, ...]:
    return mask_to_numbers(complement_mask(numbers_to_mask(numbers)))


def pair_match_count(ticket: Iterable[int], drawn: Iterable[int]) -> int:
    """Best symmetric match count for a EUR2 line: max(k, 10-k), range 5..10."""
    t = set(ticket)
    d = set(drawn)
    if len(t) != 10 or len(d) != 10:
        raise ValueError("ticket and drawn must each contain 10 unique numbers")
    k = len(t & d)
    return max(k, 10 - k)


def exact_main_pair_probability() -> float:
    return 1.0 / N_COMPLEMENT_PAIRS


def exact_jackpot_probability_eur2() -> float:
    return 1.0 / JACKPOT_ODDS_EUR2


def single_set_match_probability(k: int) -> float:
    """P(K=k) for a fixed 10-number set vs random 10-of-20 draw."""
    if not 0 <= k <= 10:
        return 0.0
    return comb(10, k) * comb(10, 10 - k) / comb(20, 10)


def symmetric_pair_match_probability(m: int) -> float:
    """P(max(K,10-K)=m), m in 5..10."""
    if m < 5 or m > 10:
        return 0.0
    if m == 5:
        return single_set_match_probability(5)
    return single_set_match_probability(m) + single_set_match_probability(10 - m)

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
from math import comb
from typing import Iterable

import pandas as pd

from .analysis import MAIN_COLS

ENGINE = "HMAC_DRBG_SHA256"
MAPPERS = ("fisher_yates", "sequential", "rejection", "rank_unrank")
STREAM_MODES = ("single", "split_main_numerone")
RESEED_MODES = ("continuous", "per_day", "per_contest")

N = 20
K = 10
STATE_COUNT = comb(N, K)


class HmacDrbgSha256:
    """Synthetic HMAC_DRBG-SHA256 comparator.

    The update/generate core follows the HMAC_DRBG structure in NIST
    SP 800-90A Rev.1. Instantiation material here is deterministically derived
    from a PUBLIC EXPERIMENT seed; this class is not a CAVP/FIPS validation
    claim and must never be used to search for an operational Classico seed.
    """

    OUTLEN = 32

    def __init__(self, seed_material: bytes):
        if not isinstance(seed_material, (bytes, bytearray)) or not seed_material:
            raise ValueError("seed_material must be non-empty bytes")
        self.key = b"\x00" * self.OUTLEN
        self.value = b"\x01" * self.OUTLEN
        self.reseed_counter = 1
        self._update(bytes(seed_material))

    @staticmethod
    def _mac(key: bytes, data: bytes) -> bytes:
        return hmac.new(key, data, sha256).digest()

    def _update(self, provided_data: bytes = b"") -> None:
        data = bytes(provided_data)
        self.key = self._mac(self.key, self.value + b"\x00" + data)
        self.value = self._mac(self.key, self.value)
        if data:
            self.key = self._mac(self.key, self.value + b"\x01" + data)
            self.value = self._mac(self.key, self.value)

    def generate(self, nbytes: int, additional_input: bytes = b"") -> bytes:
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if additional_input:
            self._update(additional_input)

        out = bytearray()
        while len(out) < nbytes:
            self.value = self._mac(self.key, self.value)
            out.extend(self.value)

        self._update(additional_input)
        self.reseed_counter += 1
        return bytes(out[:nbytes])

    def randbelow(self, bound: int) -> int:
        """Uniform integer in [0,bound) via exact 64-bit rejection."""
        bound = int(bound)
        if bound <= 0:
            raise ValueError("bound must be positive")
        span = 1 << 64
        limit = span - (span % bound)
        while True:
            raw = int.from_bytes(self.generate(8), "big", signed=False)
            if raw < limit:
                return raw % bound


@dataclass(frozen=True)
class DrbgCandidateConfig:
    mapper: str
    stream_mode: str = "single"
    reseed_mode: str = "continuous"
    experiment_seed: int = 2026092601
    contests_per_day: int = 17

    def validate(self) -> None:
        if self.mapper not in MAPPERS:
            raise ValueError(f"unknown mapper {self.mapper}")
        if self.stream_mode not in STREAM_MODES:
            raise ValueError(f"unknown stream mode {self.stream_mode}")
        if self.reseed_mode not in RESEED_MODES:
            raise ValueError(f"unknown reseed mode {self.reseed_mode}")
        if int(self.contests_per_day) <= 0:
            raise ValueError("contests_per_day must be positive")

    def identity(self) -> str:
        self.validate()
        return f"{ENGINE}|{self.mapper}|{self.stream_mode}|{self.reseed_mode}"

    def to_dict(self) -> dict:
        return asdict(self)


def _seed_material(root: int, identity: str, scope: object, label: str) -> bytes:
    payload = f"{int(root)}|{identity}|{scope}|{label}".encode("utf-8")
    # 384 deterministic bits are supplied as synthetic experiment material.
    return sha256(b"A|" + payload).digest() + sha256(b"B|" + payload).digest()[:16]


def _drbg_pair(
    config: DrbgCandidateConfig,
    scope: object,
) -> tuple[HmacDrbgSha256, HmacDrbgSha256]:
    identity = config.identity()
    main = HmacDrbgSha256(
        _seed_material(config.experiment_seed, identity, scope, "main")
    )
    if config.stream_mode == "single":
        return main, main
    num = HmacDrbgSha256(
        _seed_material(config.experiment_seed, identity, scope, "numerone")
    )
    return main, num


def _fisher_yates(rng: HmacDrbgSha256) -> tuple[int, ...]:
    arr = list(range(1, N + 1))
    for i in range(N - 1, 0, -1):
        j = rng.randbelow(i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return tuple(sorted(arr[:K]))


def _sequential(rng: HmacDrbgSha256) -> tuple[int, ...]:
    selected: list[int] = []
    need = K
    remaining = N
    for value in range(1, N + 1):
        if need == 0:
            break
        if rng.randbelow(remaining) < need:
            selected.append(value)
            need -= 1
        remaining -= 1
    if len(selected) != K:
        raise AssertionError("sequential sampler did not select K values")
    return tuple(selected)


def _rejection(rng: HmacDrbgSha256) -> tuple[int, ...]:
    selected: set[int] = set()
    while len(selected) < K:
        selected.add(1 + rng.randbelow(N))
    return tuple(sorted(selected))


def _unrank_lex(rank: int, n: int = N, k: int = K) -> tuple[int, ...]:
    rank = int(rank)
    if not 0 <= rank < comb(n, k):
        raise ValueError("rank out of range")
    r = rank
    out: list[int] = []
    start = 1
    remaining_slots = k
    while remaining_slots:
        max_value = n - remaining_slots + 1
        for value in range(start, max_value + 1):
            block = comb(n - value, remaining_slots - 1)
            if r < block:
                out.append(value)
                start = value + 1
                remaining_slots -= 1
                break
            r -= block
        else:
            raise AssertionError("unrank failed")
    return tuple(out)


def _rank_unrank(rng: HmacDrbgSha256) -> tuple[int, ...]:
    return _unrank_lex(rng.randbelow(STATE_COUNT))


def draw_main(rng: HmacDrbgSha256, mapper: str) -> tuple[int, ...]:
    if mapper == "fisher_yates":
        return _fisher_yates(rng)
    if mapper == "sequential":
        return _sequential(rng)
    if mapper == "rejection":
        return _rejection(rng)
    if mapper == "rank_unrank":
        return _rank_unrank(rng)
    raise ValueError(mapper)


def draw_numerone(rng: HmacDrbgSha256) -> int:
    return 1 + rng.randbelow(N)


def _validate_result(main: Iterable[int], numerone: int) -> None:
    vals = tuple(sorted(int(x) for x in main))
    if len(vals) != K or len(set(vals)) != K or vals[0] < 1 or vals[-1] > N:
        raise AssertionError(f"invalid synthetic main result {vals}")
    if not 1 <= int(numerone) <= N:
        raise AssertionError(f"invalid synthetic Numerone {numerone}")


def simulate_drbg(n_contests: int, config: DrbgCandidateConfig) -> pd.DataFrame:
    """Generate a deterministic synthetic certification-style comparator history."""
    config.validate()
    n_contests = int(n_contests)
    if n_contests < 0:
        raise ValueError("n_contests must be non-negative")

    continuous_pair = None
    day_pair = None
    day_scope = None
    if config.reseed_mode == "continuous":
        continuous_pair = _drbg_pair(config, "continuous")

    rows: list[dict] = []
    for i in range(n_contests):
        day_index = i // int(config.contests_per_day)
        if config.reseed_mode == "continuous":
            main_rng, num_rng = continuous_pair
        elif config.reseed_mode == "per_contest":
            main_rng, num_rng = _drbg_pair(config, f"contest:{i}")
        elif config.reseed_mode == "per_day":
            if day_pair is None or day_scope != day_index:
                day_scope = day_index
                day_pair = _drbg_pair(config, f"day:{day_index}")
            main_rng, num_rng = day_pair
        else:
            raise ValueError(config.reseed_mode)

        main = draw_main(main_rng, config.mapper)
        numerone = draw_numerone(num_rng)
        _validate_result(main, numerone)

        row = {
            "synthetic_index": i,
            "synthetic_day": day_index,
            "synthetic_contest_in_day": i % int(config.contests_per_day),
            "numerone": numerone,
        }
        for j, value in enumerate(main, start=1):
            row[f"n{j}"] = int(value)
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            "synthetic_index",
            "synthetic_day",
            "synthetic_contest_in_day",
            *MAIN_COLS,
            "numerone",
        ],
    )


def drbg_candidate_grid(seed_root: int = 2026092601) -> list[DrbgCandidateConfig]:
    out = []
    for mapper in MAPPERS:
        for stream_mode in STREAM_MODES:
            for reseed_mode in RESEED_MODES:
                out.append(
                    DrbgCandidateConfig(
                        mapper=mapper,
                        stream_mode=stream_mode,
                        reseed_mode=reseed_mode,
                        experiment_seed=int(seed_root),
                    )
                )
    return out

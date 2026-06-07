"""Fixture-backed Chess benchmark implementation entry point."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from leibniz.benchmark_implementations import Benchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.observation_generation import (
    ComplexityCandidate,
    ComplexityRequest,
    ComplexityValue,
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.timing import TimingCollector

__all__ = ["benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.chess.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.chess.uci-moves@0.1.0")
_fen = "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"
_observation_id = f"fen:{_fen}"
_valid_moves = ("g7f8", "g7g8", "g6f7")
_accepted_move = "g7f8"


def benchmark(root: Path) -> BenchmarkProtocol:
    """Return the Chess fixture benchmark implementation."""

    return Benchmark(root=root)


class Benchmark:
    """Executable Chess fixture benchmark declaration."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._manifest = _manifest()
        self._generator = Generator(manifest=self._manifest)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> BenchmarkManifest:
        return self._manifest

    @property
    def generator(self) -> Generator:
        return self._generator


@dataclass(frozen=True, slots=True)
class Generator:
    """Generate Chess positions from the fixture-backed scientific surface."""

    manifest: BenchmarkManifest

    @property
    def id(self) -> ProtocolIdentifier:
        return _generator_id

    @property
    def version(self) -> str:
        return "0.1.0"

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        include_metadata: bool = True,
        complexity_request: ComplexityRequest | None = None,
        runtime: object | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        """Generate a shape-aware Chess sample set."""

        _ = include_fields
        _ = outcome_ids
        _ = timing
        _ = timing_prefix
        if runtime is not None:
            raise ObservationGenerationError(
                "Chess fixture does not define tensor generation"
            )
        if not include_metadata:
            raise ObservationGenerationError(
                "Chess fixture metadata-free generation requires a tensor runtime"
            )
        sample_shape = _sample_shape(shape)
        measure = _complexity_value()
        if complexity_request is not None and not complexity_request.contains(measure):
            return GeneratedSampleSet(
                benchmark_id=self.manifest.id,
                generator_id=self.id,
                generator_version=self.version,
                seed=seed,
                shape=(0,),
                complexity_request=complexity_request,
                samples=(),
            )
        samples = (
            tuple(
                GeneratedSample(
                    index=index,
                    outcome_id=_accepted_move,
                    complexity=measure.value,
                    complexity_value=measure,
                    latent_coordinates=_latent_coordinates(),
                )
                for index in range(_sample_count(sample_shape))
            )
            if include_metadata
            else ()
        )
        return GeneratedSampleSet(
            benchmark_id=self.manifest.id,
            generator_id=self.id,
            generator_version=self.version,
            seed=seed,
            shape=sample_shape,
            complexity_request=complexity_request,
            samples=samples,
        )

    def minimum_complexity(self) -> ComplexityValue:
        """Return the smallest exact Chess fixture complexity."""

        return _complexity_value()

    def complexity_candidate_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> ComplexityCandidate | None:
        """Return the exact sample cardinality complexity class for this fixture."""

        measure = _complexity_value()
        if not request.contains(measure):
            return None
        return ComplexityCandidate(
            request=ComplexityRequest(
                minimum=measure.value,
                maximum=measure.value,
            ),
            cardinality=1,
            metadata={
                "kind": "chess-sample-cardinality",
                "fen": _fen,
                "sample_cardinality": 1,
                "valid_move_count": len(_valid_moves),
            },
        )

    def complexity_candidates_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> tuple[ComplexityCandidate, ...]:
        """Return exact sample cardinality candidates in a complexity band."""

        complexity_class = self.complexity_candidate_for_request(request=request)
        if complexity_class is None:
            return ()
        return (complexity_class,)

    def complexity_curriculum_candidates(
        self,
        *,
        start_index: int,
        count: int,
    ) -> tuple[ComplexityCandidate, ...]:
        """Return the fixture's exact sample cardinality schedule."""

        if start_index < 0:
            raise ObservationGenerationError("start_index must be non-negative")
        if count < 0:
            raise ObservationGenerationError("count must be non-negative")
        if start_index > 0 or count == 0:
            return ()
        return (
            ComplexityCandidate(
                request=ComplexityRequest(
                    minimum=_complexity_value().value,
                    maximum=_complexity_value().value,
                ),
                cardinality=1,
                metadata={
                    "kind": "chess-sample-cardinality",
                    "fen": _fen,
                    "sample_cardinality": 1,
                    "valid_move_count": len(_valid_moves),
                },
            ),
        )


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id=_benchmark_id,
        name=_benchmark_id.name,
        outcome_space=OutcomeSpace(
            id=_outcome_space_id,
            outcomes=tuple(Outcome(id=move) for move in _valid_moves),
        ),
        observation_ids=frozenset({_observation_id}),
    )


def _complexity_value() -> ComplexityValue:
    return ComplexityValue(
        value=0.0,
    )


def _latent_coordinates() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "benchmarks.chess.position.fen",
            "role": "content",
            "degree_measure": {"kind": "fixed-fixture", "count": 1.0},
            "multiplicity": 1,
            "values": _fen,
        },
        {
            "name": "benchmarks.chess.position.valid-move-count",
            "role": "content",
            "degree_measure": {"kind": "finite-set", "count": len(_valid_moves)},
            "multiplicity": 1,
            "values": len(_valid_moves),
        },
    )


def _sample_shape(shape: int | Sequence[int] | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        if shape < 1:
            raise ObservationGenerationError("sample shape axes must be positive integers")
        return (shape,)
    normalized = tuple(shape)
    if any(type(axis) is not int or axis < 1 for axis in normalized):
        raise ObservationGenerationError("sample shape axes must be positive integers")
    return normalized


def _sample_count(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    count = 1
    for axis in shape:
        count *= axis
    return count

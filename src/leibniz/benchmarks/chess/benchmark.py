"""Chess benchmark implementation entry point."""

from __future__ import annotations

import math
import random
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

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
from leibniz.tensor_runtime import (
    TensorRuntime,
    tensor_runtime_backend,
)
from leibniz.timing import TimingCollector

__all__ = ["all_meaningful_uci_moves", "benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.chess.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.chess.uci-moves@0.1.0")
_console_preview_limit = 4
_console_preview_cardinalities = (1, 2, 4, 8, 16, 32, 64)
_tensor_shape = (18, 8, 8)
_board_preview_size = 512
_board_preview_square_size = _board_preview_size // 8

_mate_in_one_family_id = "corner-net-indexed-family"
_preview_representative_limit = 4


def benchmark(root: Path) -> BenchmarkProtocol:
    """Return the Chess benchmark implementation."""

    return Benchmark(root=root)


class Benchmark:
    """Executable Chess benchmark declaration."""

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
class _MateInOnePosition:
    fen: str
    legal_moves: tuple[str, ...]
    mate_moves: tuple[str, ...]
    mechanism_name: str
    transform_name: str
    family_index: int
    spectator_count: int

    @property
    def observation_id(self) -> str:
        return f"fen:{self.fen}"

    @property
    def legal_move_count(self) -> int:
        return len(self.legal_moves)

    @property
    def complexity(self) -> float:
        return math.log2(1)

    @property
    def complexity_value(self) -> ComplexityValue:
        return ComplexityValue(value=self.complexity)

    @property
    def target_distribution(self) -> Mapping[str, float]:
        probability = 1.0 / len(self.mate_moves)
        return dict.fromkeys(self.mate_moves, probability)

    @property
    def legal_move_piece_symbols(self) -> tuple[str, ...]:
        return _move_piece_symbols(fen=self.fen, moves=self.legal_moves)

    @property
    def mate_move_piece_symbols(self) -> tuple[str, ...]:
        return _move_piece_symbols(fen=self.fen, moves=self.mate_moves)

    def representative_analysis(self) -> Mapping[str, object]:
        board = chess.Board(self.fen)
        piece_map = board.piece_map()
        white_piece_count = sum(
            1 for piece in piece_map.values() if piece.color == chess.WHITE
        )
        black_piece_count = len(piece_map) - white_piece_count
        mate_sources = tuple(sorted(move[:2] for move in self.mate_moves))
        mate_targets = tuple(sorted(move[2:4] for move in self.mate_moves))
        return {
            "kind": "chess-mate-in-one-representative-analysis",
            "family": _mate_in_one_family_id,
            "mechanism": self.mechanism_name,
            "mechanism_piece_count": len(piece_map) - self.spectator_count,
            "transform": self.transform_name,
            "family_index": self.family_index,
            "spectator_count": self.spectator_count,
            "piece_count": len(piece_map),
            "white_piece_count": white_piece_count,
            "black_piece_count": black_piece_count,
            "board_piece_symbols": list(_board_piece_symbols(board)),
            "legal_move_piece_symbols": list(self.legal_move_piece_symbols),
            "mate_move_piece_symbols": list(self.mate_move_piece_symbols),
            "mate_move_count": len(self.mate_moves),
            "mate_source_square_count": len(frozenset(mate_sources)),
            "mate_target_square_count": len(frozenset(mate_targets)),
            "quality_flags": list(_representative_quality_flags(self)),
        }

    def representative_metadata(self) -> Mapping[str, object]:
        return {
            "kind": "chess-mate-in-one-sample-representative",
            "family": _mate_in_one_family_id,
            "mechanism": self.mechanism_name,
            "transform": self.transform_name,
            "family_index": self.family_index,
            "spectator_count": self.spectator_count,
            "fen": self.fen,
            "target_policy": "mate-in-one",
            "legal_move_count": self.legal_move_count,
            "legal_move_piece_symbols": list(self.legal_move_piece_symbols),
            "mate_move_piece_symbols": list(self.mate_move_piece_symbols),
            "mate_move_count": len(self.mate_moves),
            "analysis": dict(self.representative_analysis()),
        }


@dataclass(frozen=True, slots=True)
class Generator:
    """Generate Chess mate-in-one positions by sample-space complexity."""

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
        component_indices: Sequence[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: object | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        """Generate mate-in-one samples and optional board-state tensors."""

        _ = include_fields
        _ = component_indices
        _ = memory_limit_bytes
        _ = resolution_assignment
        _ = variation_extent
        _ = timing
        _ = timing_prefix
        if runtime is not None and outcome_ids is None:
            raise ObservationGenerationError("tensor generation requires outcome_ids")
        if runtime is None and not include_metadata:
            raise ObservationGenerationError(
                "Chess metadata-free generation requires a tensor runtime"
            )
        sample_shape = _sample_shape(shape)
        sample_space = self._sample_space_for_request(complexity_request)
        if sample_space is None:
            return GeneratedSampleSet(
                benchmark_id=self.manifest.id,
                generator_id=self.id,
                generator_version=self.version,
                seed=seed,
                shape=(0,),
                complexity_request=complexity_request,
                samples=(),
            )

        rng = random.Random(seed)
        sample_count = _sample_count(sample_shape)
        complexity = math.log2(sample_space.cardinality)
        selected_local_indices = _sample_local_indices(
            rng=rng,
            cardinality=sample_space.cardinality,
            sample_count=sample_count,
        )
        selected_global_indices = _global_sample_indices(
            cardinality=sample_space.cardinality,
            local_indices=selected_local_indices,
        )
        samples = (
            _samples_for_global_indices(
                global_indices=selected_global_indices,
                outcome_ids=outcome_ids,
                sample_space_cardinality=sample_space.cardinality,
                complexity=complexity,
                full_metadata=runtime is None,
            )
            if include_metadata
            else ()
        )
        fields, targets = _tensor_batch_for_global_indices(
            runtime=runtime,
            global_indices=selected_global_indices,
            cardinality=sample_space.cardinality,
            outcome_ids=outcome_ids,
            sample_shape=sample_shape,
        )
        return GeneratedSampleSet(
            benchmark_id=self.manifest.id,
            generator_id=self.id,
            generator_version=self.version,
            seed=seed,
            shape=sample_shape,
            complexity_request=complexity_request,
            samples=samples,
            fields=fields,
            targets=targets,
        )

    def minimum_complexity(self) -> ComplexityValue:
        """Return the smallest supported Chess sample-space complexity."""

        return ComplexityValue(value=0.0)

    def complexity_candidate_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> ComplexityCandidate | None:
        """Return the first sample-space-cardinality class inside a complexity band."""

        minimum_cardinality = _ceil_cardinality(request.minimum)
        maximum_cardinality = min(_floor_cardinality(request.maximum), _family_capacity())
        if maximum_cardinality < minimum_cardinality:
            return None
        return self._candidate_for_cardinality(minimum_cardinality)

    def complexity_curriculum_candidates(
        self,
        *,
        start_index: int,
        count: int,
    ) -> tuple[ComplexityCandidate, ...]:
        """Return Chess' ordered exact sample-space-cardinality schedule."""

        if start_index < 0:
            raise ObservationGenerationError("start_index must be non-negative")
        if count < 0:
            raise ObservationGenerationError("count must be non-negative")
        if count == 0:
            return ()

        candidates: list[ComplexityCandidate] = []
        for offset in range(count):
            cardinality = start_index + offset + 1
            if cardinality > _family_capacity():
                break
            candidates.append(self._candidate_for_cardinality(cardinality))
        return tuple(candidates)

    def _sample_space_cardinality_candidates(self) -> tuple[ComplexityCandidate, ...]:
        return tuple(
            self._candidate_for_cardinality(cardinality)
            for cardinality in _console_preview_cardinalities
            if cardinality <= _family_capacity()
        )

    def _candidate_for_cardinality(self, cardinality: int) -> ComplexityCandidate:
        _require_sample_cardinality(cardinality)
        complexity = math.log2(cardinality)
        representative_indices = _representative_local_indices(cardinality)
        representatives = [
            dict(
                _position_for_sample_index(
                    _global_sample_index(
                        cardinality=cardinality,
                        local_index=index,
                    )
                ).representative_metadata()
            )
            for index in representative_indices
        ]
        legal_move_counts = [
            _position_for_sample_index(
                _global_sample_index(
                    cardinality=cardinality,
                    local_index=index,
                )
            ).legal_move_count
            for index in representative_indices
        ]
        preview_legal_move_count_max = max(legal_move_counts) if legal_move_counts else 0
        oracle_compute = _oracle_inference_compute_for_cardinality(cardinality)
        return ComplexityCandidate(
            request=ComplexityRequest(
                minimum=complexity,
                maximum=complexity,
            ),
            cardinality=cardinality,
            metadata={
                "kind": "chess-sample-space-cardinality",
                "family": _mate_in_one_family_id,
                "sample_cardinality": cardinality,
                "target_policy": "mate-in-one",
                "transform_count": len(_board_transforms()),
                "spectator_square_count": len(_spectator_squares()),
                "output_move_count": len(all_meaningful_uci_moves),
                "representative_preview_count": len(representatives),
                "representatives": representatives,
                "oracle_inference_compute": {
                    "kind": "oracle-inference-compute-reference-v1",
                    "unit": "abstract-ops",
                    "aggregation": "analytic-upper-bound",
                    "value": oracle_compute,
                    "components": {
                        "preview_legal_move_count_max": preview_legal_move_count_max,
                        "max_spectator_count": _max_spectator_count_for_cardinality(
                            cardinality
                        ),
                        "sample_cardinality": cardinality,
                    },
                },
            },
        )

    def console_preview_batches(
        self,
        *,
        atom_count: int,
    ) -> tuple[Mapping[str, object], ...]:
        """Return browser-preview batches for sample-space cardinality classes."""

        if atom_count != len(self.manifest.outcome_space.outcomes):
            raise ObservationGenerationError("atom_count does not match outcome space")
        batches: list[Mapping[str, object]] = []
        for candidate in self._sample_space_cardinality_candidates():
            if candidate.cardinality is None:
                continue
            request = ComplexityRequest(
                minimum=candidate.complexity,
                maximum=candidate.complexity,
            )
            sample_set = self(
                seed=401 + len(batches),
                shape=min(candidate.cardinality, _console_preview_limit),
                complexity_request=request,
            )
            samples = [
                _sample_preview_record(sample)
                for sample in sample_set.samples[:_console_preview_limit]
            ]
            batches.append(
                {
                    "mode": "complexity-window",
                    "label": f"{candidate.cardinality} puzzle states",
                    "seed": sample_set.seed,
                    "sample_count": len(samples),
                    "complexity_window": request.to_record(),
                    "complexity_cardinalities": [candidate.cardinality],
                    "presentation": {
                        "sample_card_density": "standard",
                        "aggregate_mode": False,
                    },
                    "samples": samples,
                }
            )
        return tuple(batches)

    def _sample_space_for_request(
        self,
        request: ComplexityRequest | None,
    ) -> _ChessSampleSpace | None:
        if request is None:
            return _ChessSampleSpace(cardinality=1)
        minimum_cardinality = _ceil_cardinality(request.minimum)
        maximum_cardinality = _floor_cardinality(request.maximum)
        if maximum_cardinality < minimum_cardinality:
            return None
        if minimum_cardinality > _family_capacity():
            return None
        return _ChessSampleSpace(cardinality=minimum_cardinality)

def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id=_benchmark_id,
        name=_benchmark_id.name,
        outcome_space=OutcomeSpace(
            id=_outcome_space_id,
            outcomes=tuple(Outcome(id=move) for move in all_meaningful_uci_moves),
        ),
        observation_ids=None,
    )


@dataclass(frozen=True, slots=True)
class _ChessSampleSpace:
    cardinality: int

    def __post_init__(self) -> None:
        _require_sample_cardinality(self.cardinality)


@dataclass(frozen=True, slots=True)
class _BoardTransform:
    name: str
    square: Callable[[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class _MateMechanism:
    name: str
    queen_square: chess.Square
    support_pieces: tuple[tuple[chess.Square, chess.Piece], ...] = ()


def _mate_mechanisms() -> tuple[_MateMechanism, ...]:
    return (
        _MateMechanism("queen-adjacent-capture", chess.C2),
        _MateMechanism("queen-file-capture", chess.B3),
        _MateMechanism("queen-diagonal-capture", chess.D3),
        _MateMechanism(
            "supported-queen-adjacent-capture",
            chess.C2,
            ((chess.D1, chess.Piece(chess.ROOK, chess.WHITE)),),
        ),
        _MateMechanism(
            "supported-queen-file-capture",
            chess.B3,
            ((chess.D1, chess.Piece(chess.BISHOP, chess.WHITE)),),
        ),
        _MateMechanism(
            "supported-queen-diagonal-capture",
            chess.D3,
            ((chess.D1, chess.Piece(chess.KNIGHT, chess.WHITE)),),
        ),
    )


def _board_transforms() -> tuple[_BoardTransform, ...]:
    return (
        _BoardTransform("identity", lambda file_index, rank_index: (file_index, rank_index)),
        _BoardTransform("mirror-file", lambda file_index, rank_index: (7 - file_index, rank_index)),
        _BoardTransform("mirror-rank", lambda file_index, rank_index: (file_index, 7 - rank_index)),
        _BoardTransform(
            "rotate-180",
            lambda file_index, rank_index: (7 - file_index, 7 - rank_index),
        ),
        _BoardTransform("transpose", lambda file_index, rank_index: (rank_index, file_index)),
        _BoardTransform(
            "anti-transpose",
            lambda file_index, rank_index: (7 - rank_index, 7 - file_index),
        ),
        _BoardTransform("rotate-90", lambda file_index, rank_index: (rank_index, 7 - file_index)),
        _BoardTransform("rotate-270", lambda file_index, rank_index: (7 - rank_index, file_index)),
    )


def _spectator_squares() -> tuple[chess.Square, ...]:
    occupied = frozenset({chess.A1, chess.B1, chess.C1})
    mechanism_squares = frozenset(mechanism.queen_square for mechanism in _mate_mechanisms())
    support_squares = frozenset(
        square
        for mechanism in _mate_mechanisms()
        for square, _piece in mechanism.support_pieces
    )
    unsafe = frozenset({chess.B2})
    return tuple(
        square
        for square in chess.SQUARES
        if square not in occupied
        and square not in mechanism_squares
        and square not in support_squares
        and square not in unsafe
    )


def _family_capacity() -> int:
    return (
        len(_mate_mechanisms())
        * len(_board_transforms())
        * (1 << len(_spectator_squares()))
    )


def _max_spectator_count_for_cardinality(cardinality: int) -> int:
    _require_sample_cardinality(cardinality)
    return _enabled_spectator_count_for_cardinality(cardinality)


def _oracle_inference_compute_for_cardinality(cardinality: int) -> int:
    return 2 + 8 * _max_spectator_count_for_cardinality(cardinality)


def _global_sample_index(*, cardinality: int, local_index: int) -> int:
    _require_sample_cardinality(cardinality)
    if type(local_index) is not int or local_index < 0 or local_index >= cardinality:
        raise ObservationGenerationError("Chess local sample index is outside cardinality")
    return _global_sample_indices(cardinality=cardinality, local_indices=(local_index,))[0]


def _global_sample_indices(
    *,
    cardinality: int,
    local_indices: Sequence[int],
) -> tuple[int, ...]:
    _require_sample_cardinality(cardinality)
    invalid_indices = tuple(
        local_index
        for local_index in local_indices
        if type(local_index) is not int or local_index < 0 or local_index >= cardinality
    )
    if invalid_indices:
        raise ObservationGenerationError("Chess local sample index is outside cardinality")
    enabled_capacity = _enabled_family_capacity_for_cardinality(cardinality)
    lower_bound = _family_index_lower_bound_for_cardinality(cardinality)
    if cardinality > enabled_capacity - lower_bound:
        lower_bound = 0
    offset = cardinality * (cardinality - 1) // 2
    if offset + cardinality > enabled_capacity:
        offset = lower_bound + offset % (enabled_capacity - lower_bound - cardinality + 1)
    return tuple(offset + local_index for local_index in local_indices)


def _enabled_family_capacity_for_cardinality(cardinality: int) -> int:
    return len(_mate_mechanisms()) * len(_board_transforms()) * (
        1 << _enabled_spectator_count_for_cardinality(cardinality)
    )


def _enabled_spectator_count_for_cardinality(cardinality: int) -> int:
    _require_sample_cardinality(cardinality)
    base_count = len(_mate_mechanisms()) * len(_board_transforms())
    required_family_size = cardinality * (cardinality + 1) // 2
    required_masks = math.ceil(required_family_size / base_count)
    if required_masks <= 1:
        return 0
    return min(
        len(_spectator_squares()),
        math.ceil(math.log2(required_masks)),
    )


def _family_index_lower_bound_for_cardinality(cardinality: int) -> int:
    _require_sample_cardinality(cardinality)
    spectator_count = _enabled_spectator_count_for_cardinality(cardinality)
    if spectator_count == 0:
        return 0
    return len(_mate_mechanisms()) * len(_board_transforms()) * (
        1 << (spectator_count - 1)
    )


def _representative_local_indices(cardinality: int) -> tuple[int, ...]:
    _require_sample_cardinality(cardinality)
    count = min(cardinality, _preview_representative_limit)
    if count == 1:
        return (0,)
    return tuple(
        index * (cardinality - 1) // (count - 1)
        for index in range(count)
    )


def _sample_local_indices(
    *,
    rng: random.Random,
    cardinality: int,
    sample_count: int,
) -> tuple[int, ...]:
    _require_sample_cardinality(cardinality)
    if type(sample_count) is not int or sample_count < 0:
        raise ObservationGenerationError("Chess sample count must be non-negative")
    if sample_count == 0:
        return ()
    indices: list[int] = []
    while len(indices) + cardinality <= sample_count:
        shuffled = list(range(cardinality))
        rng.shuffle(shuffled)
        indices.extend(shuffled)
    remaining = sample_count - len(indices)
    if remaining:
        indices.extend(rng.sample(range(cardinality), remaining))
    return tuple(indices)


def _spectator_mask_for_combination_rank(rank: int) -> int:
    if type(rank) is not int or rank < 0:
        raise ObservationGenerationError("Chess spectator combination rank must be non-negative")
    remaining_rank = rank
    spectator_square_count = len(_spectator_squares())
    for spectator_count in range(spectator_square_count + 1):
        count_at_weight = math.comb(spectator_square_count, spectator_count)
        if remaining_rank < count_at_weight:
            return _spectator_mask_for_weight_rank(
                spectator_count=spectator_count,
                rank=remaining_rank,
            )
        remaining_rank -= count_at_weight
    raise ObservationGenerationError("Chess spectator combination rank exceeds capacity")


def _spectator_count_for_combination_rank(rank: int) -> int:
    return _spectator_mask_for_combination_rank(rank).bit_count()


def _spectator_mask_for_weight_rank(*, spectator_count: int, rank: int) -> int:
    if spectator_count == 0:
        if rank != 0:
            raise ObservationGenerationError("Chess zero-spectator rank must be zero")
        return 0
    remaining_rank = rank
    selected = spectator_count
    mask = 0
    for bit_index in range(len(_spectator_squares())):
        if selected == 0:
            break
        skip_count = math.comb(len(_spectator_squares()) - bit_index - 1, selected)
        if remaining_rank < skip_count:
            continue
        remaining_rank -= skip_count
        mask |= 1 << bit_index
        selected -= 1
    if selected != 0 or remaining_rank != 0:
        raise ObservationGenerationError("Chess spectator rank exceeds weight capacity")
    return mask


def _require_sample_cardinality(cardinality: int) -> None:
    if type(cardinality) is not int or cardinality < 1:
        raise ObservationGenerationError("Chess sample cardinality must be positive")
    if cardinality > _family_capacity():
        raise ObservationGenerationError("Chess sample cardinality exceeds generator capacity")


def _ceil_cardinality(complexity: float) -> int:
    if not math.isfinite(complexity):
        raise ObservationGenerationError("complexity must be finite")
    if complexity <= 0.0:
        return 1
    cardinality = 2**complexity
    rounded = round(cardinality)
    if math.isclose(cardinality, rounded, rel_tol=1e-12, abs_tol=1e-9):
        return max(1, rounded)
    return max(1, math.ceil(cardinality))


def _floor_cardinality(complexity: float) -> int:
    if not math.isfinite(complexity):
        raise ObservationGenerationError("complexity must be finite")
    if complexity < 0.0:
        return 0
    cardinality = 2**complexity
    rounded = round(cardinality)
    if math.isclose(cardinality, rounded, rel_tol=1e-12, abs_tol=1e-9):
        return min(_family_capacity(), rounded)
    return min(_family_capacity(), math.floor(cardinality))


def _base_mate_pieces(
    *,
    mechanism: _MateMechanism,
    spectator_combination_rank: int = 0,
) -> tuple[tuple[chess.Square, chess.Piece], ...]:
    pieces: list[tuple[chess.Square, chess.Piece]] = [
        (chess.A1, chess.Piece(chess.KING, chess.BLACK)),
        (chess.B1, chess.Piece(chess.ROOK, chess.BLACK)),
        (chess.C1, chess.Piece(chess.KING, chess.WHITE)),
        (mechanism.queen_square, chess.Piece(chess.QUEEN, chess.WHITE)),
        *mechanism.support_pieces,
    ]
    spectator_mask = _spectator_mask_for_combination_rank(spectator_combination_rank)
    pieces.extend(
        (square, chess.Piece(chess.KNIGHT, chess.WHITE))
        for bit_index, square in enumerate(_spectator_squares())
        if spectator_mask & (1 << bit_index)
    )
    return tuple(pieces)


def _board_from_pieces(pieces: Sequence[tuple[chess.Square, chess.Piece]]) -> chess.Board:
    board = chess.Board.empty()
    board.turn = chess.WHITE
    board.castling_rights = 0
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    for square, piece in pieces:
        board.set_piece_at(square, piece)
    return board


def _transformed_square(square: chess.Square, *, transform: _BoardTransform) -> chess.Square:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    transformed_file, transformed_rank = transform.square(file_index, rank_index)
    return chess.square(transformed_file, transformed_rank)


def _transformed_move_uci(
    *,
    from_square: chess.Square,
    to_square: chess.Square,
    transform: _BoardTransform,
) -> str:
    return (
        chess.square_name(_transformed_square(from_square, transform=transform))
        + chess.square_name(_transformed_square(to_square, transform=transform))
    )


def _position_for_sample_index(index: int) -> _MateInOnePosition:
    if type(index) is not int or index < 0:
        raise ObservationGenerationError("Chess sample index must be non-negative")
    mechanisms = _mate_mechanisms()
    transforms = _board_transforms()
    base_count = len(mechanisms) * len(transforms)
    base_index = index % base_count
    spectator_combination_rank = index // base_count
    if spectator_combination_rank >= (1 << len(_spectator_squares())):
        raise ObservationGenerationError("Chess sample index exceeds generator capacity")
    mechanism = mechanisms[base_index // len(transforms)]
    transform = transforms[base_index % len(transforms)]
    base_pieces = _base_mate_pieces(
        mechanism=mechanism,
        spectator_combination_rank=spectator_combination_rank,
    )
    board_pieces = tuple(
        (_transformed_square(square, transform=transform), piece)
        for square, piece in base_pieces
    )
    board = _board_from_pieces(board_pieces)
    return _mate_in_one_position(
        board.fen(),
        mechanism_name=mechanism.name,
        transform_name=transform.name,
        family_index=index,
        spectator_count=_spectator_count_for_combination_rank(spectator_combination_rank),
    )


def _mate_in_one_position(
    fen: str,
    *,
    mechanism_name: str,
    transform_name: str,
    family_index: int,
    spectator_count: int,
) -> _MateInOnePosition:
    board = chess.Board(fen)
    legal_moves = tuple(sorted(move.uci() for move in board.legal_moves))
    mate_moves = tuple(
        sorted(
            move.uci()
            for move in board.legal_moves
            if _is_checkmate_after_move(board, move)
        )
    )
    if not legal_moves:
        raise ObservationGenerationError(f"Chess position has no legal moves: {fen}")
    if not mate_moves:
        raise ObservationGenerationError(f"Chess position has no mate-in-one moves: {fen}")
    unknown_moves = tuple(
        move
        for move in (*legal_moves, *mate_moves)
        if move not in all_meaningful_uci_moves
    )
    if unknown_moves:
        raise ObservationGenerationError(
            "Chess position contains moves outside the declared outcome space"
        )
    return _MateInOnePosition(
        fen=fen,
        legal_moves=legal_moves,
        mate_moves=mate_moves,
        mechanism_name=mechanism_name,
        transform_name=transform_name,
        family_index=family_index,
        spectator_count=spectator_count,
    )


def _move_piece_symbols(*, fen: str, moves: Sequence[str]) -> tuple[str, ...]:
    board = chess.Board(fen)
    symbols: set[str] = set()
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ObservationGenerationError(
                f"Chess move has no piece on its source square: {move_uci}"
            )
        symbols.add(piece.symbol().upper())
    return tuple(sorted(symbols))


def _board_piece_symbols(board: chess.Board) -> tuple[str, ...]:
    return tuple(sorted({piece.symbol().upper() for piece in board.piece_map().values()}))


def _representative_quality_flags(position: _MateInOnePosition) -> tuple[str, ...]:
    flags: list[str] = []
    if len(chess.Board(position.fen).piece_map()) <= 4:
        flags.append("minimal-material")
    if len(position.mate_moves) == 1:
        flags.append("single-mate-move")
    if position.mate_move_piece_symbols == ("Q",):
        flags.append("queen-only-mate")
    if len(position.legal_move_piece_symbols) == 1:
        flags.append("single-legal-piece-type")
    return tuple(flags)


def _is_checkmate_after_move(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()


def _latent_coordinates(
    position: _MateInOnePosition,
    *,
    sample_space_cardinality: int,
) -> tuple[Mapping[str, object], ...]:
    return (
        {
            "name": "benchmarks.chess.sample-space.cardinality",
            "role": "complexity",
            "degree_measure": {
                "kind": "discrete-log2-count",
                "count": sample_space_cardinality,
            },
            "multiplicity": 1,
            "values": sample_space_cardinality,
        },
        {
            "name": "benchmarks.chess.position.fen",
            "role": "content",
            "degree_measure": {"kind": "fixture-position", "count": 1.0},
            "multiplicity": 1,
            "values": position.fen,
        },
        {
            "name": "benchmarks.chess.position.legal-move-count",
            "role": "content",
            "degree_measure": {
                "kind": "finite-set",
                "count": position.legal_move_count,
            },
            "multiplicity": 1,
            "values": position.legal_move_count,
        },
        {
            "name": "benchmarks.chess.position.legal-moves",
            "role": "content",
            "degree_measure": {
                "kind": "finite-set",
                "count": position.legal_move_count,
            },
            "multiplicity": position.legal_move_count,
            "values": list(position.legal_moves),
        },
        {
            "name": "benchmarks.chess.position.mate-in-one-moves",
            "role": "target",
            "degree_measure": {
                "kind": "finite-set",
                "count": len(position.mate_moves),
            },
            "multiplicity": len(position.mate_moves),
            "values": list(position.mate_moves),
        },
        {
            "name": "benchmarks.chess.position.representative-piece-coverage",
            "role": "content",
            "degree_measure": {
                "kind": "finite-set",
                "count": len(position.legal_move_piece_symbols),
            },
            "multiplicity": len(position.legal_move_piece_symbols),
            "values": {
                "legal_move_piece_symbols": list(position.legal_move_piece_symbols),
                "mate_move_piece_symbols": list(position.mate_move_piece_symbols),
                "analysis": dict(position.representative_analysis()),
            },
        },
    )


def _samples_for_global_indices(
    *,
    global_indices: Sequence[int],
    outcome_ids: tuple[str, ...] | None,
    sample_space_cardinality: int,
    complexity: float,
    full_metadata: bool,
) -> tuple[GeneratedSample, ...]:
    if full_metadata:
        return tuple(
            _full_sample(
                index=index,
                position=_position_for_sample_index(global_index),
                sample_space_cardinality=sample_space_cardinality,
                complexity=complexity,
            )
            for index, global_index in enumerate(global_indices)
        )
    if outcome_ids is None:
        raise ObservationGenerationError("lightweight Chess samples require outcome_ids")
    base_count = len(_mate_mechanisms()) * len(_board_transforms())
    target_indices_by_base = _base_target_indices(outcome_ids)
    complexity_value = ComplexityValue(value=complexity)
    return tuple(
        GeneratedSample(
            index=index,
            outcome_id=outcome_ids[target_indices_by_base[global_index % base_count]],
            complexity=complexity,
            complexity_value=complexity_value,
        )
        for index, global_index in enumerate(global_indices)
    )


def _full_sample(
    *,
    index: int,
    position: _MateInOnePosition,
    sample_space_cardinality: int,
    complexity: float,
) -> GeneratedSample:
    return GeneratedSample(
        index=index,
        outcome_id=position.mate_moves[0],
        complexity=complexity,
        complexity_value=ComplexityValue(value=complexity),
        available_outcome_ids=position.legal_moves,
        observable_state_id=position.observation_id,
        latent_coordinates=_latent_coordinates(
            position,
            sample_space_cardinality=sample_space_cardinality,
        ),
    )


def _sample_preview_record(sample: GeneratedSample) -> Mapping[str, object]:
    record: dict[str, object] = {
        "index": sample.index,
        "outcome_id": sample.outcome_id,
        "complexity": sample.complexity,
        "complexity_value": (
            None
            if sample.complexity_value is None
            else sample.complexity_value.to_record()
        ),
        "observable_state_id": sample.observable_state_id,
        "available_outcome_ids": list(sample.available_outcome_ids),
        "latent_coordinates": [
            dict(coordinate) for coordinate in sample.latent_coordinates
        ],
    }
    if sample.target_distribution is not None:
        record["target_distribution"] = [
            {"outcome_id": outcome_id, "probability": probability}
            for outcome_id, probability in sample.target_distribution.items()
        ]
    image_data_url = _sample_board_image_data_url(sample)
    if image_data_url is not None:
        record["image_data_url"] = image_data_url
        record["image_overlay"] = _sample_board_image_overlay(sample)
    return record


def _sample_board_image_data_url(sample: GeneratedSample) -> str | None:
    observable_state_id = sample.observable_state_id
    if observable_state_id is None or not observable_state_id.startswith("fen:"):
        return None
    fen = observable_state_id.removeprefix("fen:")
    return _board_svg_data_url(fen)


def _board_svg_data_url(fen: str) -> str:
    board = chess.Board(fen)
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {_board_preview_size} {_board_preview_size}" '
            f'width="{_board_preview_size}" height="{_board_preview_size}" '
            f'role="img" aria-label="Chess board">'
        ),
        f'<rect width="{_board_preview_size}" height="{_board_preview_size}" fill="#f0d9b5"/>',
    ]
    for rank in range(8):
        for file_index in range(8):
            x = file_index * _board_preview_square_size
            y = (7 - rank) * _board_preview_square_size
            fill = "#b58863" if (rank + file_index) % 2 else "#f0d9b5"
            elements.append(
                f'<rect x="{x}" y="{y}" width="{_board_preview_square_size}" '
                f'height="{_board_preview_square_size}" fill="{fill}"/>'
            )
    for square, piece in sorted(board.piece_map().items()):
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        center_x = file_index * _board_preview_square_size + _board_preview_square_size / 2
        center_y = (7 - rank) * _board_preview_square_size + _board_preview_square_size / 2
        elements.append(_piece_svg_shape(piece, center_x=center_x, center_y=center_y))
    elements.append("</svg>")
    svg = "".join(elements)
    encoded = b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _sample_board_image_overlay(sample: GeneratedSample) -> Mapping[str, object]:
    moves: list[dict[str, object]] = []
    target_probabilities = sample.target_distribution_or_one_hot()
    for outcome_id in sample.available_outcome_ids:
        if len(outcome_id) < 4:
            continue
        from_square = chess.parse_square(outcome_id[:2])
        to_square = chess.parse_square(outcome_id[2:4])
        moves.append(
            {
                "from": _square_grid_coordinate(from_square),
                "to": _square_grid_coordinate(to_square),
                "target_probability": target_probabilities.get(outcome_id, 0.0),
            }
        )
    return {
        "kind": "grid-move-highlights",
        "columns": 8,
        "rows": 8,
        "moves": moves,
    }


def _square_grid_coordinate(square: chess.Square) -> list[int]:
    return [chess.square_file(square), 7 - chess.square_rank(square)]


def _piece_svg_shape(piece: chess.Piece, *, center_x: float, center_y: float) -> str:
    fill = "#f8f8f8" if piece.color == chess.WHITE else "#202020"
    stroke = "#202020" if piece.color == chess.WHITE else "#f8f8f8"
    highlight = "#ffffff" if piece.color == chess.WHITE else "#4a4a4a"
    scale = _board_preview_square_size / 32
    group_start = (
        f'<g aria-label="{piece.symbol()}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round" '
        f'transform="translate({center_x:.1f} {center_y:.1f}) scale({scale:.3f}) '
        f'translate({-center_x:.1f} {-center_y:.1f})">'
    )
    base = (
        f'<path d="M {center_x - 11:.1f} {center_y + 12:.1f} '
        f'H {center_x + 11:.1f} '
        f'Q {center_x + 9:.1f} {center_y + 8:.1f} {center_x + 5:.1f} {center_y + 7:.1f} '
        f'H {center_x - 5:.1f} '
        f'Q {center_x - 9:.1f} {center_y + 8:.1f} {center_x - 11:.1f} {center_y + 12:.1f} Z"/>'
    )
    collar = (
        f'<path d="M {center_x - 7:.1f} {center_y + 5:.1f} '
        f'H {center_x + 7:.1f} '
        f'Q {center_x + 5:.1f} {center_y + 1:.1f} {center_x:.1f} {center_y + 1:.1f} '
        f'Q {center_x - 5:.1f} {center_y + 1:.1f} {center_x - 7:.1f} {center_y + 5:.1f} Z"/>'
    )
    stem = (
        f'<path d="M {center_x - 4:.1f} {center_y + 7:.1f} '
        f'C {center_x - 2:.1f} {center_y + 1:.1f} '
        f'{center_x - 2:.1f} {center_y - 4:.1f} '
        f'{center_x:.1f} {center_y - 8:.1f} '
        f'C {center_x + 2:.1f} {center_y - 4:.1f} '
        f'{center_x + 2:.1f} {center_y + 1:.1f} '
        f'{center_x + 4:.1f} {center_y + 7:.1f} Z"/>'
    )
    gleam = (
        f'<path d="M {center_x - 4:.1f} {center_y + 8:.1f} '
        f'C {center_x - 7:.1f} {center_y + 5:.1f} '
        f'{center_x - 7:.1f} {center_y:.1f} '
        f'{center_x - 3:.1f} {center_y - 6:.1f}" '
        f'fill="none" stroke="{highlight}" stroke-opacity="0.55" stroke-width="1"/>'
    )
    if piece.piece_type == chess.KING:
        return (
            f"{group_start}{base}{collar}{stem}"
            f'<circle cx="{center_x:.1f}" cy="{center_y - 8:.1f}" r="5.2"/>'
            f'<path d="M {center_x:.1f} {center_y - 7:.1f} '
            f'V {center_y - 18:.1f} M {center_x - 4:.1f} {center_y - 14:.1f} '
            f'H {center_x + 4:.1f}" fill="none"/>'
            f"{gleam}</g>"
        )
    if piece.piece_type == chess.QUEEN:
        return (
            f"{group_start}{base}{collar}"
            f'<path d="M {center_x - 9:.1f} {center_y - 4:.1f} '
            f'L {center_x - 6:.1f} {center_y - 15:.1f} '
            f'L {center_x - 2:.1f} {center_y - 6:.1f} '
            f'L {center_x:.1f} {center_y - 17:.1f} '
            f'L {center_x + 2:.1f} {center_y - 6:.1f} '
            f'L {center_x + 6:.1f} {center_y - 15:.1f} '
            f'L {center_x + 9:.1f} {center_y - 4:.1f} Z"/>'
            f'<circle cx="{center_x - 6:.1f}" cy="{center_y - 15:.1f}" r="2.0"/>'
            f'<circle cx="{center_x:.1f}" cy="{center_y - 17:.1f}" r="2.0"/>'
            f'<circle cx="{center_x + 6:.1f}" cy="{center_y - 15:.1f}" r="2.0"/>'
            f"{gleam}</g>"
        )
    if piece.piece_type == chess.ROOK:
        return (
            f"{group_start}{base}{collar}"
            f'<path d="M {center_x - 8:.1f} {center_y + 1:.1f} '
            f'V {center_y - 13:.1f} H {center_x - 5:.1f} V {center_y - 9:.1f} '
            f'H {center_x - 1.5:.1f} V {center_y - 13:.1f} H {center_x + 1.5:.1f} '
            f'V {center_y - 9:.1f} H {center_x + 5:.1f} V {center_y - 13:.1f} '
            f'H {center_x + 8:.1f} V {center_y + 1:.1f} Z"/>'
            f"{gleam}</g>"
        )
    if piece.piece_type == chess.BISHOP:
        return (
            f"{group_start}{base}{collar}{stem}"
            f'<ellipse cx="{center_x:.1f}" cy="{center_y - 8:.1f}" rx="6.5" ry="8.5"/>'
            f'<path d="M {center_x - 3:.1f} {center_y - 2:.1f} '
            f'L {center_x + 4:.1f} {center_y - 12:.1f}" fill="none"/>'
            f"{gleam}</g>"
        )
    if piece.piece_type == chess.KNIGHT:
        return (
            f"{group_start}{base}"
            f'<path d="M {center_x - 5:.1f} {center_y + 7:.1f} '
            f'C {center_x - 7:.1f} {center_y + 1:.1f} '
            f'{center_x - 5:.1f} {center_y - 7:.1f} '
            f'{center_x + 3:.1f} {center_y - 15:.1f} '
            f'C {center_x + 1:.1f} {center_y - 9:.1f} '
            f'{center_x + 7:.1f} {center_y - 7:.1f} '
            f'{center_x + 8:.1f} {center_y - 2:.1f} '
            f'C {center_x + 6:.1f} {center_y - 1:.1f} '
            f'{center_x + 4:.1f} {center_y:.1f} '
            f'{center_x + 3:.1f} {center_y + 2:.1f} '
            f'L {center_x + 6:.1f} {center_y + 7:.1f} Z"/>'
            f'<circle cx="{center_x + 2:.1f}" cy="{center_y - 8:.1f}" '
            f'r="0.9" fill="{stroke}" stroke="none"/>'
            f"{gleam}</g>"
        )
    return (
        f"{group_start}{base}{collar}{stem}"
        f'<circle cx="{center_x:.1f}" cy="{center_y - 9:.1f}" r="5.8"/>'
        f"{gleam}</g>"
    )


def _tensor_batch_for_global_indices(
    *,
    runtime: TensorRuntime | None,
    global_indices: Sequence[int],
    cardinality: int,
    outcome_ids: tuple[str, ...] | None,
    sample_shape: tuple[int, ...],
) -> tuple[Any | None, Any | None]:
    if runtime is None:
        return None, None
    if outcome_ids is None:
        raise ObservationGenerationError("tensor generation requires outcome_ids")
    return _tensor_batch_from_global_indices(
        runtime=runtime,
        global_indices=global_indices,
        cardinality=cardinality,
        outcome_ids=outcome_ids,
        sample_shape=sample_shape,
    )


def _tensor_batch_from_global_indices(
    *,
    runtime: TensorRuntime,
    global_indices: Sequence[int],
    cardinality: int,
    outcome_ids: tuple[str, ...],
    sample_shape: tuple[int, ...],
) -> tuple[Any, Any]:
    backend = tensor_runtime_backend(runtime)
    sample_count = len(global_indices)
    fields = backend.zeros(
        (sample_count, *_tensor_shape),
        dtype=backend.float32,
        device=runtime.device,
    )
    targets = backend.zeros(
        (sample_count, len(outcome_ids)),
        dtype=backend.float32,
        device=runtime.device,
    )
    fields[:, 12, :, :] = 1.0
    if sample_count == 0:
        return (
            fields.reshape((*sample_shape, *_tensor_shape)),
            targets.reshape((*sample_shape, len(outcome_ids))),
        )

    indices = backend.tensor(global_indices, dtype=backend.long, device=runtime.device)
    enabled_spectator_count = _enabled_spectator_count_for_cardinality(cardinality)
    transform_table = _device_long_tensor(
        runtime,
        _transform_square_table(),
    )
    queen_squares = _device_long_tensor(runtime, _queen_squares())
    support_squares = _device_long_tensor(runtime, _support_squares())
    support_planes = _device_long_tensor(runtime, _support_planes())
    target_indices_by_base = _device_long_tensor(
        runtime,
        _base_target_indices(outcome_ids),
    )
    spectator_squares = _device_long_tensor(
        runtime,
        _spectator_squares() if enabled_spectator_count > 0 else (0,),
    )
    combination_table = _device_long_tensor(
        runtime,
        _spectator_combination_table() if enabled_spectator_count > 0 else ((1,),),
    )
    _render_tensor_batch_portable(
        runtime=runtime,
        fields=fields,
        targets=targets,
        indices=indices,
        transform_table=transform_table,
        queen_squares=queen_squares,
        support_squares=support_squares,
        support_planes=support_planes,
        target_indices_by_base=target_indices_by_base,
        spectator_squares=spectator_squares,
        combination_table=combination_table,
        outcome_count=len(outcome_ids),
        enabled_spectator_count=enabled_spectator_count,
    )
    return (
        fields.reshape((*sample_shape, *_tensor_shape)),
        targets.reshape((*sample_shape, len(outcome_ids))),
    )


def _render_tensor_batch_portable(
    *,
    runtime: TensorRuntime,
    fields: Any,
    targets: Any,
    indices: Any,
    transform_table: Any,
    queen_squares: Any,
    support_squares: Any,
    support_planes: Any,
    target_indices_by_base: Any,
    spectator_squares: Any,
    combination_table: Any,
    outcome_count: int,
    enabled_spectator_count: int,
) -> None:
    _ = outcome_count
    backend = tensor_runtime_backend(runtime)
    sample_count = int(indices.numel())
    mechanism_count = len(_mate_mechanisms())
    transform_count = len(_board_transforms())
    base_count = mechanism_count * transform_count
    sample_indices = backend.arange(sample_count, dtype=backend.long, device=runtime.device)
    sample_base_indices = indices.remainder(base_count)
    sample_target_indices = target_indices_by_base[sample_base_indices]
    targets[sample_indices, sample_target_indices] = 1.0

    piece_slot_count = 5 + enabled_spectator_count
    slot_offsets = backend.arange(
        sample_count * piece_slot_count,
        dtype=backend.long,
        device=runtime.device,
    )
    slot_sample_indices = backend.div(
        slot_offsets,
        piece_slot_count,
        rounding_mode="floor",
    )
    piece_slots = slot_offsets.remainder(piece_slot_count)
    slot_global_indices = indices[slot_sample_indices]
    base_indices = slot_global_indices.remainder(base_count)
    mechanism_indices = backend.div(base_indices, transform_count, rounding_mode="floor")
    transform_indices = base_indices.remainder(transform_count)
    spectator_combination_ranks = backend.div(
        slot_global_indices,
        base_count,
        rounding_mode="floor",
    )

    queen_source_squares = queen_squares[mechanism_indices]
    support_source_squares = support_squares[mechanism_indices]
    support_source_planes = support_planes[mechanism_indices]
    black_king_plane = _piece_plane(chess.Piece(chess.KING, chess.BLACK))
    black_rook_plane = _piece_plane(chess.Piece(chess.ROOK, chess.BLACK))
    white_king_plane = _piece_plane(chess.Piece(chess.KING, chess.WHITE))
    white_queen_plane = _piece_plane(chess.Piece(chess.QUEEN, chess.WHITE))
    field_planes = backend.where(
        piece_slots == 0,
        backend.full_like(piece_slots, black_king_plane),
        backend.full_like(piece_slots, black_rook_plane),
    )
    field_planes = backend.where(
        piece_slots == 2,
        backend.full_like(piece_slots, white_king_plane),
        field_planes,
    )
    field_planes = backend.where(
        piece_slots == 3,
        backend.full_like(piece_slots, white_queen_plane),
        field_planes,
    )
    field_planes = backend.where(piece_slots == 4, support_source_planes, field_planes)
    source_squares = backend.where(
        piece_slots == 0,
        backend.full_like(piece_slots, chess.A1),
        backend.full_like(piece_slots, chess.B1),
    )
    source_squares = backend.where(
        piece_slots == 2,
        backend.full_like(piece_slots, chess.C1),
        source_squares,
    )
    source_squares = backend.where(piece_slots == 3, queen_source_squares, source_squares)
    source_squares = backend.where(piece_slots == 4, support_source_squares, source_squares)
    write_mask = (piece_slots < 4) | ((piece_slots == 4) & (support_source_squares >= 0))

    spectator_square_count = len(_spectator_squares())
    if enabled_spectator_count > 0:
        selected_counts = backend.zeros_like(piece_slots)
        ranks_within_count = spectator_combination_ranks
        unresolved = backend.ones_like(piece_slots, dtype=backend.bool)
        for selected_count in range(enabled_spectator_count + 1):
            count_at_weight = combination_table[spectator_square_count, selected_count]
            selected = unresolved & (ranks_within_count < count_at_weight)
            selected_counts = backend.where(
                selected,
                backend.full_like(selected_counts, selected_count),
                selected_counts,
            )
            unresolved = unresolved & ~selected
            ranks_within_count = backend.where(
                unresolved,
                ranks_within_count - count_at_weight,
                ranks_within_count,
            )
        spectator_ordinals = piece_slots - 5
        remaining_selected = selected_counts
        remaining_rank = ranks_within_count
        chosen_so_far = backend.zeros_like(piece_slots)
        selected_spectator_squares = backend.zeros_like(piece_slots)
        selected_spectator_found = backend.zeros_like(piece_slots, dtype=backend.bool)
        for bit_index in range(spectator_square_count):
            remaining_slots = spectator_square_count - bit_index - 1
            skip_counts = combination_table[remaining_slots, remaining_selected]
            choose = (remaining_selected > 0) & (remaining_rank >= skip_counts)
            use_square = choose & (chosen_so_far == spectator_ordinals)
            selected_spectator_squares = backend.where(
                use_square,
                spectator_squares[bit_index],
                selected_spectator_squares,
            )
            selected_spectator_found = selected_spectator_found | use_square
            remaining_rank = backend.where(choose, remaining_rank - skip_counts, remaining_rank)
            remaining_selected = backend.where(
                choose,
                remaining_selected - backend.ones_like(remaining_selected),
                remaining_selected,
            )
            chosen_so_far = backend.where(
                choose,
                chosen_so_far + backend.ones_like(chosen_so_far),
                chosen_so_far,
            )
        spectator_mask = (
            (piece_slots >= 5)
            & (spectator_ordinals < selected_counts)
            & selected_spectator_found
        )
        source_squares = backend.where(spectator_mask, selected_spectator_squares, source_squares)
        field_planes = backend.where(
            spectator_mask,
            backend.full_like(field_planes, _piece_plane(chess.Piece(chess.KNIGHT, chess.WHITE))),
            field_planes,
        )
        write_mask = write_mask | spectator_mask

    transformed_squares = transform_table[transform_indices, source_squares]
    field_offsets = (
        (slot_sample_indices * _tensor_shape[0] + field_planes)
        * (_tensor_shape[1] * _tensor_shape[2])
        + transformed_squares
    )
    fields.reshape(-1)[field_offsets[write_mask]] = 1.0


def _device_long_tensor(
    runtime: TensorRuntime,
    values: object,
) -> Any:
    backend = tensor_runtime_backend(runtime)
    return backend.tensor(values, dtype=backend.long, device=runtime.device)


def _transform_square_table() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(_transformed_square(square, transform=transform) for square in chess.SQUARES)
        for transform in _board_transforms()
    )


def _queen_squares() -> tuple[int, ...]:
    return tuple(mechanism.queen_square for mechanism in _mate_mechanisms())


def _support_squares() -> tuple[int, ...]:
    support_squares: list[int] = []
    for mechanism in _mate_mechanisms():
        if len(mechanism.support_pieces) > 1:
            raise ObservationGenerationError("Chess GPU renderer expects at most one support piece")
        support_squares.append(
            -1 if not mechanism.support_pieces else mechanism.support_pieces[0][0]
        )
    return tuple(support_squares)


def _support_planes() -> tuple[int, ...]:
    support_planes: list[int] = []
    for mechanism in _mate_mechanisms():
        support_planes.append(
            -1
            if not mechanism.support_pieces
            else _piece_plane(mechanism.support_pieces[0][1])
        )
    return tuple(support_planes)


def _spectator_combination_table() -> tuple[tuple[int, ...], ...]:
    square_count = len(_spectator_squares())
    return tuple(
        tuple(math.comb(remaining_slots, selected) for selected in range(square_count + 1))
        for remaining_slots in range(square_count + 1)
    )


def _base_target_indices(outcome_ids: tuple[str, ...]) -> tuple[int, ...]:
    outcome_index_by_id = {outcome_id: index for index, outcome_id in enumerate(outcome_ids)}
    target_indices: list[int] = []
    for mechanism in _mate_mechanisms():
        for transform in _board_transforms():
            target_move = _transformed_move_uci(
                from_square=mechanism.queen_square,
                to_square=chess.B1,
                transform=transform,
            )
            try:
                target_indices.append(outcome_index_by_id[target_move])
            except KeyError as error:
                raise ObservationGenerationError(
                    "tensor generation outcome_ids do not cover generated Chess moves"
                ) from error
    return tuple(target_indices)


def _piece_plane(piece: chess.Piece) -> int:
    piece_offsets = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    color_offset = 0 if piece.color == chess.WHITE else 6
    return color_offset + piece_offsets[piece.piece_type]


def _all_meaningful_uci_moves() -> tuple[str, ...]:
    moves: set[str] = set()
    for file_index in range(8):
        for rank_index in range(8):
            from_square = _square_name(file_index, rank_index)
            moves.update(_non_promotion_moves_from(file_index, rank_index, from_square))
            moves.update(_promotion_moves_from(file_index, rank_index, from_square))
    moves.update(("e1g1", "e1c1", "e8g8", "e8c8"))
    return tuple(sorted(moves))


def _non_promotion_moves_from(
    file_index: int,
    rank_index: int,
    from_square: str,
) -> tuple[str, ...]:
    moves: set[str] = set()
    for file_delta, rank_delta in _ray_directions():
        for distance in range(1, 8):
            moves.update(
                _move_if_on_board(
                    from_square,
                    file_index + file_delta * distance,
                    rank_index + rank_delta * distance,
                )
            )
    for file_delta, rank_delta in _knight_deltas():
        moves.update(
            _move_if_on_board(
                from_square,
                file_index + file_delta,
                rank_index + rank_delta,
            )
        )
    for rank_delta, home_rank_index in ((1, 1), (-1, 6)):
        moves.update(_move_if_on_board(from_square, file_index, rank_index + rank_delta))
        moves.update(
            _move_if_on_board(
                from_square,
                file_index,
                rank_index + 2 * rank_delta,
                allowed=rank_index == home_rank_index,
            )
        )
        for file_delta in (-1, 1):
            moves.update(
                _move_if_on_board(
                    from_square,
                    file_index + file_delta,
                    rank_index + rank_delta,
                )
            )
    return tuple(moves)


def _promotion_moves_from(
    file_index: int,
    rank_index: int,
    from_square: str,
) -> tuple[str, ...]:
    moves: set[str] = set()
    for rank_delta, promotion_rank_index in ((1, 6), (-1, 1)):
        if rank_index != promotion_rank_index:
            continue
        for file_delta in (-1, 0, 1):
            to_file_index = file_index + file_delta
            to_rank_index = rank_index + rank_delta
            if not _is_on_board(to_file_index, to_rank_index):
                continue
            to_square = _square_name(to_file_index, to_rank_index)
            for promotion in ("q", "r", "b", "n"):
                moves.add(from_square + to_square + promotion)
    return tuple(moves)


def _move_if_on_board(
    from_square: str,
    to_file_index: int,
    to_rank_index: int,
    *,
    allowed: bool = True,
) -> tuple[str, ...]:
    if not allowed or not _is_on_board(to_file_index, to_rank_index):
        return ()
    return (from_square + _square_name(to_file_index, to_rank_index),)


def _ray_directions() -> tuple[tuple[int, int], ...]:
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _knight_deltas() -> tuple[tuple[int, int], ...]:
    return (
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    )


def _is_on_board(file_index: int, rank_index: int) -> bool:
    return 0 <= file_index < 8 and 0 <= rank_index < 8


def _square_name(file_index: int, rank_index: int) -> str:
    return chr(ord("a") + file_index) + str(rank_index + 1)


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


all_meaningful_uci_moves = _all_meaningful_uci_moves()

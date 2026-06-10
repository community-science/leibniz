"""Chess benchmark implementation entry point."""

from __future__ import annotations

import math
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import chess

from leibniz.benchmark_implementations import Benchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.observation_generation import (
    ComplexityRequest,
    ComplexityValue,
    GeneratedSample,
    GeneratedSampleSet,
    GenerationRequestOutcome,
    ObservationGenerationError,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.tensor_runtime import (
    TensorElementParameter,
    TensorElementProgram,
    TensorElementRecipe,
    TensorRuntime,
    tensor_runtime_construct_tensor,
)
from leibniz.timing import TimingCollector

__all__ = ["all_meaningful_uci_moves", "benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.chess.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.chess.uci-moves@0.1.0")
_tensor_shape = (18, 8, 8)
_board_preview_size = 512
_board_preview_square_size = _board_preview_size // 8

_mate_in_one_family_id = "corner-net-indexed-family"


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
        include_artifacts: bool = False,
        complexity_request: ComplexityRequest | None = None,
        sample_indices: Sequence[int] | None = None,
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
                request_outcome=_chess_unrealized_request_outcome(complexity_request),
            )

        sample_count = _sample_count(sample_shape)
        resolved_sample_indices = _sample_indices(
            sample_count=sample_count,
            sample_indices=sample_indices,
        )
        complexity = math.log2(sample_space.cardinality)
        selected_local_indices = _sample_local_indices(
            seed=seed,
            cardinality=sample_space.cardinality,
            sample_indices=resolved_sample_indices,
        )
        selected_global_indices = _global_sample_indices(
            cardinality=sample_space.cardinality,
            local_indices=selected_local_indices,
        )
        samples: tuple[GeneratedSample, ...] = ()
        if include_metadata:
            samples = _samples_for_global_indices(
                global_indices=selected_global_indices,
                outcome_ids=outcome_ids,
                sample_space_cardinality=sample_space.cardinality,
                complexity=complexity,
                full_metadata=runtime is None,
            )
            if include_artifacts:
                samples = tuple(_with_chess_artifacts(sample) for sample in samples)
        fields, targets = _tensor_batch_for_global_indices(
            runtime=runtime,
            seed=seed,
            sample_indices=resolved_sample_indices,
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


def _global_sample_index(*, cardinality: int, local_index: int) -> int:
    _require_sample_cardinality(cardinality)
    if type(local_index) is not int or local_index < 0 or local_index >= cardinality:
        raise ObservationGenerationError("Chess local sample index is outside cardinality")
    enabled_capacity = _enabled_family_capacity_for_cardinality(cardinality)
    lower_bound = _family_index_lower_bound_for_cardinality(cardinality)
    if cardinality > enabled_capacity - lower_bound:
        lower_bound = 0
    offset = cardinality * (cardinality - 1) // 2
    if offset + cardinality > enabled_capacity:
        offset = lower_bound + offset % (enabled_capacity - lower_bound - cardinality + 1)
    return offset + local_index


def _global_sample_indices(
    *,
    cardinality: int,
    local_indices: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        _global_sample_index(cardinality=cardinality, local_index=local_index)
        for local_index in local_indices
    )


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


def _sample_local_indices(
    *,
    seed: int,
    cardinality: int,
    sample_indices: Sequence[int],
) -> tuple[int, ...]:
    _require_sample_cardinality(cardinality)
    return tuple((seed + index) % cardinality for index in sample_indices)


def _spectator_mask_for_rank(rank: int) -> int:
    if type(rank) is not int or rank < 0:
        raise ObservationGenerationError("Chess spectator rank must be non-negative")
    if rank >= (1 << len(_spectator_squares())):
        raise ObservationGenerationError("Chess spectator rank exceeds capacity")
    return rank


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


def _chess_unrealized_request_outcome(
    request: ComplexityRequest | None,
) -> GenerationRequestOutcome:
    if request is None:
        return GenerationRequestOutcome(kind="unrepresentable-below-minimum")
    if _ceil_cardinality(request.minimum) > _family_capacity():
        return GenerationRequestOutcome(kind="exhausted-capacity")
    return GenerationRequestOutcome(kind="unrepresentable-below-minimum")


def _base_mate_pieces(
    *,
    mechanism: _MateMechanism,
    spectator_rank: int = 0,
) -> tuple[tuple[chess.Square, chess.Piece], ...]:
    pieces: list[tuple[chess.Square, chess.Piece]] = [
        (chess.A1, chess.Piece(chess.KING, chess.BLACK)),
        (chess.B1, chess.Piece(chess.ROOK, chess.BLACK)),
        (chess.C1, chess.Piece(chess.KING, chess.WHITE)),
        (mechanism.queen_square, chess.Piece(chess.QUEEN, chess.WHITE)),
        *mechanism.support_pieces,
    ]
    spectator_mask = _spectator_mask_for_rank(spectator_rank)
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
    spectator_rank = index // base_count
    if spectator_rank >= (1 << len(_spectator_squares())):
        raise ObservationGenerationError("Chess sample index exceeds generator capacity")
    mechanism = mechanisms[base_index // len(transforms)]
    transform = transforms[base_index % len(transforms)]
    base_pieces = _base_mate_pieces(
        mechanism=mechanism,
        spectator_rank=spectator_rank,
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
        spectator_count=_spectator_mask_for_rank(spectator_rank).bit_count(),
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


def _with_chess_artifacts(sample: GeneratedSample) -> GeneratedSample:
    artifacts: dict[str, object] = {}
    image_data_url = _sample_board_image_data_url(sample)
    if image_data_url is not None:
        artifacts["image_data_url"] = image_data_url
        artifacts["image_overlay"] = _sample_board_image_overlay(sample)
    if not artifacts:
        return sample
    return replace(sample, artifacts=artifacts)


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
    seed: int,
    sample_indices: tuple[int, ...],
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
        seed=seed,
        sample_indices=sample_indices,
        cardinality=cardinality,
        outcome_ids=outcome_ids,
        sample_shape=sample_shape,
    )


def _tensor_batch_from_global_indices(
    *,
    runtime: TensorRuntime,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    outcome_ids: tuple[str, ...],
    sample_shape: tuple[int, ...],
) -> tuple[Any, Any]:
    sample_count = _sample_count(sample_shape)
    _require_sample_cardinality(cardinality)
    global_offset = _global_sample_index(cardinality=cardinality, local_index=0)
    field_shape = (sample_count, *_tensor_shape)
    target_shape = (sample_count, len(outcome_ids))
    fields = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(
            shape=field_shape,
            dtype="float32",
            program=_chess_field_tensor_program(
                seed=seed,
                sample_indices=sample_indices,
                cardinality=cardinality,
                global_offset=global_offset,
            ),
        ),
    )
    targets = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(
            shape=target_shape,
            dtype="float32",
            program=_chess_target_tensor_program(
                seed=seed,
                sample_indices=sample_indices,
                cardinality=cardinality,
                global_offset=global_offset,
                base_target_indices=_base_target_indices(outcome_ids),
                outcome_count=len(outcome_ids),
            ),
        ),
    )
    return (
        fields.reshape((*sample_shape, *_tensor_shape)),
        targets.reshape((*sample_shape, len(outcome_ids))),
    )


def _chess_field_tensor_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    global_offset: int,
) -> TensorElementProgram:
    mechanisms = _mate_mechanisms()
    spectator_squares = _spectator_squares()
    base_count = len(mechanisms) * len(_board_transforms())
    mechanism_queen_squares = tuple(mechanism.queen_square for mechanism in mechanisms)
    support_squares = tuple(
        mechanism.support_pieces[0][0] if mechanism.support_pieces else -1
        for mechanism in mechanisms
    )
    support_planes = tuple(
        _piece_plane(mechanism.support_pieces[0][1]) if mechanism.support_pieces else -1
        for mechanism in mechanisms
    )

    def transformed_square(square: Any, transform_index: Any) -> Any:
        file_index = square.remainder(8)
        rank_index = square.div(8, rounding_mode="floor")
        transformed_file = file_index
        transformed_rank = rank_index
        transformed_file = transformed_file.where(transform_index != 1, 7 - file_index)
        transformed_rank = transformed_rank.where(transform_index != 2, 7 - rank_index)
        transformed_file = transformed_file.where(transform_index != 3, 7 - file_index)
        transformed_rank = transformed_rank.where(transform_index != 3, 7 - rank_index)
        transformed_file = transformed_file.where(transform_index != 4, rank_index)
        transformed_rank = transformed_rank.where(transform_index != 4, file_index)
        transformed_file = transformed_file.where(transform_index != 5, 7 - rank_index)
        transformed_rank = transformed_rank.where(transform_index != 5, 7 - file_index)
        transformed_file = transformed_file.where(transform_index != 6, rank_index)
        transformed_rank = transformed_rank.where(transform_index != 6, 7 - file_index)
        transformed_file = transformed_file.where(transform_index != 7, 7 - rank_index)
        transformed_rank = transformed_rank.where(transform_index != 7, file_index)
        return transformed_rank * 8 + transformed_file

    def element_function(
        coordinates: tuple[Any, ...],
        flat_indices: Any,
        *,
        seed_value: Any,
        sample_indices_value: Any,
        cardinality_value: Any,
        global_offset_value: Any,
        mechanism_queen_square_values: Any,
        support_square_values: Any,
        support_plane_values: Any,
        spectator_square_values: Any,
    ) -> Any:
        _ = flat_indices
        sample_axis_index, plane_index, rank_index, file_index = coordinates
        sample_index = sample_indices_value[sample_axis_index]
        square = rank_index * 8 + file_index
        local_index = (sample_index + seed_value).remainder(cardinality_value)
        global_index = global_offset_value + local_index
        base_index = global_index.remainder(base_count)
        mechanism_index = base_index.div(8, rounding_mode="floor")
        transform_index = base_index.remainder(8)
        spectator_mask = global_index.div(base_count, rounding_mode="floor")
        occupied = (
            (plane_index == 12)
            | ((plane_index == _piece_plane(chess.Piece(chess.KING, chess.BLACK)))
             & (square == transformed_square(square * 0 + chess.A1, transform_index)))
            | ((plane_index == _piece_plane(chess.Piece(chess.ROOK, chess.BLACK)))
               & (square == transformed_square(square * 0 + chess.B1, transform_index)))
            | ((plane_index == _piece_plane(chess.Piece(chess.KING, chess.WHITE)))
               & (square == transformed_square(square * 0 + chess.C1, transform_index)))
            | ((plane_index == _piece_plane(chess.Piece(chess.QUEEN, chess.WHITE)))
               & (
                   square
                   == transformed_square(
                       mechanism_queen_square_values[mechanism_index],
                       transform_index,
                   )
               ))
        )
        support_square = support_square_values[mechanism_index]
        support_plane = support_plane_values[mechanism_index]
        occupied = occupied | (
            (support_square >= 0)
            & (plane_index == support_plane)
            & (square == transformed_square(support_square, transform_index))
        )
        for spectator_index in range(len(spectator_squares)):
            spectator_active = (
                spectator_mask.div(1 << spectator_index, rounding_mode="floor").remainder(2)
                == 1
            )
            occupied = occupied | (
                spectator_active
                & (plane_index == _piece_plane(chess.Piece(chess.KNIGHT, chess.WHITE)))
                & (
                    square
                    == transformed_square(
                        spectator_square_values[spectator_index],
                        transform_index,
                    )
                )
            )
        return occupied * 1.0

    return TensorElementProgram(
        kernel=element_function,
        parameters={
            "seed_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(seed,),
            ),
            "sample_indices_value": TensorElementParameter(
                dtype="int64",
                shape=(len(sample_indices),),
                values=sample_indices,
                dynamic_axes=(0,),
            ),
            "cardinality_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(cardinality,),
            ),
            "global_offset_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(global_offset,),
            ),
            "mechanism_queen_square_values": TensorElementParameter(
                dtype="int64",
                shape=(len(mechanisms),),
                values=mechanism_queen_squares,
            ),
            "support_square_values": TensorElementParameter(
                dtype="int64",
                shape=(len(mechanisms),),
                values=support_squares,
            ),
            "support_plane_values": TensorElementParameter(
                dtype="int64",
                shape=(len(mechanisms),),
                values=support_planes,
            ),
            "spectator_square_values": TensorElementParameter(
                dtype="int64",
                shape=(len(spectator_squares),),
                values=spectator_squares,
            ),
        },
        cache_key=("chess-field", len(mechanisms), len(spectator_squares)),
    )


def _chess_target_tensor_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    global_offset: int,
    base_target_indices: tuple[int, ...],
    outcome_count: int,
) -> TensorElementProgram:
    base_count = len(base_target_indices)

    def element_function(
        coordinates: tuple[Any, ...],
        flat_indices: Any,
        *,
        seed_value: Any,
        sample_indices_value: Any,
        cardinality_value: Any,
        global_offset_value: Any,
        base_target_index_values: Any,
    ) -> Any:
        _ = coordinates
        sample_axis_index = flat_indices.div(outcome_count, rounding_mode="floor")
        sample_index = sample_indices_value[sample_axis_index]
        outcome_index = flat_indices.remainder(outcome_count)
        local_index = (sample_index + seed_value).remainder(cardinality_value)
        global_index = global_offset_value + local_index
        target_index = base_target_index_values[global_index.remainder(base_count)]
        return (target_index == outcome_index) * 1.0

    return TensorElementProgram(
        kernel=element_function,
        parameters={
            "seed_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(seed,),
            ),
            "sample_indices_value": TensorElementParameter(
                dtype="int64",
                shape=(len(sample_indices),),
                values=sample_indices,
                dynamic_axes=(0,),
            ),
            "cardinality_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(cardinality,),
            ),
            "global_offset_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(global_offset,),
            ),
            "base_target_index_values": TensorElementParameter(
                dtype="int64",
                shape=(len(base_target_indices),),
                values=base_target_indices,
            )
        },
        cache_key=("chess-target", outcome_count),
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


def _sample_indices(
    *,
    sample_count: int,
    sample_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if sample_indices is None:
        return tuple(range(sample_count))
    normalized = tuple(sample_indices)
    if len(normalized) != sample_count:
        raise ObservationGenerationError("sample_indices length must match sample shape")
    if any(type(index) is not int or index < 0 for index in normalized):
        raise ObservationGenerationError("sample_indices must be nonnegative integers")
    return normalized


all_meaningful_uci_moves = _all_meaningful_uci_moves()

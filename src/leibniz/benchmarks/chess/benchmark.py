"""Chess benchmark implementation entry point."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
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
from leibniz.tensor_runtime import TensorRuntime, make_float_tensor
from leibniz.timing import TimingCollector

__all__ = ["all_meaningful_uci_moves", "benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.chess.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.chess.uci-moves@0.1.0")
_console_preview_limit = 12
_complexity_rung_size = 0.1
_tensor_shape = (18, 8, 8)

_mate_in_one_fens = (
    "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1",
)


def benchmark(root: Path) -> BenchmarkProtocol:
    """Return the Chess benchmark implementation."""

    return Benchmark(root=root)


class Benchmark:
    """Executable Chess benchmark declaration."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._manifest = _manifest()
        self._generator = Generator(
            manifest=self._manifest,
            positions=_mate_in_one_positions(),
        )

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

    @property
    def observation_id(self) -> str:
        return f"fen:{self.fen}"

    @property
    def legal_move_count(self) -> int:
        return len(self.legal_moves)

    @property
    def complexity(self) -> float:
        return math.log2(self.legal_move_count)

    @property
    def complexity_value(self) -> ComplexityValue:
        return ComplexityValue(value=self.complexity)

    @property
    def target_distribution(self) -> Mapping[str, float]:
        probability = 1.0 / len(self.mate_moves)
        return dict.fromkeys(self.mate_moves, probability)


@dataclass(frozen=True, slots=True)
class Generator:
    """Generate Chess mate-in-one positions by legal-move-count complexity."""

    manifest: BenchmarkManifest
    positions: tuple[_MateInOnePosition, ...]

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
        positions = self._positions_for_request(complexity_request)
        if not positions:
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
        selected_positions = tuple(rng.choice(positions) for _index in range(sample_count))
        samples = (
            tuple(
                self._sample(
                    index=index,
                    position=position,
                )
                for index, position in enumerate(selected_positions)
            )
            if include_metadata
            else ()
        )
        fields, targets = _tensor_batch(
            runtime=runtime,
            positions=selected_positions,
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
        """Return the smallest known legal-move-count complexity."""

        return min(self.positions, key=lambda position: position.complexity).complexity_value

    def complexity_rung_size(self) -> float:
        """Return the log2 complexity width for curriculum rungs."""

        return _complexity_rung_size

    def complexity_candidate_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> ComplexityCandidate | None:
        """Return the first legal-move-count class inside a complexity band."""

        candidates = self.complexity_candidates_for_request(request=request)
        if not candidates:
            return None
        return candidates[0]

    def complexity_candidates_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> tuple[ComplexityCandidate, ...]:
        """Return exact legal-move-count candidates inside a complexity band."""

        candidates: list[ComplexityCandidate] = []
        seen_counts: set[int] = set()
        for position in sorted(
            self.positions,
            key=lambda item: (item.legal_move_count, item.fen),
        ):
            if position.legal_move_count in seen_counts:
                continue
            if not request.contains(position.complexity_value):
                continue
            seen_counts.add(position.legal_move_count)
            candidates.append(
                ComplexityCandidate(
                    request=ComplexityRequest(
                        minimum=position.complexity,
                        maximum=position.complexity,
                    ),
                    cardinality=position.legal_move_count,
                    metadata={
                        "kind": "chess-legal-move-cardinality",
                        "legal_move_count": position.legal_move_count,
                    },
                )
            )
        return tuple(candidates)

    def console_preview_batches(
        self,
        *,
        atom_count: int,
    ) -> tuple[Mapping[str, object], ...]:
        """Return browser-preview batches for known legal-move-count classes."""

        if atom_count != len(self.manifest.outcome_space.outcomes):
            raise ObservationGenerationError("atom_count does not match outcome space")
        batches: list[Mapping[str, object]] = []
        for position in sorted(
            self.positions,
            key=lambda item: (item.legal_move_count, item.fen),
        ):
            request = ComplexityRequest(
                minimum=position.complexity,
                maximum=position.complexity,
            )
            sample_set = self(
                seed=401 + len(batches),
                shape=1,
                complexity_request=request,
            )
            samples = [
                _sample_preview_record(sample)
                for sample in sample_set.samples[:_console_preview_limit]
            ]
            batches.append(
                {
                    "mode": "complexity-window",
                    "label": f"{position.legal_move_count} legal moves",
                    "seed": sample_set.seed,
                    "sample_count": len(samples),
                    "complexity_window": request.to_record(),
                    "complexity_cardinalities": [position.legal_move_count],
                    "presentation": {
                        "sample_card_density": "standard",
                        "aggregate_mode": False,
                    },
                    "samples": samples,
                }
            )
        return tuple(batches)

    def _positions_for_request(
        self,
        request: ComplexityRequest | None,
    ) -> tuple[_MateInOnePosition, ...]:
        if request is None:
            return self.positions
        return tuple(
            position
            for position in self.positions
            if request.contains(position.complexity_value)
        )

    def _sample(self, *, index: int, position: _MateInOnePosition) -> GeneratedSample:
        return GeneratedSample(
            index=index,
            outcome_id=position.mate_moves[0],
            complexity=position.complexity,
            complexity_value=position.complexity_value,
            available_outcome_ids=position.legal_moves,
            observable_state_id=position.observation_id,
            target_distribution=position.target_distribution,
            latent_coordinates=_latent_coordinates(position),
        )


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id=_benchmark_id,
        name=_benchmark_id.name,
        outcome_space=OutcomeSpace(
            id=_outcome_space_id,
            outcomes=tuple(Outcome(id=move) for move in all_meaningful_uci_moves),
        ),
        observation_ids=frozenset(position.observation_id for position in _mate_in_one_positions()),
    )


def _mate_in_one_positions() -> tuple[_MateInOnePosition, ...]:
    positions = tuple(_mate_in_one_position(fen) for fen in _mate_in_one_fens)
    if not positions:
        raise ObservationGenerationError("Chess benchmark must declare at least one position")
    return positions


def _mate_in_one_position(fen: str) -> _MateInOnePosition:
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
    )


def _is_checkmate_after_move(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()


def _latent_coordinates(position: _MateInOnePosition) -> tuple[Mapping[str, object], ...]:
    return (
        {
            "name": "benchmarks.chess.position.fen",
            "role": "content",
            "degree_measure": {"kind": "fixture-position", "count": 1.0},
            "multiplicity": 1,
            "values": position.fen,
        },
        {
            "name": "benchmarks.chess.position.legal-move-count",
            "role": "complexity",
            "degree_measure": {
                "kind": "discrete-log2-count",
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
    )


def _sample_preview_record(sample: GeneratedSample) -> Mapping[str, object]:
    return {
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
        "target_distribution": [
            {"outcome_id": outcome_id, "probability": probability}
            for outcome_id, probability in sample.target_distribution_or_one_hot().items()
        ],
        "latent_coordinates": [
            dict(coordinate) for coordinate in sample.latent_coordinates
        ],
    }


def _tensor_batch(
    *,
    runtime: TensorRuntime | None,
    positions: Sequence[_MateInOnePosition],
    outcome_ids: tuple[str, ...] | None,
    sample_shape: tuple[int, ...],
) -> tuple[Any | None, Any | None]:
    if runtime is None:
        return None, None
    if outcome_ids is None:
        raise ObservationGenerationError("tensor generation requires outcome_ids")
    unknown_outcomes = tuple(
        outcome_id
        for position in positions
        for outcome_id in (*position.legal_moves, *position.mate_moves)
        if outcome_id not in outcome_ids
    )
    if unknown_outcomes:
        raise ObservationGenerationError(
            "tensor generation outcome_ids do not cover generated Chess moves"
        )
    fields = make_float_tensor(
        runtime,
        [_board_tensor(position.fen) for position in positions],
        device=runtime.device,
    )
    targets = make_float_tensor(
        runtime,
        [
            _target_row(position.target_distribution, outcome_ids=outcome_ids)
            for position in positions
        ],
        device=runtime.device,
    )
    return (
        fields.reshape((*sample_shape, *_tensor_shape)),
        targets.reshape((*sample_shape, len(outcome_ids))),
    )


def _board_tensor(fen: str) -> list[list[list[float]]]:
    board = chess.Board(fen)
    planes = [
        [[0.0 for _file in range(8)] for _rank in range(8)]
        for _plane in range(_tensor_shape[0])
    ]
    for square, piece in board.piece_map().items():
        plane = _piece_plane(piece)
        planes[plane][chess.square_rank(square)][chess.square_file(square)] = 1.0
    if board.turn == chess.WHITE:
        _fill_plane(planes[12], 1.0)
    if board.has_kingside_castling_rights(chess.WHITE):
        _fill_plane(planes[13], 1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        _fill_plane(planes[14], 1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        _fill_plane(planes[15], 1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        _fill_plane(planes[16], 1.0)
    if board.ep_square is not None:
        planes[17][chess.square_rank(board.ep_square)][
            chess.square_file(board.ep_square)
        ] = 1.0
    return planes


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


def _fill_plane(plane: list[list[float]], value: float) -> None:
    for rank in plane:
        for file_index in range(len(rank)):
            rank[file_index] = value


def _target_row(
    distribution: Mapping[str, float],
    *,
    outcome_ids: tuple[str, ...],
) -> list[float]:
    return [float(distribution.get(outcome_id, 0.0)) for outcome_id in outcome_ids]


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

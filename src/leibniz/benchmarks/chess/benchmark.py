"""Chess benchmark implementation entry point."""

from __future__ import annotations

import math
import random
from base64 import b64encode
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
_tensor_shape = (18, 8, 8)
_board_preview_size = 512
_board_preview_square_size = _board_preview_size // 8

_mate_in_one_fens = (
    "8/8/8/8/8/8/2Q5/krK5 w - - 0 1",
    "8/8/8/8/8/1Q6/8/krK5 w - - 0 1",
    "8/q5R1/8/k1K5/2n5/8/6Q1/6B1 w - - 0 1",
    "Q3K3/p7/k3r1R1/7Q/8/8/8/8 w - - 0 1",
    "1Q6/P7/8/8/1n6/2B3bk/5K2/8 w - - 0 1",
    "8/7k/8/1Q4Qr/5N2/7K/8/3Q4 w - - 0 1",
    "2N1K3/B7/8/k7/b1QQ4/8/2N5/8 w - - 0 1",
    "8/8/8/8/6QP/4K3/2Q5/B2Qrk2 w - - 0 1",
    "1r6/8/8/2B1K3/1Q6/1Q2r2k/4Q3/8 w - - 0 1",
    "7R/3R1rk1/8/7R/2Q5/3B1K2/8/8 w - - 0 1",
    "3Q4/K7/4P3/6Q1/2Q5/8/8/4k1bR w - - 0 1",
    "8/8/8/8/8/8/k2r4/2KQ4 w - - 0 1",
    "Q2n4/7q/5r2/n7/8/6K1/8/6kN w - - 0 1",
    "8/8/8/8/8/8/k3r3/2KQ4 w - - 0 1",
    "8/8/8/8/1r6/k7/8/KQ6 w - - 0 1",
    "8/8/3r4/8/8/8/8/k1KQ4 w - - 0 1",
    "8/8/8/8/r7/k7/8/1QK5 w - - 0 1",
    "8/8/8/8/8/8/8/K1krQ3 w - - 0 1",
    "8/8/8/8/r7/k7/8/KQ6 w - - 0 1",
    "8/8/8/8/r7/k7/8/K3Q3 w - - 0 1",
    "8/8/8/8/1r6/k7/3Q4/K7 w - - 0 1",
    "6rk/8/8/8/8/8/8/K1Q5 w - - 0 1",
    "8/8/8/8/r7/k7/2Q5/K7 w - - 0 1",
    "8/8/8/5Q2/8/8/3r4/K1k5 w - - 0 1",
    "6rk/8/8/8/8/8/3Q4/K7 w - - 0 1",
    "8/8/8/8/3Q4/8/8/K1kr4 w - - 0 1",
    "6rk/8/8/8/8/4Q3/8/K7 w - - 0 1",
    "8/8/8/8/4Q3/8/8/1K1kr3 w - - 0 1",
    "6rk/8/8/8/8/4Q3/8/1K6 w - - 0 1",
    "8/8/8/8/r2Q4/k7/8/2K5 w - - 0 1",
    "8/8/8/8/3Q4/8/k1K5/8 w - - 0 1",
    "8/8/8/3Q4/8/8/2K5/k7 w - - 0 1",
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
        sample_space_cardinality = len(positions)
        complexity = math.log2(sample_space_cardinality)
        selected_positions = tuple(rng.choice(positions) for _index in range(sample_count))
        samples = (
            tuple(
                self._sample(
                    index=index,
                    position=position,
                    sample_space_cardinality=sample_space_cardinality,
                    complexity=complexity,
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
        """Return the smallest supported Chess sample-space complexity."""

        return ComplexityValue(value=0.0)

    def complexity_candidate_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> ComplexityCandidate | None:
        """Return the first sample-space-cardinality class inside a complexity band."""

        candidates = self.complexity_candidates_for_request(request=request)
        if not candidates:
            return None
        return candidates[0]

    def complexity_candidates_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> tuple[ComplexityCandidate, ...]:
        """Return exact sample-space-cardinality candidates inside a complexity band."""

        return tuple(
            candidate
            for candidate in self._sample_space_cardinality_candidates()
            if request.contains(
                ComplexityValue(
                    measure_id=candidate.request.measure_id,
                    value=candidate.complexity,
                )
            )
        )

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

        candidates = self._sample_space_cardinality_candidates()
        return candidates[start_index : start_index + count]

    def _sample_space_cardinality_candidates(self) -> tuple[ComplexityCandidate, ...]:
        candidates: list[ComplexityCandidate] = []
        sorted_positions = _sorted_positions(self.positions)
        for cardinality in range(1, len(sorted_positions) + 1):
            positions = sorted_positions[:cardinality]
            complexity = math.log2(cardinality)
            legal_move_counts = [position.legal_move_count for position in positions]
            candidates.append(
                ComplexityCandidate(
                    request=ComplexityRequest(
                        minimum=complexity,
                        maximum=complexity,
                    ),
                    cardinality=cardinality,
                    metadata={
                        "kind": "chess-sample-space-cardinality",
                        "sample_cardinality": cardinality,
                        "target_policy": "mate-in-one",
                        "output_move_count": len(all_meaningful_uci_moves),
                        "legal_move_counts": legal_move_counts,
                        "oracle_inference_compute": {
                            "kind": "oracle-inference-compute-reference-v1",
                            "unit": "abstract-ops",
                            "aggregation": "max",
                            "value": max(legal_move_counts),
                            "components": {
                                "legal_move_count_max": max(legal_move_counts),
                                "legal_move_count_mean": (
                                    sum(legal_move_counts) / len(legal_move_counts)
                                ),
                                "sample_cardinality": cardinality,
                            },
                        },
                        "representatives": [
                            dict(position.representative_metadata())
                            for position in positions
                        ],
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
        for candidate in self._sample_space_cardinality_candidates():
            if candidate.cardinality is None:
                continue
            positions = _positions_for_sample_cardinality(
                self.positions,
                cardinality=candidate.cardinality,
            )
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
                    "legal_move_counts": [
                        position.legal_move_count for position in positions
                    ],
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
            return _sorted_positions(self.positions)
        candidates = self.complexity_candidates_for_request(request=request)
        if not candidates:
            return ()
        cardinality = candidates[0].cardinality
        if cardinality is None:
            return ()
        return _positions_for_sample_cardinality(
            self.positions,
            cardinality=cardinality,
        )

    def _sample(
        self,
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
            target_distribution=position.target_distribution,
            latent_coordinates=_latent_coordinates(
                position,
                sample_space_cardinality=sample_space_cardinality,
            ),
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
    _validate_representative_ladder(positions)
    return positions


def _sorted_positions(
    positions: Sequence[_MateInOnePosition],
) -> tuple[_MateInOnePosition, ...]:
    return tuple(sorted(positions, key=lambda item: (item.legal_move_count, item.fen)))


def _positions_for_sample_cardinality(
    positions: Sequence[_MateInOnePosition],
    *,
    cardinality: int,
) -> tuple[_MateInOnePosition, ...]:
    if type(cardinality) is not int or cardinality < 1:
        raise ObservationGenerationError("Chess sample cardinality must be positive")
    sorted_positions = _sorted_positions(positions)
    if cardinality > len(sorted_positions):
        raise ObservationGenerationError(
            "Chess sample cardinality exceeds representative corpus size"
        )
    return sorted_positions[:cardinality]


def _validate_representative_ladder(positions: Sequence[_MateInOnePosition]) -> None:
    seen_counts: set[int] = set()
    duplicate_counts: set[int] = set()
    for position in positions:
        if position.legal_move_count in seen_counts:
            duplicate_counts.add(position.legal_move_count)
        seen_counts.add(position.legal_move_count)
    if duplicate_counts:
        duplicate = min(duplicate_counts)
        raise ObservationGenerationError(
            f"Chess benchmark has multiple representatives for {duplicate} legal moves"
        )
    legal_piece_symbols = {
        symbol
        for position in positions
        for symbol in position.legal_move_piece_symbols
    }
    required_legal_piece_symbols = frozenset({"B", "K", "N", "Q", "R"})
    missing = required_legal_piece_symbols - legal_piece_symbols
    if missing:
        raise ObservationGenerationError(
            "Chess representative ladder is missing legal moves by: "
            + ", ".join(sorted(missing))
        )
    mate_piece_symbols = {
        symbol
        for position in positions
        for symbol in position.mate_move_piece_symbols
    }
    if mate_piece_symbols == {"Q"}:
        raise ObservationGenerationError(
            "Chess representative ladder must include a non-queen mate motif"
        )


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
        "target_distribution": [
            {"outcome_id": outcome_id, "probability": probability}
            for outcome_id, probability in sample.target_distribution_or_one_hot().items()
        ],
        "latent_coordinates": [
            dict(coordinate) for coordinate in sample.latent_coordinates
        ],
    }
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

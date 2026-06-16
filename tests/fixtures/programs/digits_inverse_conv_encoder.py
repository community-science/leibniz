from __future__ import annotations

from typing import Any

from leibniz.program_graphs import (
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramTensorContract,
)


def build_program_graph(runtime: Any) -> ProgramGraph:
    torch = runtime.torch

    class DigitsInverseConvEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 24, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(24, 24, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(24, 48, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(48, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 4 * 4, 128),
                torch.nn.ReLU(),
            )
            self.readout = torch.nn.Linear(128, 85)
            torch.nn.init.zeros_(self.readout.weight)
            torch.nn.init.zeros_(self.readout.bias)
            with torch.no_grad():
                self.readout.bias[12] = 1.0
                self.readout.bias[14] = 1.0

        def forward(self, image: Any) -> Any:
            return self.readout(self.features(image))

    encoder = DigitsInverseConvEncoder()
    return ProgramGraph(
        contract_kind="inverse",
        inputs=(ProgramTensorContract("image", (1, "N", "N")),),
        outputs=(ProgramTensorContract("latent", (85,)),),
        nodes=(
            ProgramGraphNode(
                id="encoder",
                kind="submitted-digits-inverse-conv-encoder",
                operation=encoder,
            ),
        ),
        edges=(
            ProgramGraphEdge("image", "encoder"),
            ProgramGraphEdge("encoder", "latent"),
        ),
    )

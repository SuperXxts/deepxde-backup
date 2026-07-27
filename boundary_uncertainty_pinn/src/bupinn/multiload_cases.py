"""C-group multi-load FEM gate case definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiLoadCaseDefinition:
    case_id: str
    short_name: str
    num_loads: int
    top: str
    learnable_A: bool = False
    reaction: bool = False
    anchor: bool = False
    boundary_layer: bool = False
    decoupled_obs: bool = False
    purpose: str = ""

    def settings(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "short_name": self.short_name,
            "num_loads": self.num_loads,
            "top": self.top,
            "learnable_A": self.learnable_A,
            "reaction": self.reaction,
            "anchor": self.anchor,
            "boundary_layer": self.boundary_layer,
            "decoupled_obs": self.decoupled_obs,
            "purpose": self.purpose,
        }


MULTILOAD_CASE_DEFINITIONS: tuple[MultiLoadCaseDefinition, ...] = (
    MultiLoadCaseDefinition(
        "C0",
        "single-load-true-boundary",
        num_loads=1,
        top="true",
        purpose=(
            "Single vertical plate-settlement load with the correct boundary. "
            "This is the gate test for whether a single load can identify a spatially varying K/mu field."
        ),
    ),
    MultiLoadCaseDefinition(
        "C1",
        "three-load-true-boundary",
        num_loads=3,
        top="true",
        purpose=(
            "Three independent loading modes with correct boundaries and a shared K/mu field. "
            "This tests whether multi-load excitation restores identifiability before any boundary uncertainty is introduced."
        ),
    ),
    MultiLoadCaseDefinition(
        "C8",
        "three-load-true-boundary-reaction",
        num_loads=3,
        top="true",
        reaction=True,
        purpose=(
            "Three independent loading modes with correct displacement boundaries and resultant reaction constraints. "
            "This diagnosis isolates whether the high material error in the true-boundary baseline is caused by "
            "the missing global force/stiffness-scale information rather than by boundary uncertainty."
        ),
    ),
    MultiLoadCaseDefinition(
        "C2",
        "three-load-wrong-fixed-boundary",
        num_loads=3,
        top="wrong",
        purpose=(
            "Three loading modes with the loading-plate displacement fixed to 85% of its true value. "
            "This tests whether boundary error contaminates the shared K/mu inversion after the load information is sufficient."
        ),
    ),
    MultiLoadCaseDefinition(
        "C3",
        "three-load-learnable-boundary",
        num_loads=3,
        top="learnable_A",
        learnable_A=True,
        purpose=(
            "Three loading modes with one trainable displacement-scale parameter per load. "
            "This checks whether low-dimensional boundary calibration alone is enough."
        ),
    ),
    MultiLoadCaseDefinition(
        "C4",
        "three-load-learnable-boundary-reaction",
        num_loads=3,
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        purpose=(
            "Three loading modes with trainable boundary scales and resultant reaction constraints. "
            "This checks whether global force information restores the stiffness scale."
        ),
    ),
    MultiLoadCaseDefinition(
        "C5",
        "three-load-learnable-boundary-reaction-anchor",
        num_loads=3,
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        anchor=True,
        purpose=(
            "Three loading modes with trainable boundary scales, resultant reaction constraints, "
            "and sparse displacement anchors on the loading boundary. "
            "This isolates the incremental value of anchors before adding a boundary branch."
        ),
    ),
    MultiLoadCaseDefinition(
        "C6",
        "three-load-boundary-branch-reaction-anchor",
        num_loads=3,
        top="network_top",
        boundary_layer=True,
        reaction=True,
        anchor=True,
        purpose=(
            "Three loading modes with a low-dimensional boundary branch, resultant reactions, "
            "and sparse boundary anchors, but without decoupled observation residuals. "
            "This isolates the boundary branch effect."
        ),
    ),
    MultiLoadCaseDefinition(
        "C7",
        "three-load-full-decoupled-method",
        num_loads=3,
        top="network_top",
        boundary_layer=True,
        decoupled_obs=True,
        reaction=True,
        anchor=True,
        purpose=(
            "Full multi-load method: shared K/mu material branch, load-specific state branches, "
            "low-dimensional boundary branches, decoupled observation residuals, boundary anchors, and resultant reactions."
        ),
    ),
)

MULTILOAD_CASE_CHOICES = tuple(case.case_id for case in MULTILOAD_CASE_DEFINITIONS)
MULTILOAD_CASE_TABLE = {case.case_id: case for case in MULTILOAD_CASE_DEFINITIONS}


def multiload_case_settings(case_id: str) -> dict[str, object]:
    return MULTILOAD_CASE_TABLE[case_id].settings()


def multiload_case_description(case_id: str) -> str:
    case = MULTILOAD_CASE_TABLE[case_id]
    return f"{case.case_id} {case.short_name}\n\n{case.purpose}\n"


def multiload_case_markdown_table() -> str:
    lines = [
        "| Case | Loads | Boundary model | Decoupled observations | Reactions | Anchors | Purpose |",
        "|---|---:|---|---|---|---|---|",
    ]
    for case in MULTILOAD_CASE_DEFINITIONS:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.case_id,
                    str(case.num_loads),
                    case.top,
                    "yes" if case.decoupled_obs else "-",
                    "yes" if case.reaction else "-",
                    "yes" if case.anchor else "-",
                    case.purpose,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"

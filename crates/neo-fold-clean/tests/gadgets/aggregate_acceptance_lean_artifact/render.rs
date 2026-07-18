//! Lean-data rendering for the active aggregate-acceptance leaf artifact.
//!
//! Owns: deterministic serialization of the arity, matrix bindings, nine
//! role-normalized rows, and sparse-polynomial specialization.
//!
//! Does not own: extraction, semantic proof, fixture geometry, physical
//! placement, or promotion of the generated `.expected` candidate.
//!
//! Emits constraints: no.
//!
//! | Data branch | Serialized object | Review purpose |
//! |---|---|---|
//! | Shape | schema version and arity | reject gate-schema drift |
//! | Bindings | 40 matrix roles | bind readable roles to production indices |
//! | Rows | nine active row families | expose every leaf equation |
//! | Polynomial | 25 sparse monomials | expose exact gate coefficients |

use std::fmt::Write as _;

use super::{
    ActiveRow, ArtifactAudit, CoordinateRole, MatrixBinding, MatrixLinearCombination, MatrixRole, PolynomialTerm,
    RoleTerm,
};

fn coordinate_role(role: CoordinateRole) -> String {
    match role {
        CoordinateRole::One => ".one".to_owned(),
        CoordinateRole::ChunkBit(index) => format!("(.chunkBit {index})"),
        CoordinateRole::Accept => ".accept".to_owned(),
        CoordinateRole::TreeOutput(index) => format!("(.treeOutput {index})"),
    }
}

fn matrix_role(role: MatrixRole) -> String {
    match role {
        MatrixRole::Selector => ".selector".to_owned(),
        MatrixRole::ProductLeft(index) => format!("(.productLeft {index})"),
        MatrixRole::ProductRight(index) => format!("(.productRight {index})"),
        MatrixRole::ProductOut => ".productOut".to_owned(),
        MatrixRole::QuadraticBitLeft => ".quadraticBitLeft".to_owned(),
        MatrixRole::QuadraticBitRight => ".quadraticBitRight".to_owned(),
    }
}

fn coordinate_terms(terms: &[RoleTerm<CoordinateRole>]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|term| format!("\u{27e8}{}, {}\u{27e9}", coordinate_role(term.role), term.coefficient))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_matrix_bindings(bindings: &[MatrixBinding]) -> String {
    let mut out = String::from("[\n");
    for (index, binding) in bindings.iter().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            out,
            "{separator}{{ role := {}, index := {} }}",
            matrix_role(binding.role),
            binding.index
        )
        .unwrap();
    }
    out.push(']');
    out
}

fn matrix_lc(lc: &MatrixLinearCombination) -> String {
    format!(
        "\u{27e8}{}, {}\u{27e9}",
        matrix_role(lc.role),
        coordinate_terms(&lc.terms)
    )
}

fn render_active_rows(rows: &[ActiveRow]) -> String {
    let mut out = String::from("[\n");
    for (index, row) in rows.iter().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        writeln!(
            out,
            "{separator}[{}]",
            row.iter().map(matrix_lc).collect::<Vec<_>>().join(", ")
        )
        .unwrap();
    }
    out.push(']');
    out
}

fn render_polynomial(terms: &[PolynomialTerm]) -> String {
    let mut out = String::from("[\n");
    for (index, term) in terms.iter().enumerate() {
        let separator = if index == 0 { "  " } else { ", " };
        let powers = term
            .powers
            .iter()
            .map(|power| format!("\u{27e8}{}, {}\u{27e9}", matrix_role(power.role), power.power))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(out, "{separator}\u{27e8}{}, [{powers}]\u{27e9}", term.coefficient).unwrap();
    }
    out.push(']');
    out
}

pub(super) fn render(artifact: &ArtifactAudit) -> String {
    let mut out = String::from(
        "import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceSchema\n\n\
/-! Generated exact aggregate-acceptance leaf data; do not hand-edit.\n\n\
Owns: the production gate arity, role-to-matrix bindings, nine normalized\n\
active rows, and exact sparse-polynomial specialization.\n\n\
Does not own: singleton fixture geometry, source-bit decoding, selectors,\n\
inactive rows, or the fixed-F' 960-chunk physical outer image.\n\n\
Emits constraints: no.\n\n\
Authority boundary: this is artifact evidence only. Handwritten correspondence\n\
must prove that these generated equations implement independent semantics.\n\n\
| Data branch | Exact production evidence | Semantic owner |\n\
|---|---|---|\n\
| `matrixBindings` | forty occupied matrix roles in arity 56 | aggregate artifact refinement |\n\
| `activeRows` | seven bit pairs, one radix-3 aggregate, one root binding | `AggregateAcceptanceRows` |\n\
| `polynomialTerms` | exact 25-term gate specialization | aggregate artifact refinement |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifactData\n\n\
open AggregateAcceptanceArtifact\n\n",
    );
    writeln!(out, "def schemaVersion : Nat := {}", artifact.schema_version).unwrap();
    writeln!(out, "def gateArity : Nat := {}", artifact.gate_arity).unwrap();
    writeln!(
        out,
        "def matrixBindings : List MatrixBinding :=\n{}",
        render_matrix_bindings(&artifact.matrix_bindings)
    )
    .unwrap();
    writeln!(
        out,
        "def activeRows : List ActiveRow :=\n{}",
        render_active_rows(&artifact.active_rows)
    )
    .unwrap();
    writeln!(
        out,
        "def polynomialTerms : List PolynomialTerm :=\n{}",
        render_polynomial(&artifact.polynomial_terms)
    )
    .unwrap();
    out.push_str(
        "\nend Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifactData\n",
    );
    out
}

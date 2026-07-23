//! Focused Lean rendering fragments for the factored raw-old-block row program.

use std::fmt::Write as _;

use super::{GeneratedLeanFile, GENERATED_ROOT, NAMESPACE_ROOT};

pub(super) fn generated_header(owner: &str) -> String {
    format!(
        "/-\nGenerated file: production combined-NC execution artifact; do not hand-edit.\n\nOwns: {owner}.\n\nDoes not own: row satisfaction, commitment binding, semantic acceptance,\nsecurity reductions, costs, or permission to remove rows.\n\nEmits constraints: no.\n\n| Stable stage path | Obligation | Authority class |\n|---|---|---|\n| `f_prime.pi_ccs_nc.delayed.combined.execution` | The generated execution payload named by `Owns` above | computed artifact |\n-/\n\n"
    )
}

/// Definitions inserted after the prefix tensor and per-block products.
pub(super) fn final_scale_program_fragment() -> &'static str {
    r#"
def laneTerms (lane limb : Nat) : List (Nat × Nat) :=
  (List.range blockCount).map fun block => (productColumn lane block limb, 1)
def finalScaleTrace (lane : Nat) : KMulTrace :=
  let left : KTerms := { c0 := laneTerms lane 0, c1 := laneTerms lane 1 }
  let right := oneMinusPointTerms factoredVariable
  let first := finalScaleFirstColumn + 5 * lane
  { left := left
    right := right
    sumLeft := left.c0 ++ left.c1
    sumRight := right.c0 ++ right.c1
    productC0 := first
    productC1 := first + 1
    productSum := first + 2
    output := kColumnsAt (first + 3) }
inductive RowOwner where
  | tensor (round parent definition : Nat)
  | product (lane block limb : Nat)
  | finalScale (lane definition : Nat)
  | terminal (lane limb : Nat)
deriving DecidableEq, Repr

"#
}

/// Row-owner and row-formula dispatch for the production factored profile.
pub(super) fn final_scale_row_dispatch_fragment() -> &'static str {
    r#"
def ownerAtNat (row : Nat) : RowOwner :=
  if row < tensorRows then
    let ordinal := row / 5
    let owner := tensorOwner ordinal
    .tensor owner.1 owner.2 (row % 5)
  else if row < finalScaleRowFirst then
    let offset := row - productRowFirst
    let product := offset / 2
    .product (product / blockCount) (product % blockCount) (offset % 2)
  else if row < terminalRowFirst then
    let offset := row - finalScaleRowFirst
    .finalScale (offset / 5) (offset % 5)
  else
    let offset := row - terminalRowFirst
    .terminal (offset / 2) (offset % 2)
def ownerAt (row : Fin totalRows) : RowOwner := ownerAtNat row.val
def tensorRow (round parent definition : Nat) : Row :=
  match (tensorTrace round parent).definitions[definition]? with
  | some current => current.builderRow
  | none => emptyRow
def productRow (lane block limb : Nat) : Row :=
  let chi := chiTerms block
  let selected := if limb = 0 then chi.c0 else chi.c1
  (Definition.mk (productColumn lane block limb)
    (.product (rawTerms lane block) selected)).builderRow
def finalScaleRow (lane definition : Nat) : Row :=
  match (finalScaleTrace lane).definitions[definition]? with
  | some current => current.builderRow
  | none => emptyRow
def terminalRow (lane limb : Nat) : Row :=
  builderLinearRow
    (if limb = 0 then (parentColumnsNat lane).c0 else (parentColumnsNat lane).c1)
    [(if limb = 0 then (finalScaleTrace lane).output.c0
      else (finalScaleTrace lane).output.c1, 1)]
def artifactRowForOwner : RowOwner -> Row
  | .tensor round parent definition => tensorRow round parent definition
  | .product lane block limb => productRow lane block limb
  | .finalScale lane definition => finalScaleRow lane definition
  | .terminal lane limb => terminalRow lane limb
def artifactRow (row : Fin totalRows) : Row := artifactRowForOwner (ownerAt row)

"#
}

pub(super) fn list_root(family: &str, chunks: usize, value_type: &str) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution.{family}");
    let mut contents = generated_header(&format!(
        "the exact ordered concatenation of {chunks} bounded {family} shards"
    ));
    for index in 0..chunks {
        writeln!(contents, "import {namespace}.Chunk{index}").expect("root import");
    }
    writeln!(contents, "\nnamespace {namespace}\n\ndef values : List {value_type} :=").expect("root def");
    for index in 0..chunks {
        let suffix = if index + 1 == chunks { "" } else { " ++" };
        writeln!(contents, "  Chunk{index}.values{suffix}").expect("root chunk");
    }
    writeln!(contents, "\nend {namespace}").expect("root end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/{family}.lean"),
        contents,
    }
}

pub(super) fn execution_root() -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution");
    let mut contents = generated_header("the stable facade over the complete runtime execution certificate");
    for module in [
        "Header",
        "RawOldBlockProjectionPlan",
        "RawOldBlockProjectionRowAt",
        "ProductionEmitterLayout",
        "PublicWrites",
        "RawOldBlockLanes",
        "Rounds",
        "GeneratedKBindings",
        "TerminalProjectionFixture",
    ] {
        writeln!(contents, "import {namespace}.{module}").expect("execution root import");
    }
    writeln!(contents, "\nnamespace {namespace}\n\ndef header : RawExecutionHeader := Header.value\ndef publicWrites : List RawPublicWrite := PublicWrites.values\ndef rawOldBlockLanes : List RawOldBlockLane := RawOldBlockLanes.values\ndef rounds : List RawCombinedNcRound := Rounds.values\ndef generatedKBindings : List RawGeneratedKBinding := GeneratedKBindings.values\n\nend {namespace}").expect("execution root");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution.lean"),
        contents,
    }
}

import NightstreamFPrime.Export.OptionB.StepPackage

/-! Parity data for Rust's expansion of the step packages (CW-7.1). A circuit's plan
(`CircuitRelation.plan`) is one plan row per row of `Package.relationRows`, followed by 150 S-box
rows per block (`PoseidonSboxFamilyPlan.plan`). Rust expands the relation rows itself: each
package piece, the ring rows and the binding rows. `summary` gives their count and a checksum that
mixes every entry with its row, port and column, so a Rust test that recomputes the checksum from
its own expansion compares the two expansions entry by entry. Rust builds the S-box rows from the
shared formula library (`SharedFormulas.poseidonComponent`), as today's production relation does;
Lean does not expand them here, because `SparseLayer` does not merge terms and the partial rounds
grow each form sixteenfold. The probe packages check Rust's identity code against
`StepPackage.identityFast`. -/

namespace NightstreamFPrime.Export.OptionB.Parity

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Generic

/-- A field element that depends on the plan row, the port and the column. -/
def mix (unit column : Nat) : F :=
  let point : F := Poseidon2.ofNat (unit * 16777259 + column + 1)
  point * point * point + Poseidon2.ofNat unit

def formSum {width : Nat} (unit : Nat) (form : SparseForm width) (total : F) : F :=
  form.entries.foldl (fun sum entry => sum + entry.coefficient * mix unit entry.column.val) total

/-- An R1CS row's plan ports: `a`, `b` and `c` as `formOf` gives them. -/
def ordinarySum (width : Nat) (rows : List R1CS.Row) (total : F) : F :=
  (rows.zipIdx).foldl (fun sum (row, index) =>
    let sum := formSum (4 * index) (CircuitRelation.formOf (width := width) row.a) sum
    let sum := formSum (4 * index + 1) (CircuitRelation.formOf (width := width) row.b) sum
    formSum (4 * index + 2) (CircuitRelation.formOf (width := width) row.c) sum) total

/-- A circuit whose blocks pack: its relation rows, its S-box rows, and the checksum of its
relation rows. -/
def summary (width : Nat) (pieces : List Pieces.Piece) (next : Nat) : Value :=
  let rows := CircuitRelation.relationRows CircuitRelation.publicColumns pieces next width
  .array [.atom rows.length, .atom (150 * (CircuitRelation.blocksOf pieces).length),
    .atom (ordinarySum width rows 0).val]

/-- Two small packages whose identity Lean computes (`StepPackage.identityFast`): a probe for
Rust's identity code. -/
def probeLight : StepPackage.Package :=
  ⟨0, 270, 16, 300, 400,
    [.rows [⟨⟨5, [(1, 7)]⟩, .one, .zero⟩], .block 2 [.var 3, .const 9]],
    [⟨301, [.add (.var 1) (.const 2)], [.inverseOrZero (.var 4)]⟩]⟩

def probeRefresh : StepPackage.Package :=
  ⟨1, 270, 32, 10, 20, [.ring 1 4 30 [.mul (.const 3) (.var 8)]], []⟩

/-- The parity file: its version, both circuits' summaries, and the probe packages with their
identity. -/
def parity (width : Nat) (light refresh : List Pieces.Piece × Nat) : Value :=
  .array [.atom 1, summary width light.1 light.2, summary width refresh.1 refresh.2,
    StepPackage.identityValue probeLight probeRefresh,
    .array ((StepPackage.identityFast probeLight probeRefresh).map fun word => .atom word.val)]

end NightstreamFPrime.Export.OptionB.Parity

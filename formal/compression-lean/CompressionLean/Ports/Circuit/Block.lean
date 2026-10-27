import CompressionLean.Ports.Shrink.Layout
import NightstreamFPrime.Gadgets.Poseidon2.Permutation
import CompressionLean.Lib.List

/-!
Owns the Poseidon2 permutation block of the shrink CCS
(`crates/neo-spartan/src/circuit/poseidon2.rs`): the template that `build`
and `entries` produce, written in Lean (`poseidonTemplate`).

- A block has 182 cells: the 16 inputs, the 150 S-box outputs in round
  order and the 16 outputs. Row `r < 150` is `(X·z)^7 = cell(16 + r)`, with
  `X` the S-box input as a linear form of earlier cells; row `150 + i` is
  `cell(166 + i) = output_i` (`poseidon2.rs`).
- `build` runs the permutation on linear forms in the steps of F′'s
  `Permutation.schedule`, with F′'s round constants. `combine` merges terms,
  drops zero coefficients and sorts by cell as Rust does, so `entries` are
  Rust's entries in Rust's order. `CompressionLeanTests/BlockDigest.lean`
  pins this against Rust.

`Lowering/Block.lean` proves that the template is well formed and sound.
-/

namespace CompressionLean.Circuit.Block

open CompressionLean.Lib
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Gadgets.Poseidon2
open CompressionLean.Shrink
open Fin.CommRing

/-- A linear form over the cells of one block, plus a constant
(`Linear`, `poseidon2.rs::Linear`). -/
structure Linear where
  terms : List (ℕ × F)
  constant : F

namespace Linear

/-- The form of one cell (`poseidon2.rs::Linear::cell`). -/
def cell (k : ℕ) : Linear := ⟨[(k, 1)], 0⟩

/-- The empty form, read past the end of a state. -/
def zero : Linear := ⟨[], 0⟩

/-- Add `t` to the first term on its cell, or append it (`poseidon2.rs::Linear::combine`). -/
def addTerm : List (ℕ × F) → ℕ × F → List (ℕ × F)
  | [], t => [t]
  | u :: rest, t => if u.1 = t.1 then (u.1, u.2 + t.2) :: rest else u :: addTerm rest t

/-- One part of a combination, added to the terms and the constant so far
(`poseidon2.rs::Linear::combine`). -/
def mergePart (acc : List (ℕ × F) × F) (part : Linear × F) : List (ℕ × F) × F :=
  (part.1.terms.foldl (fun terms t => addTerm terms (t.1, t.2 * part.2)) acc.1,
    acc.2 + part.1.constant * part.2)

/-- `Σ weight·part`, merged by cell, zero coefficients dropped, sorted by
cell (`Linear::combine`, `poseidon2.rs::Linear::combine`). -/
def combine (parts : List (Linear × F)) : Linear :=
  let merged := parts.foldl mergePart ([], 0)
  ⟨(merged.1.filter fun t => t.2 ≠ 0).mergeSort fun a b => a.1 ≤ b.1, merged.2⟩

/-- The value of a form on the cells `c` (`Linear::evaluate`,
`poseidon2.rs::Linear::evaluate`). -/
def eval (c : ℕ → F) (l : Linear) : F :=
  l.constant + (l.terms.map fun t => t.2 * c t.1).sum

end Linear

/-- `M₄` (`poseidon2.rs::external_linear`). -/
def m4 : List (List F) := [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]

/-- `M₄` on each chunk of four lanes (`poseidon2.rs::external_linear`). -/
def mixLinear (state : List Linear) : List Linear :=
  (List.range 16).map fun lane =>
    Linear.combine ((List.range 4).map fun j =>
      (state.getD (4 * (lane / 4) + j) .zero, (m4.getD (lane % 4) []).getD j 0))

/-- `M_E`: each mixed lane plus the mixed lanes of its residue class mod 4
(`external_linear`, `poseidon2.rs::external_linear`). -/
def externalLinear (state : List Linear) : List Linear :=
  let mixed := mixLinear state
  (List.range 16).map fun lane =>
    Linear.combine ((mixed.getD lane .zero, 1) ::
      (List.range 4).map fun chunk => (mixed.getD (4 * chunk + lane % 4) .zero, 1))

/-- The lane forms and the S-box inputs so far (`poseidon2.rs::build`). -/
structure Build where
  state : List Linear
  inputs : List Linear

/-- A full round: lane `i` plus its constant is the next S-box input, lane `i`
becomes that S-box's output cell, then `M_E` (`full_rounds`,
`poseidon2.rs::full_rounds`). -/
def fullRound (rows : List (List ℕ)) (round : ℕ) (b : Build) : Build :=
  ⟨externalLinear ((List.range 16).map fun lane => .cell (16 + b.inputs.length + lane)),
    b.inputs ++ (List.range 16).map fun lane =>
      ⟨(b.state.getD lane .zero).terms,
        (b.state.getD lane .zero).constant + Spec.Poseidon2.constantAt rows round lane⟩⟩

/-- The lanes of a partial round after its S-box: lane 0 becomes the next
S-box output cell (`poseidon2.rs::build`). -/
def partialLanes (b : Build) : List Linear :=
  (List.range 16).map fun lane =>
    if lane = 0 then .cell (16 + b.inputs.length) else b.state.getD lane .zero

/-- A partial round: lane 0 plus its constant is the next S-box input and
becomes its output cell, then lane `i` is `Σ lanes + v_i·lane_i`
(`poseidon2.rs::build`). -/
def partialRound (round : ℕ) (b : Build) : Build :=
  let sum := Linear.combine ((partialLanes b).map fun form => (form, 1))
  ⟨(List.range 16).map fun lane =>
      Linear.combine [(sum, 1), ((partialLanes b).getD lane .zero,
        Spec.Poseidon2.ofNat (Spec.Poseidon2.internalDiagonal.getD lane 0))],
    b.inputs ++ [⟨(b.state.getD 0 .zero).terms, (b.state.getD 0 .zero).constant +
      Spec.Poseidon2.ofNat (Spec.Poseidon2.internalConstants.getD round 0)⟩]⟩

/-- One step of F′'s permutation schedule on forms. -/
def step (b : Build) : Permutation.Step → Build
  | .initialLayer => ⟨externalLinear b.state, b.inputs⟩
  | .initialFullRound round => fullRound Spec.Poseidon2.initialConstants round b
  | .partialRound round => partialRound round b
  | .terminalFullRound round => fullRound Spec.Poseidon2.terminalConstants round b

/-- The input cells as forms. -/
def start : Build := ⟨(List.range 16).map .cell, []⟩

/-- `M_E`, the 4 initial full rounds, the 22 partial rounds and the 4
terminal full rounds on the input cells (`build`, `poseidon2.rs::build`). -/
def build : Build := Permutation.schedule.foldl step start

/-- The S-box rows (`SBOXES`, `poseidon2.rs::SBOXES`). -/
def sboxes : ℕ := 150

/-- The first output cell (`OUTPUT`, `poseidon2.rs::OUTPUT`). -/
def output : ℕ := 16 + sboxes

/-- A form's entries in one row, times `sign`; a zero constant has none
(`push`, `poseidon2.rs::entries`). -/
def push (matrix : Fin 8) (row : ℕ) (form : Linear) (sign : F) : List (TemplateEntry F) :=
  form.terms.map (fun t => ⟨matrix, row, some t.1, sign * t.2⟩) ++
    if form.constant ≠ 0 then [⟨matrix, row, none, sign * form.constant⟩] else []

/-- The output cell of row `row`: S-box output `16 + row`, or output cell
`166 + row - 150` (`poseidon2.rs::entries`). -/
def rowCell (row : ℕ) : ℕ := if row < sboxes then 16 + row else output + row - sboxes

/-- Every nonzero of the block (`entries`, `poseidon2.rs::entries`): `X` of
S-box row `r` is S-box input `r`; `C` of output row `150 + i` is minus output
form `i`; `C` of every row is its output cell. -/
def entries (sboxInputs outputs : List Linear) : List (TemplateEntry F) :=
  (sboxInputs.zipIdx.flatMap fun p => push 0 p.2 p.1 1) ++
    (outputs.zipIdx.flatMap fun p => push 7 (sboxes + p.2) p.1 (-1)) ++
    (List.range (sboxes + 16)).map fun row => ⟨7, row, some (rowCell row), 1⟩

/-- The Poseidon2 permutation block: `ROWS = 166` rows over `CELLS = 182`
cells (`poseidon2.rs::CELLS`, `poseidon2.rs::ROWS`). -/
def poseidonTemplate : Template F := ⟨sboxes + 16, output + 16, entries build.inputs build.state⟩

end CompressionLean.Circuit.Block

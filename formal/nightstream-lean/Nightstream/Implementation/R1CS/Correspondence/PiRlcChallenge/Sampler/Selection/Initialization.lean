import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.LinearEquality
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.SelectionRows

/-!
Semantic refinement of the checked zero prefix for the `Pi_RLC` bounded
first-accepted tail.

Owns: the single initialization equation that fixes the accepted count before
candidate zero to the integer zero.

Does not own: candidate classification, later cumulative counters, selection,
production placement, Rust conformance, row removal, or costs.

Emits constraints: no.

Authority boundary: this equation initializes a verifier-derived count chain;
it does not authorize any later count unless every candidate step is separately
refined.

| Protocol | Phase | Constraint family | Equation | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | sampler/selection init | zero prefix | `prefix_0 * 1 = 0` | the canonical integer prefix before candidate zero is exactly zero |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Initialization

open Nightstream.Implementation.R1CS

private theorem satisfies_zeroPrefixRow
    {assignment : Nat -> Nat}
    (satisfies : Satisfies SelectionRows.rows assignment) :
    RowHolds assignment SelectionRows.zeroPrefixRow := by
  apply satisfies
  simp [SelectionRows.rows]

theorem zeroPrefix_eq_zero
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment SelectionRows.zeroPrefixCol = 0 := by
  have holds := satisfies_zeroPrefixRow satisfies
  have valueCanonical := canonical SelectionRows.zeroPrefixCol
  simpa [SelectionRows.zeroPrefixRow, SelectionRows.zeroEqualityRow,
    RowHolds, lcEval, one, Nat.mod_eq_of_lt valueCanonical] using holds

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Initialization

import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ActivatedRawProgram
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector

/-!
Contract: refinement from the legacy residual activation wrapper to the
native CCS selector.

Assurance tier: model-level.

Owns:
- derivation of `S · (A · B - C) = 0` from each legacy lifted-row and
  residual-gate pair;
- exact preservation of source row order and row ownership.

Does not own: protocol call semantics, receipt replacement, a manifest, or
Rust emission.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NativeCcsActivatedBridge

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

private theorem selected_of_activated_from
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (assignment : ColumnId → Field)
    (lengthEqual : source.length = residuals.length)
    (satisfied :
      RawSatisfies
        (ActivatedRawProgram.rawRows active source residuals) assignment) :
    NativeCcsSelector.Satisfies
      (NativeCcsSelector.select active
        (ownRowsFrom owner ordinal source))
      assignment := by
  induction source generalizing ordinal residuals with
  | nil =>
      trivial
  | cons row source inductionHypothesis =>
      cases residuals with
      | nil =>
          simp at lengthEqual
      | cons residual residuals =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have lifted := satisfied.1
          have gated := satisfied.2.1
          have liftedEquation :
              row.a.eval assignment * row.b.eval assignment =
                row.c.eval assignment + assignment residual := by
            simpa only [ActivatedRawProgram.liftedRow, Row.Holds,
              ActivatedRawProgram.linearCombination_eval_append,
              Goldilocks.singleton, LinearCombination.eval_cons,
              LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero] using
              lifted
          have residualEquation :
              row.a.eval assignment * row.b.eval assignment -
                  row.c.eval assignment =
                assignment residual := by
            calc
              row.a.eval assignment * row.b.eval assignment -
                    row.c.eval assignment =
                  (row.c.eval assignment + assignment residual) -
                    row.c.eval assignment := by rw [liftedEquation]
              _ = assignment residual := by
                rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.add_assoc,
                  Lean.Grind.Fin.add_comm (assignment residual),
                  ← Lean.Grind.Fin.add_assoc,
                  Lean.Grind.AddCommGroup.add_neg_cancel,
                  Fin.zero_add]
          have gatedEquation :
              assignment active * assignment residual = 0 := by
            simpa only [ActivatedRawProgram.gateRow, Row.Holds,
              Goldilocks.singleton, LinearCombination.eval_cons,
              LinearCombination.eval_nil, Fin.one_mul, Fin.add_zero] using
              gated
          constructor
          · unfold NativeCcsSelector.SelectedRow.Holds
              NativeCcsSelector.polynomial
            rw [residualEquation]
            exact gatedEquation
          · exact
              inductionHypothesis (ordinal + 1) residuals
                lengthEqual satisfied.2.2

/-- Every satisfying legacy activated program satisfies the native selector
program on the same assignment. Residual values can remain present, but they
are not read or allocated by the native program. -/
theorem selected_of_activated
    (owner : PhysicalOwner)
    (active : ColumnId)
    (source : List Row)
    (residuals : List ColumnId)
    (assignment : ColumnId → Field)
    (lengthEqual : source.length = residuals.length)
    (satisfied :
      Goldilocks.Satisfies
        (ActivatedRawProgram.rows owner active source residuals) assignment) :
    NativeCcsSelector.Satisfies
      (NativeCcsSelector.select active (ownRows owner source))
      assignment := by
  apply selected_of_activated_from owner 0 active source residuals assignment
    lengthEqual
  exact
    (satisfies_ownRows_iff owner
      (ActivatedRawProgram.rawRows active source residuals)
      assignment).mp
      (by simpa [ActivatedRawProgram.rows] using satisfied)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NativeCcsActivatedBridge

import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorSound

/-!
Executable semantics for the compact terminal accumulator-v1 owner.

| Check | Mathematical obligation | Reads exact R1CS rows? |
|---|---|---|
| `prefix` | All 27 normalized prefix definitions hold | no |
| `source` | The 1,682 source values equal the checked PiDEC-parent projection | no |
| `digest` | The four output lanes equal a pure Poseidon2 replay | no |

Owns: independent executable acceptance of the three semantic branches.
Does not own: PiDEC/PiRLC authority or `y_zcol` validation.
Authority boundary: digest equality is checked against the derived preimage;
the digest does not become authority on its own.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorNative

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryAccumulatorClaimSerialization

/-- Row-free native evaluation of every obligation exposed by `Facts`. -/
def nativeCheck (assignment : Nat → Nat) : Bool :=
  [ assignmentCheck segment0Instructions assignment
  , decide
      (accumulatorClaimSourceColumns.map assignment =
        terminalParentPreimage assignment)
  , accumulatorTrace.valueCheck assignment
  ].all id

theorem nativeCheck_eq_true_iff (assignment : Nat → Nat) :
    nativeCheck assignment = true ↔
      FPrimeFullHistoryTerminalAccumulatorSound.Facts assignment := by
  simp only [nativeCheck, List.all_cons, List.all_nil, id_eq,
    Bool.and_eq_true, and_true, assignmentCheck_eq_true_iff,
    decide_eq_true_eq, Poseidon2Sponge.Trace.valueCheck_eq_true_iff]
  constructor
  · rintro ⟨segment0, parentClaimSource, accumulatorDigest⟩
    exact ⟨segment0, parentClaimSource, accumulatorDigest⟩
  · intro facts
    exact ⟨facts.segment0, facts.parentClaimSource, facts.accumulatorDigest⟩

/-- Exact rows force the independent row-free checker to accept. -/
theorem nativeCheck_of_satisfies
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    nativeCheck assignment = true :=
  (nativeCheck_eq_true_iff assignment).2
    (FPrimeFullHistoryTerminalAccumulatorSound.sound canonical one satisfies)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorNative

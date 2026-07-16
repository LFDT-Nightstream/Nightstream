import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryRecursiveAccumulatorCoreSound

/-!
Executable acceptance for the compact recursive accumulator-digest core.

| Check | Mathematical obligation | Reads R1CS rows? |
|---|---|---|
| prefix | All 27 exact definitions hold on the supplied assignment | no |
| source | The 1,682 source values equal the decoded recursive-parent projection | no |
| digest | The four outputs equal a value-level Poseidon2 execution | no |

Owns: an independent checker for the three core obligations.
Does not own: PiDEC acceptance, PiRLC authority, `y_zcol` validation, or digest
authority.  Exact-row soundness reaches this checker only through
`FPrimeFullHistoryRecursiveAccumulatorCoreSound.sound`.

Assurance tier: artifact-checked executable correspondence for the exact
recursive accumulator core.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreNative

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCore
open Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSound

/-- Independent executable acceptance of every semantic obligation owned by
the exact compact recursive accumulator core. -/
def nativeCheck (assignment : Nat → Nat) : Bool :=
  [ assignmentCheck segment0Instructions assignment
  , decide
      (accumulatorClaimSourceColumns.map assignment =
        FPrimeFullHistoryAccumulatorClaimSerialization.recursiveParentPreimage
          assignment)
  , digestTrace.valueCheck assignment
  ].all id

theorem nativeCheck_eq_true_iff (assignment : Nat → Nat) :
    nativeCheck assignment = true ↔ Facts assignment := by
  simp only [nativeCheck, List.all_cons, List.all_nil, id_eq,
    Bool.and_eq_true, and_true, assignmentCheck_eq_true_iff,
    Poseidon2Sponge.Trace.valueCheck_eq_true_iff, decide_eq_true_eq]
  constructor
  · rintro ⟨segment0, parentClaimSource, accumulatorDigest⟩
    exact ⟨segment0, parentClaimSource, accumulatorDigest⟩
  · intro facts
    exact ⟨facts.segment0, facts.parentClaimSource, facts.accumulatorDigest⟩

/-- Exact rows force the independent executable checker to accept. -/
theorem nativeCheck_of_satisfies
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    nativeCheck assignment = true :=
  (nativeCheck_eq_true_iff assignment).2 (sound canonical one satisfies)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreNative

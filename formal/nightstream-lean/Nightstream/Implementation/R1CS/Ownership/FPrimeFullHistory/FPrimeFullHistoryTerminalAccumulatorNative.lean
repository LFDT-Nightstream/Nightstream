import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryTerminalAccumulatorSound

/-!
Executable semantics for the exact terminal accumulator compiler.

The checker evaluates normalized definitions/assertions, canonical ternary
openings, seeded linear maps, and both Poseidon2 sponge functions directly.
It never calls the R1CS `Satisfies` predicate.  Exact-row soundness reaches
this checker through `FPrimeFullHistoryTerminalAccumulatorSound.sound`.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorNative

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorSound

def canonicalMapsCheck (assignment : Nat → Nat) : Bool :=
  shiftedTernaryMaps.all fun mapping =>
    canonicalOpeningCheck (Pulled mapping assignment)

theorem canonicalMapsCheck_eq_true_iff (assignment : Nat → Nat) :
    canonicalMapsCheck assignment = true ↔
      ∀ mapping ∈ shiftedTernaryMaps,
        CanonicalOpening (Pulled mapping assignment) := by
  simp [canonicalMapsCheck, List.all_eq_true,
    canonicalOpeningCheck_eq_true_iff]

def traceFourCheck (trace : Poseidon2Sponge.Trace)
    (assignment : Nat → Nat) : Bool :=
  (List.range 4).all fun lane =>
    decide
      (assignment (trace.outputColumns.getD lane 0) =
        Poseidon2Sponge.runValueRounds trace.rounds
          (trace.inputColumns.map assignment) (fun _ => 0) lane)

theorem traceFourCheck_eq_true_iff (trace : Poseidon2Sponge.Trace)
    (assignment : Nat → Nat) :
    traceFourCheck trace assignment = true ↔
      ∀ lane, lane < 4 →
        assignment (trace.outputColumns.getD lane 0) =
          Poseidon2Sponge.runValueRounds trace.rounds
            (trace.inputColumns.map assignment) (fun _ => 0) lane := by
  simp only [traceFourCheck, List.all_eq_true, decide_eq_true_eq]
  constructor
  · intro checked lane laneLt
    exact checked lane (List.mem_range.mpr laneLt)
  · intro accepted lane laneMember
    exact accepted lane (List.mem_range.mp laneMember)

/-- Independent executable acceptance of every semantic obligation owned by
the exact terminal accumulator compiler. -/
def nativeCheck (assignment : Nat → Nat) : Bool :=
  [ assignmentCheck segment0Instructions assignment
  , assignmentCheck segment1Instructions assignment
  , assignmentCheck segment2Instructions assignment
  , canonicalMapsCheck assignment
  , FPrimeFullHistorySeededPhi81.block16.check assignment
  , FPrimeFullHistorySeededPhi81.block17.check assignment
  , decide
      (parentCeClaimSourceColumns.map assignment =
        FPrimeFullHistoryParentCeSerialization.parentPreimage assignment)
  , decide
      ((FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.accumulatorDigestTrace
          ).inputColumns.map assignment =
        (decodedPiDecAuthority assignment).preimage)
  , traceFourCheck
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.parentCeDigestTrace
      assignment
  , traceFourCheck
      FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.accumulatorDigestTrace
      assignment
  ].all id

theorem nativeCheck_eq_true_iff (assignment : Nat → Nat) :
    nativeCheck assignment = true ↔ Facts assignment := by
  simp only [nativeCheck, List.all_cons, List.all_nil, id_eq, Bool.and_eq_true,
    and_true, assignmentCheck_eq_true_iff, canonicalMapsCheck_eq_true_iff,
    SeededPhi81.Block.check_eq_true_iff, traceFourCheck_eq_true_iff,
    decide_eq_true_eq]
  constructor
  · rintro ⟨segment0, segment1, segment2, canonicalOpenings, seeded16,
      seeded17, parentClaimSource, parentAuthorityPreimage, parentCeDigest,
      accumulatorDigest⟩
    exact {
      program := ⟨segment0, segment1, segment2⟩
      canonicalOpenings := canonicalOpenings
      seeded16 := seeded16
      seeded17 := seeded17
      parentClaimSource := parentClaimSource
      parentAuthorityPreimage := parentAuthorityPreimage
      parentCeDigest := parentCeDigest
      accumulatorDigest := accumulatorDigest
    }
  · intro facts
    exact ⟨facts.program.segment0, facts.program.segment1,
      facts.program.segment2, facts.canonicalOpenings, facts.seeded16,
      facts.seeded17, facts.parentClaimSource, facts.parentAuthorityPreimage,
      facts.parentCeDigest, facts.accumulatorDigest⟩

/-- Exact rows force the independent executable checker to accept. -/
theorem nativeCheck_of_satisfies
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    nativeCheck assignment = true :=
  (nativeCheck_eq_true_iff assignment).2
    (sound goldilocksPrime canonical one satisfies)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulatorNative

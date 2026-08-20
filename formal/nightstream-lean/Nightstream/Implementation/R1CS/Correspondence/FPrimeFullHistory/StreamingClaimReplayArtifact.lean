import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: row-soundness boundary for one exact Rust-emitted claim-replay step.

Assurance tier: artifact-checked for property
`FPRIME-STREAMING-CLAIM-REPLAY-ROW-REFINEMENT-V6`.

Owns transport from satisfaction of the exact stored canonical-u64,
Poseidon2, coordinate, and glue rows to their existing Lean semantics.

Does not own sampler liveness, complete artifact validity, interpretation of
glue columns as a replay state, the complete frame theorem, recursive
lifecycle integration, or the Poseidon2 collision reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Program

/-- Every canonical-word leaf in a satisfying arm has the unique canonical
64-bit value specified by the Lean-owned recipe. -/
theorem canonical_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call : CanonicalCall) (member : call ∈ arm.canonicalCalls) :
    Refines assignment call.layout := by
  apply CanonicalU64RecipeSound.sound goldilocks_euclidPrime canonical one
  exact satisfied.1 call member

/-- Every Poseidon2 leaf in a satisfying arm agrees with the exact production
permutation interpreter under its Rust-emitted column renaming. -/
theorem poseidon2_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call : Poseidon2Call.Call) (member : call ∈ arm.poseidon2Calls) :
    AgreeOn
      (Poseidon2PermutationSound.interpret
        (pullAssignment assignment call.columnMap))
      (pullAssignment assignment call.columnMap)
      (knownAfter Poseidon2Permutation.inputColumns
        Poseidon2Permutation.definitions) := by
  apply Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
    call.columnMap call.columnMap_zero canonical one
  simpa [Poseidon2Call.Call.rows] using satisfied.2.1 call member

/-- Every selected coordinate call enforces its compact source, shape, and
seeded output rows. -/
theorem coordinate_call_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (call : CoordinateCall) (member : call ∈ arm.coordinateCalls) :
    Satisfies call.rows assignment := by
  exact satisfied.2.2.1 call member

/-- Glue rows are stored exactly and remain enforced after the repeated leaf
programs are compressed into call certificates. -/
theorem glue_row_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (indexed : IndexedRow) (member : indexed ∈ arm.glueRows) :
    RowHolds assignment indexed.row := by
  exact satisfied.2.2.2 indexed member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact

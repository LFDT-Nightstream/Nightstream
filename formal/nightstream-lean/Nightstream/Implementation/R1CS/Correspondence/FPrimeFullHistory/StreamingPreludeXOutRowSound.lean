import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeXOut

/-!
Contract: satisfying either normalized Prelude XOut row block computes
Poseidon2 from the same 32 assignment values in the Rust-emitted order.

Does not claim that those values are lifecycle-authoritative. That separate
bridge must prove exact frame equality or return a named collision event.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-PRELUDE-XOUT-ROWS-V1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeXOutRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOut.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeXOut

abbrev DigestValues := Fin 4 → Nat

def inputValues (block : HashBlock) (assignment : Nat → Nat) : List Nat :=
  block.recipe.inputColumns.map (artifact.sourceAssignment assignment)

def computedDigest (block : HashBlock) (values : List Nat) : DigestValues :=
  fun lane => runValueRounds block.recipe.trace.rounds values (fun _ => 0) lane.val

def assignedDigest (block : HashBlock) (assignment : Nat → Nat) : DigestValues :=
  fun lane => artifact.sourceAssignment assignment
    (block.recipe.outputColumns.getD lane.val 0)

private theorem rows_imply_hash
    (block : HashBlock) (valid : block.recipe.trace.OwnedValid)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : block.SourceSatisfied (artifact.sourceAssignment assignment)) :
    assignedDigest block assignment = computedDigest block (inputValues block assignment) := by
  funext lane
  apply ownedTrace_values_sound valid
    (fun column => canonical (artifact.normalizedColumn column))
    (by simpa [RawArtifact.sourceAssignment, RawArtifact.normalizedColumn] using one)
    satisfied.1 lane.val lane.isLt

theorem after_rows_imply_hash
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment) :
    assignedDigest artifact.afterXOut assignment =
      computedDigest artifact.afterXOut (inputValues artifact.afterXOut assignment) :=
  rows_imply_hash artifact.afterXOut after_trace_ownedValid assignment canonical one satisfied.1

theorem before_rows_imply_hash
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment) :
    assignedDigest artifact.beforeXOut assignment =
      computedDigest artifact.beforeXOut (inputValues artifact.beforeXOut assignment) :=
  rows_imply_hash artifact.beforeXOut before_trace_ownedValid assignment canonical one satisfied.2

theorem source_rows_imply_hashes
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment) :
    assignedDigest artifact.afterXOut assignment =
        computedDigest artifact.afterXOut (inputValues artifact.afterXOut assignment) ∧
      assignedDigest artifact.beforeXOut assignment =
        computedDigest artifact.beforeXOut (inputValues artifact.beforeXOut assignment) :=
  ⟨after_rows_imply_hash assignment canonical one satisfied,
    before_rows_imply_hash assignment canonical one satisfied⟩

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeXOutRowSound

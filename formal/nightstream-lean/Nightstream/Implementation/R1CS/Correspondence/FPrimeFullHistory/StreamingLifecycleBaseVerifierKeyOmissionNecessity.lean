import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmission

/-!
Contract: Lean-checked mutation for the exact base verifier-key omission audit.

Rust owns the complete assignment and its replay against every retained source
row. The generated exhaustive column projection proves that every use of the
changed column belongs to the removed family. This module proves that the same
mutation is canonical, preserves the constant-one column, and violates the
typed verifier-owned target.

Assurance tier: Rust-conformant for the Nightstream b2/k16 base lifecycle
profile when combined with the exact Rust replay.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmissionNecessity

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmission

def omissionAssignment (baseline : Nat -> Nat) (column : Nat) : Nat :=
  if column = rawArtifact.changedColumn then rawArtifact.candidateValue
  else baseline column

theorem omissionAssignment_one
    (baseline : Nat -> Nat)
    (baselineOne : baseline rawArtifact.constantOneColumn = 1) :
    omissionAssignment baseline rawArtifact.constantOneColumn = 1 := by
  have different :
      rawArtifact.constantOneColumn ≠ rawArtifact.changedColumn := by decide
  simp [omissionAssignment, different, baselineOne]

theorem omissionAssignment_canonical
    (baseline : Nat -> Nat)
    (baselineCanonical : forall column, baseline column < goldilocksP)
    (column : Nat) : omissionAssignment baseline column < goldilocksP := by
  by_cases changed : column = rawArtifact.changedColumn
  · simp [omissionAssignment, changed, rawArtifact_valid.candidateCanonical]
  · simp [omissionAssignment, changed, baselineCanonical]

theorem typedTarget_fails (baseline : Nat -> Nat) :
    omissionAssignment baseline rawArtifact.changedColumn ≠
      rawArtifact.baselineValue := by
  simpa [omissionAssignment] using rawArtifact_valid.targetFails

/-- Lean-checked component of the exact Rust removal counterexample. Rust
separately replays the full assignment against every retained emitted row. -/
theorem exact_removal_counterexample
    (baseline : Nat -> Nat)
    (baselineOne : baseline rawArtifact.constantOneColumn = 1)
    (baselineCanonical : forall column, baseline column < goldilocksP) :
    omissionAssignment baseline rawArtifact.constantOneColumn = 1 /\
      (forall column, omissionAssignment baseline column < goldilocksP) /\
      RetainedRowsIgnoreChangedColumn rawArtifact /\
      omissionAssignment baseline rawArtifact.changedColumn ≠
        rawArtifact.baselineValue :=
  ⟨omissionAssignment_one baseline baselineOne,
    omissionAssignment_canonical baseline baselineCanonical,
    rawArtifact_valid.retainedRowsIgnore, typedTarget_fails baseline⟩

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmissionNecessity

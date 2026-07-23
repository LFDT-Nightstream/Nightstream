import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedSourceSoundness

/-!
Literal retained source-row truth on the generated compiler assignment.

Owns: lossless transport from the already artifact-checked 52 decoded
retained obligations to satisfaction of the 52 literal generated source rows.

Does not own: reconstructed-source agreement, eliminated rows, selector or
constant-one enforcement, transcript order, parent or raw-child authority,
commitment binding, costs, or row removal.

No executable certificate occurs here.  Each raw row is recovered from its
exact retained source pair, decoded using that pair's checked validity, and
identified with the typed row by the existing lossless decoder theorem.
-/

/-!
Emits constraints: none; this module proves semantics of retained compiler rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.retained_compiler_rows` | Decode retained checks and relate their physical rows to source obligations. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedCompilerRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

theorem rawRowsSatisfy_of_decodedRowsSatisfy
    {assignment : Nat → Nat}
    (satisfies :
      RetainedSourceArtifact.GeneratedRetainedSourceRowsSatisfy assignment) :
    Satisfies
      (RetainedSourceArtifact.retainedSourceRows.map
        SourceDecodeBridge.rawRow)
      (SourceAssignment.compilerAssignment assignment) := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨raw, rawMember, rfl⟩
  have pairSourceMember : raw ∈
      RetainedSourceArtifact.retainedSourcePairs.map
        RetainedSourceArtifact.RawRetainedSourcePair.source := by
    rw [RetainedSourceArtifact.sourceRowsExact]
    exact rawMember
  rcases List.mem_map.mp pairSourceMember with
    ⟨pair, pairMember, sourceEqual⟩
  subst raw
  have valid := RetainedSourceArtifact.pairValid pairMember
  rcases SourceDecodeBridge.decodeSourceRow_of_valid valid.1 with
    ⟨decoded, decodes⟩
  have typedHolds := satisfies pair.source (by
      have mapped : pair.source ∈
          RetainedSourceArtifact.retainedSourcePairs.map
            RetainedSourceArtifact.RawRetainedSourcePair.source :=
        List.mem_map.mpr ⟨pair, pairMember, rfl⟩
      rw [RetainedSourceArtifact.sourceRowsExact] at mapped
      exact mapped) decoded decodes
  have rowEquality :=
    SourceDecodeBridge.sourceRowToRow_eq_rawRow_of_decode decodes
  unfold Semantics.SourceRowHolds at typedHolds
  rw [rowEquality] at typedHolds
  exact typedHolds

/-- Literal selected-row satisfaction establishes every exact retained raw
source row on the unique generated compiler assignment. -/
theorem generatedEmittedRowsSatisfy_implies_retainedRawRowsSatisfy
    {assignment : Nat → Nat}
    (satisfies :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    Satisfies
      (RetainedSourceArtifact.retainedSourceRows.map
        SourceDecodeBridge.rawRow)
      (SourceAssignment.compilerAssignment assignment) := by
  apply rawRowsSatisfy_of_decodedRowsSatisfy
  exact
    RetainedSourceSoundness.generatedEmittedRowsSatisfy_implies_retainedSourceRowsSatisfy
      satisfies selectorOne

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RetainedCompilerRows

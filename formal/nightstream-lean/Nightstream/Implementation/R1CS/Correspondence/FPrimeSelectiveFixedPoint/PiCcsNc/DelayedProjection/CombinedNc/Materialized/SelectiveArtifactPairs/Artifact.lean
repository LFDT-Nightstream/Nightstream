import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Certificates
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Decode

/-!
Artifact-checked compiler obligations for every generated production
combined-NC selective row.

Owns: the exact generated emitted-row acceptance predicate and its transport
to all 1,493 decoded rewrite recurrences and all 52 decoded retained
obligations.

Does not own: reconstruction of eliminated source equations from rewrite
ranges, retained-step pairing with physical source rows, selector enforcement,
source-program execution, transcript order, parent or raw-child authority,
commitment binding, costs, or row removal.

Emits constraints: none.

This is deliberately not a theorem that all 8,021 selected source rows hold.
Rewrite provenance carries ranges, not the source equations themselves.  The
next source-program leaf must show that the checked recurrences reconstruct
those eliminated equations; assuming their satisfaction here would make the
refinement circular.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.selective_pairs.artifact` | Prove exact membership and pairing facts for generated selective rows. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Artifact

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Certificates

/-- Satisfaction of the literal generated emitted-row stream.  Decode
failure is not accepted: the artifact certificates separately construct a
typed decoding for every member. -/
def GeneratedEmittedRowsSatisfy (assignment : Nat → Nat) : Prop :=
  ∀ raw ∈ EmittedRows.values,
    ∀ decoded, Decoder.decodeEmittedRow raw = some decoded →
      Semantics.EmittedRowHolds decoded assignment

def AllRewriteObligationsHold (assignment : Nat → Nat) : Prop :=
  ∀ pair ∈ rewritePairs,
    ∃ emitted provenance,
      Decoder.decodeEmittedRow pair.emitted = some emitted ∧
      Decoder.decodeRewriteStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.provenance = some provenance ∧
      SelectiveCompilerBridge.RewriteStepHolds
        (SourceAssignment.compilerAssignment assignment)
        (SourceAssignment.derivedValue assignment) provenance

def AllRetainedObligationsHold (assignment : Nat → Nat) : Prop :=
  ∀ pair ∈ retainedPairs,
    ∃ emitted provenance,
      Decoder.decodeEmittedRow pair.emitted = some emitted ∧
      Decoder.decodeRetainedStep Metadata.sourceRelationRows
        Metadata.sourceRelationColumns Metadata.finalRelationRows
          pair.provenance = some provenance ∧
      SelectiveCompilerBridge.RetainedStepHolds
        (SourceAssignment.compilerAssignment assignment) provenance

private theorem rewritePairRowMember {pair : RawRewritePair}
    (member : pair ∈ rewritePairs) : pair.emitted ∈ EmittedRows.values := by
  have inMapped : pair.emitted ∈ rewritePairs.map RawRewritePair.emitted :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  have inAll : pair.emitted ∈
      retainedPairs.map RawRetainedPair.emitted ++
        rewritePairs.map RawRewritePair.emitted :=
    List.mem_append_right _ inMapped
  rw [allPairedEmittedRowsExact] at inAll
  exact inAll

private theorem retainedPairRowMember {pair : RawRetainedPair}
    (member : pair ∈ retainedPairs) : pair.emitted ∈ EmittedRows.values := by
  have inMapped : pair.emitted ∈ retainedPairs.map RawRetainedPair.emitted :=
    List.mem_map.mpr ⟨pair, member, rfl⟩
  have inAll : pair.emitted ∈
      retainedPairs.map RawRetainedPair.emitted ++
        rewritePairs.map RawRewritePair.emitted :=
    List.mem_append_left _ inMapped
  rw [allPairedEmittedRowsExact] at inAll
  exact inAll

/-- Exact selected-row soundness for the compiler surface.  The only extra
premise is the still-external production selector-one equation. -/
theorem generatedEmittedRowsSatisfy_implies_allCompilerObligations
    {assignment : Nat → Nat}
    (satisfies : GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    AllRewriteObligationsHold assignment ∧
      AllRetainedObligationsHold assignment := by
  constructor
  · intro pair member
    rcases exists_rewriteProvenanceMatches_of_certificate
        (rewritePairsCertified pair member) assignment selectorOne with
      ⟨emitted, provenance, emittedDecodes, provenanceDecodes,
        factorCapacity, matching⟩
    refine ⟨emitted, provenance, emittedDecodes, provenanceDecodes, ?_⟩
    exact SelectiveCompilerBridge.emittedRowHolds_implies_rewriteStepHolds
      factorCapacity
      (satisfies pair.emitted (rewritePairRowMember member)
        emitted emittedDecodes)
      matching
  · intro pair member
    rcases exists_retainedProvenanceMatches_of_certificate
        (retainedPairsCertified pair member) assignment selectorOne with
      ⟨emitted, provenance, emittedDecodes, provenanceDecodes, matching⟩
    refine ⟨emitted, provenance, emittedDecodes, provenanceDecodes, ?_⟩
    exact SelectiveCompilerBridge.emittedRowHolds_implies_retainedStepHolds
      (satisfies pair.emitted (retainedPairRowMember member)
        emitted emittedDecodes)
      matching

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Artifact

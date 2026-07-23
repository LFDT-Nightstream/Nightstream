import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Artifact

/-!
Exact generated-provenance consequences of the production combined-NC
selective rows.

Owns: transport from literal emitted-row satisfaction to one decoded
semantic obligation for every generated rewrite and retained provenance
record, in the generated order.

Does not own: source-program reconstruction, selector enforcement,
constant-one enforcement, transcript order, parent or raw-child authority,
commitment binding, costs, or permission to remove rows.

Emits constraints: none.

The pair certificates remain the coefficient authority, but callers no
longer need to reason about their auxiliary zip lists.  Coverage is obtained
from the already checked exact pair-to-provenance equalities.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.selective_obligations` | Collect the exact source obligations owned by selected generated rows. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveObligations

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open SelectiveArtifactPairs

/-- Every exact generated rewrite record decodes and its checked recurrence
holds on the independently reconstructed compiler-column view. -/
def GeneratedRewriteObligationsHold (assignment : Nat → Nat) : Prop :=
  ∀ raw ∈ Provenance.rewriteSteps,
    ∃ decoded,
      Decoder.decodeRewriteStep Metadata.sourceRelationRows
          Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded ∧
      SelectiveCompilerBridge.RewriteStepHolds
        (SourceAssignment.compilerAssignment assignment)
        (SourceAssignment.derivedValue assignment) decoded

/-- Every exact generated retained record decodes and its physical source
equation holds on the same compiler-column view. -/
def GeneratedRetainedObligationsHold (assignment : Nat → Nat) : Prop :=
  ∀ raw ∈ Provenance.retainedSteps,
    ∃ decoded,
      Decoder.decodeRetainedStep Metadata.sourceRelationRows
          Metadata.sourceRelationColumns Metadata.finalRelationRows raw =
        some decoded ∧
      SelectiveCompilerBridge.RetainedStepHolds
        (SourceAssignment.compilerAssignment assignment) decoded

private theorem rewritePair_exists_of_generated
    {raw : RawRewriteStep} (member : raw ∈ Provenance.rewriteSteps) :
    ∃ pair ∈ Chunks.rewritePairs, pair.provenance = raw := by
  have mapped : raw ∈
      Chunks.rewritePairs.map RawRewritePair.provenance := by
    rw [Certificates.rewritePairStepsExact, Chunks.rewriteStepsExact]
    exact member
  rcases List.mem_map.mp mapped with ⟨pair, pairMember, equality⟩
  exact ⟨pair, pairMember, equality⟩

private theorem retainedPair_exists_of_generated
    {raw : RawRetainedStep} (member : raw ∈ Provenance.retainedSteps) :
    ∃ pair ∈ Chunks.retainedPairs, pair.provenance = raw := by
  have mapped : raw ∈
      Chunks.retainedPairs.map RawRetainedPair.provenance := by
    rw [Certificates.retainedPairStepsExact, Chunks.retainedStepsExact]
    exact member
  rcases List.mem_map.mp mapped with ⟨pair, pairMember, equality⟩
  exact ⟨pair, pairMember, equality⟩

/-- Literal selected-row satisfaction yields the exact ordered generated
rewrite and retained obligations.  The only additional premise is the
production steady-selector equation, which remains externally visible. -/
theorem generatedEmittedRowsSatisfy_implies_generatedObligations
    {assignment : Nat → Nat}
    (satisfies :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1) :
    GeneratedRewriteObligationsHold assignment ∧
      GeneratedRetainedObligationsHold assignment := by
  rcases
      SelectiveArtifactPairs.Artifact.generatedEmittedRowsSatisfy_implies_allCompilerObligations
        satisfies selectorOne with ⟨rewrites, retained⟩
  constructor
  · intro raw member
    rcases rewritePair_exists_of_generated member with
      ⟨pair, pairMember, provenanceEq⟩
    rcases rewrites pair pairMember with
      ⟨emitted, decoded, emittedDecodes, provenanceDecodes, holds⟩
    subst raw
    exact ⟨decoded, provenanceDecodes, holds⟩
  · intro raw member
    rcases retainedPair_exists_of_generated member with
      ⟨pair, pairMember, provenanceEq⟩
    rcases retained pair pairMember with
      ⟨emitted, decoded, emittedDecodes, provenanceDecodes, holds⟩
    subst raw
    exact ⟨decoded, provenanceDecodes, holds⟩

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveObligations

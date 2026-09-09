import NightstreamFPrime.Export.Stage1.ActualPiRLCCandidates

/-!
Owns exact bounded sampling from arbitrary accepted decoder, selector and
Poseidon rows. The output is the retained First54 value list, and the source
is the concrete verifier stream from the retained PiCCS endpoint.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCSampling

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Sampling
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler (sourceAt)
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open ActualPiRLCCandidates (inputs verifierCandidate)

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def candidates
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    List Chunk :=
  FirstAccepted.streamPrefix
    (sourceAt Transcript.PiRlcSampler.specification
      (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
        assignment) source.val).stream candidateBound

theorem candidatePrefix_eq
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    (First54.semanticCandidates First54.candidateCount).map
        (fun round => verifierCandidate geometry assignment ⟨source, round⟩) =
      candidates geometry assignment source := by
  unfold candidates First54.semanticCandidates First54.candidateStream FirstAccepted.streamPrefix
  simp only [First54.candidateCount, candidateBound]
  change ((List.range 64).map First54.candidateIndex).map
      (fun round : Fin First54.candidateCount => verifierCandidate geometry assignment ⟨source, round⟩) = _
  rw [List.map_map]
  apply List.map_congr_left
  intro index member
  have indexLt : index < 64 := List.mem_range.mp member
  simp [verifierCandidate, First54.candidateIndex, First54.candidateCount,
    Nat.mod_eq_of_lt indexLt]

theorem rowsZero_implies_acceptedSymbols
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (decoderRows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    FirstAccepted.acceptedSymbols
        (First54.semanticVerifier (ActualPiRLCSelector.interface (inputs geometry) assignment source)
          0 (ActualPiRLCSelector.decodedEnv (inputs geometry) assignment source))
        (First54.semanticCandidates First54.candidateCount) =
      (FirstAccepted.acceptedSymbols verifier (candidates geometry assignment source)).map
        Sampler.coefficientWord := by
  let selected := First54.semanticVerifier
    (ActualPiRLCSelector.interface (inputs geometry) assignment source) 0
    (ActualPiRLCSelector.decodedEnv (inputs geometry) assignment source)
  let chunk := fun round => verifierCandidate geometry assignment ⟨source, round⟩
  have acceptsEq : selected.accepts = verifier.accepts ∘ chunk := by
    funext round
    exact (ActualPiRLCCandidates.rowsZero_implies_selector_verifier relation geometry assignment
      one decoderRows semantics source round).1
  have symbolEq : selected.symbol = Sampler.coefficientWord ∘ verifier.symbol ∘ chunk := by
    funext round
    exact (ActualPiRLCCandidates.rowsZero_implies_selector_verifier relation geometry assignment
      one decoderRows semantics source round).2
  change FirstAccepted.acceptedSymbols selected (First54.semanticCandidates First54.candidateCount) = _
  rw [← candidatePrefix_eq geometry assignment source]
  change FirstAccepted.acceptedSymbols selected (First54.semanticCandidates First54.candidateCount) =
    (FirstAccepted.acceptedSymbols verifier
      ((First54.semanticCandidates First54.candidateCount).map chunk)).map Sampler.coefficientWord
  simp only [FirstAccepted.acceptedSymbols, FirstAccepted.acceptedCandidates,
    List.filter_map, List.map_map, acceptsEq, symbolEq]
  rfl

/-- The real bounded verifier succeeds and returns exactly the retained words. -/
theorem rowsZero_implies_boundedSample
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (decoderRows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (selectorRows : (PiRLCFirst54DirectPlan.plan (inputs geometry)).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    ∃ coefficients,
      FirstAccepted.boundedSample verifier coefficientCount (candidates geometry assignment source) =
          some coefficients ∧
        ActualPiRLCSelector.outputValues (inputs geometry) assignment source =
          coefficients.map Sampler.coefficientWord := by
  have selector := ActualPiRLCCandidates.rowsZero_implies_bounded_selector relation geometry
    assignment one decoderRows selectorRows source
  have symbols := rowsZero_implies_acceptedSymbols relation geometry assignment one decoderRows
    semantics source
  have counts := congrArg List.length symbols
  simp only [FirstAccepted.acceptedSymbols, List.length_map] at counts
  have enough := (FirstAccepted.boundedSample_eq_some_iff.mp selector).1
  change coefficientCount ≤ _ at enough
  unfold FirstAccepted.acceptedCount at enough
  rw [counts] at enough
  refine ⟨FirstAccepted.firstAccepted verifier coefficientCount (candidates geometry assignment source),
    FirstAccepted.boundedSample_eq_some_iff.mpr ⟨enough, rfl⟩, ?_⟩
  rw [FirstAccepted.bounded_success_exact selector]
  unfold FirstAccepted.firstAccepted
  rw [symbols, List.map_take]
  rfl

def challenge
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    RingF :=
  fun lane => ((inputs geometry).value
    (PiRLCProductSourceBlocks.challengeValueDescriptor source lane)).eval assignment - 2

theorem boundedSample_implies_challenge
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (coefficients : List Coefficient)
    (success : FirstAccepted.boundedSample verifier coefficientCount
      (candidates geometry assignment source) = some coefficients)
    (output : ActualPiRLCSelector.outputValues (inputs geometry) assignment source =
      coefficients.map Sampler.coefficientWord) :
    challenge geometry assignment source =
      Phi81StrongSet.embedScalar (Transcript.PiRlcSampler.scalarOfList coefficients) := by
  have coefficientLength := FirstAccepted.bounded_success_length success
  funext lane
  have coefficientLt : lane.val < coefficients.length := by
    rw [coefficientLength]
    exact lane.isLt
  have outputLt : lane.val < First54.outputCount := lane.isLt
  have word : ((inputs geometry).value
      (PiRLCProductSourceBlocks.challengeValueDescriptor source lane)).eval assignment =
      Sampler.coefficientWord (coefficients.getD lane.val ⟨2, by decide⟩) := by
    have selected := congrArg (fun values : List F => values.getD lane.val 0) output
    simpa [ActualPiRLCSelector.outputValues, PiRLCProductSourceBlocks.challengeValueDescriptor,
      List.getD_eq_getElem?_getD, coefficientLt, outputLt] using selected
  change ((inputs geometry).value
      (PiRLCProductSourceBlocks.challengeValueDescriptor source lane)).eval assignment - 2 = _
  rw [word, Sampler.coefficientWord_sub_two_eq_embedCoefficient]
  rfl

/-- Each retained challenge is the successful result of the actual verifier sampler. -/
theorem rowsZero_implies_challenge
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (decoderRows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (selectorRows : (PiRLCFirst54DirectPlan.plan (inputs geometry)).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    Transcript.PiRlcSampler.sampleRingChallenge
        (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
          assignment) source.val = some (challenge geometry assignment source) := by
  obtain ⟨coefficients, success, output⟩ := rowsZero_implies_boundedSample relation geometry
    assignment one decoderRows selectorRows semantics source
  change ((FirstAccepted.boundedSample verifier coefficientCount (candidates geometry assignment source)).map
    Transcript.PiRlcSampler.scalarOfList).map Phi81StrongSet.embedScalar = _
  rw [success]
  exact congrArg some (boundedSample_implies_challenge geometry assignment source coefficients
    success output).symm

/-- The complete verifier batch agrees with all retained challenges and its final state. -/
theorem rowsZero_implies_batch
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (decoderRows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (selectorRows : (PiRLCFirst54DirectPlan.plan (inputs geometry)).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment) :
    Transcript.PiRlcSampler.piRlcChallengesWithState
        (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
          assignment) PiRLCFirst54DirectSchedule.sourceCount =
      some ⟨challenge geometry assignment,
        ActualPiRLCStates.state (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
          ⟨16, by decide⟩ ⟨8, by decide⟩⟩ := by
  have challenges := Transcript.PiRlcSampler.piRlcChallenges_eq_some_of_pointwise
    (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (challenge geometry assignment)
    (rowsZero_implies_challenge relation geometry assignment one decoderRows selectorRows semantics)
  cases batchEq : Transcript.PiRlcSampler.piRlcChallengesWithState
      (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
      PiRLCFirst54DirectSchedule.sourceCount with
  | none => simp [Transcript.PiRlcSampler.piRlcChallenges, batchEq] at challenges
  | some batch =>
      have challengeEq : batch.challenges = challenge geometry assignment := by
        simpa [Transcript.PiRlcSampler.piRlcChallenges, batchEq] using challenges
      have stateEq := (Transcript.PiRlcSampler.piRlcChallengesWithState_finalState batchEq).trans
        (ActualPiRLCStates.final_eq_stateAt _ _ semantics).symm
      apply congrArg some
      cases batch with
      | mk actualChallenges actualState =>
          dsimp only at challengeEq stateEq
          cases challengeEq
          cases stateEq
          rfl

end NightstreamFPrime.Export.Stage1.ActualPiRLCSampling

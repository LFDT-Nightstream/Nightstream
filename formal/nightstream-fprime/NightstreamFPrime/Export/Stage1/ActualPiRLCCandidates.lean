import NightstreamFPrime.Export.Stage1.ActualPiRLCSelector
import NightstreamFPrime.Export.Stage1.ActualPiRLCStates
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlanSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Owns candidate-decoder soundness for arbitrary accepted sampler rows. The
decoder and First54 consume the same physical reject and symbol forms.
Poseidon state evolution and exact verifier sampling are separate inputs.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCCandidates

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerOrdinaryRetainedBlocks
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler (sourceAt stateAt)
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule (stateBeforeBlock)

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def inputs (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiRLCFirst54DirectPlan.Inputs program logicalWidth :=
  PiRLCRetainedInputs.first54Inputs (PiRLCSamplerOrdinaryRetainedGeometry.piRlcGeometry geometry)

theorem decoded_logical
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment))
        (logicalSource descriptor position) =
      (PiRLCSamplerCandidateWiring.logicalForm geometry descriptor position).eval assignment := by
  unfold Spartan.pullback PiRLCSamplerOrdinaryDirectPlan.resolvedEnv
    PiRLCSamplerOrdinaryDirectPlan.resolvedForm PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan _ (logicalSource_lt descriptor position)]
  simp only [PiRLCSamplerOrdinaryDirectPlan.classifySource_logical]
  rfl

theorem reject_eval
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Lane) (part : Fin 2) :
    (Candidate16Five.rejectExpr (DigestLane.decoderOffset
        (PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
          descriptor.lane.val) part)).eval
        (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)) =
      ((inputs geometry).reject (PiRLCSamplerCandidateWiring.candidate descriptor part)).eval
        assignment := by
  change (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment))
      (PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
        descriptor.lane.val + 66 + part.val * 17 + 16) = _
  have same : PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
      descriptor.lane.val + 66 + part.val * 17 + 16 =
        logicalSource descriptor (PiRLCSamplerCandidateWiring.rejectPosition part) := by
    simp only [logicalSource, PiRLCSamplerCandidateWiring.rejectPosition]
    omega
  rw [same, decoded_logical, PiRLCSamplerCandidateWiring.logicalForm_reject]
  rfl

theorem symbol_eval
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Lane) (part : Fin 2) :
    (Candidate16Five.remainderExpr (DigestLane.decoderOffset
        (PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
          descriptor.lane.val) part)).eval
        (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)) =
      ((inputs geometry).symbol (PiRLCSamplerCandidateWiring.candidate descriptor part)).eval
        assignment := by
  change (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment))
      (PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
        descriptor.lane.val + 66 + part.val * 17 + 1) = _
  have same : PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
      descriptor.lane.val + 66 + part.val * 17 + 1 =
        logicalSource descriptor (PiRLCSamplerCandidateWiring.symbolPosition part) := by
    simp only [logicalSource, PiRLCSamplerCandidateWiring.symbolPosition]
    omega
  rw [same, decoded_logical, PiRLCSamplerCandidateWiring.logicalForm_symbol]
  rfl

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

/-- Each actual digest lane satisfies the existing canonical-word and
candidate-decoder child specifications. Only static scope facts are used. -/
theorem rowsZero_implies_lane
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (descriptor : Lane) :
    DigestLane.SpecHolds
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        descriptor.source.val descriptor.round.val descriptor.lane)
      (PiRLCStarts.digestLaneLogicalStart descriptor.source.val
        descriptor.round.val descriptor.lane.val)
      (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)) := by
  have ordinary := (PiRLCSamplerOrdinaryDirectPlan.rowsZero_iff_rowsHold relation geometry
    assignment one).mp rows
  have scope : DigestLane.Assumptions
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        descriptor.source.val descriptor.round.val descriptor.lane)
      (PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
        descriptor.lane.val)
      (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)) := by
    unfold DigestLane.Assumptions
    rw [← PiRLCSamplerOrdinaryRows.fastLaneSource_eq]
    exact PiRLCSamplerOrdinaryRows.fastLaneSource_varsBelow descriptor.source.val
      descriptor.round.val descriptor.lane descriptor.round.isLt
  exact PiRLCSamplerOrdinaryRows.rows_imply_laneSpec descriptor.source.val
    descriptor.round.val descriptor.lane descriptor.source.isLt descriptor.round.isLt
    (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment) scope ordinary

theorem rowsZero_implies_decoder
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (descriptor : Lane) (part : Fin 2) :
    Candidate16Five.SpecHolds
      (DigestLane.decoderInterface (PiRLCStarts.digestLaneLogicalStart descriptor.source.val
        descriptor.round.val descriptor.lane.val) part)
      (DigestLane.decoderOffset (PiRLCStarts.digestLaneLogicalStart descriptor.source.val
        descriptor.round.val descriptor.lane.val) part)
      (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)) :=
  (rowsZero_implies_lane relation geometry assignment one rows descriptor).decoder part

/-- Decoder source words are the actual retained Poseidon rate lanes. -/
theorem lane_eval
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (descriptor : Lane) :
    ((PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := relationLogicalWidth) (publicFits := relationPublicFits)
        descriptor.source.val descriptor.round.val descriptor.lane).source
      (PiRLCStarts.digestLaneLogicalStart descriptor.source.val descriptor.round.val
        descriptor.lane.val)).eval
        (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment)) =
      (ActualPiRLCStates.state (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
        assignment descriptor.source (PiRLCSamplerDirectSemantics.priorStep descriptor.round)).getD
        descriptor.lane.val 0 := by
  rw [← PiRLCSamplerOrdinaryRows.fastLaneSource_eq,
    PiRLCSamplerOrdinaryDirectSource.fastLaneSource_eq_var]
  change PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
      (Spartan.sourceToSpartan (PiRLCSamplerOrdinaryDirectSource.poseidonSource
        descriptor.source.val descriptor.round.val descriptor.lane)) = _
  have retained : PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment
      (Spartan.sourceToSpartan (PiRLCSamplerOrdinaryDirectSource.poseidonSource
        descriptor.source.val descriptor.round.val descriptor.lane)) =
      PiRLCSamplerPoseidonPreservation.outputValue
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment
        (PiRLCSamplerPoseidonPlan.invocation descriptor.source
          (PiRLCSamplerDirectSemantics.priorStep descriptor.round))
        (DigestWindow.rateLane descriptor.lane) := by
    rcases descriptor with ⟨source, ⟨round, bounded⟩, lane⟩
    cases round with
    | zero => exact PiRLCSamplerRetainedCustody.resolvedEnv_poseidonEntry geometry assignment source lane
    | succ previous =>
        exact PiRLCSamplerRetainedCustody.resolvedEnv_poseidonWindow
          geometry assignment source previous bounded lane
  rw [retained]
  exact (PriorStateHash.ofFn_getD _ (DigestWindow.rateLane descriptor.lane) 0).symm

/-- Every actual candidate is the verifier's low or high 16-bit digest chunk. -/
theorem rowsZero_implies_candidate_chunk
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (descriptor : Lane) (part : Fin 2) :
    ((DigestLane.candidate (PiRLCStarts.digestLaneLogicalStart descriptor.source.val
        descriptor.round.val descriptor.lane.val) part).eval
      (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment))).val =
      (Transcript.PiRlcSampler.digestChunks
        (ActualPiRLCStates.state (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
          assignment descriptor.source (PiRLCSamplerDirectSemantics.priorStep descriptor.round))
        (Fin.encodeProd (descriptor.lane, part))).val := by
  rw [DigestLane.candidateValue_eq _ _ _
    (rowsZero_implies_lane relation geometry assignment one rows descriptor), lane_eval]
  simp only [Transcript.PiRlcSampler.digestChunks, Fin.encodeProd, Fin.mkDivMod]
  have partLt := part.isLt
  rw [show (2 * descriptor.lane.val + part.val) / 2 = descriptor.lane.val by omega,
    show (2 * descriptor.lane.val + part.val) % 2 = part.val by omega]
  rfl

def verifierCandidate
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) : Chunk :=
  (sourceAt Transcript.PiRlcSampler.specification
    (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
      assignment) candidate.source.val).stream candidate.round.val

theorem verifierCandidate_eq_digest
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (descriptor : Lane) (part : Fin 2) :
    verifierCandidate geometry assignment (PiRLCSamplerCandidateWiring.candidate descriptor part) =
      Transcript.PiRlcSampler.digestChunks
        (ActualPiRLCStates.state (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
          assignment descriptor.source (PiRLCSamplerDirectSemantics.priorStep descriptor.round))
        (Fin.encodeProd (descriptor.lane, part)) := by
  rw [ActualPiRLCStates.state_eq_verifier _ _ semantics]
  change Transcript.PiRlcSampler.digestChunks
      (stateBeforeBlock Transcript.PiRlcSampler.machine
        (Transcript.PiRlcSampler.enterScalar
          (stateAt Transcript.PiRlcSampler.specification
            (ActualPiRLCStates.initialState (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)
              assignment) descriptor.source.val) descriptor.source.val) descriptor.source.val
        ((descriptor.round.val * 8 + descriptor.lane.val * 2 + part.val) / 8))
      ⟨(descriptor.round.val * 8 + descriptor.lane.val * 2 + part.val) % 8, _⟩ = _
  have laneLt : descriptor.lane.val < 4 := descriptor.lane.isLt
  have partLt := part.isLt
  rw [show (descriptor.round.val * 8 + descriptor.lane.val * 2 + part.val) / 8 =
    descriptor.round.val by omega]
  congr 1
  apply Fin.ext
  simp only [Fin.encodeProd, Fin.mkDivMod]
  omega

theorem rowsZero_implies_candidate_verifier
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (descriptor : Lane) (part : Fin 2) :
    ((DigestLane.candidate (PiRLCStarts.digestLaneLogicalStart descriptor.source.val
        descriptor.round.val descriptor.lane.val) part).eval
      (Spartan.pullback (PiRLCSamplerOrdinaryDirectPlan.resolvedEnv geometry assignment))).val =
      (verifierCandidate geometry assignment
        (PiRLCSamplerCandidateWiring.candidate descriptor part)).val := by
  rw [verifierCandidate_eq_digest geometry assignment semantics]
  exact rowsZero_implies_candidate_chunk relation geometry assignment one rows descriptor part

theorem rowsZero_implies_reject_boolean
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (descriptor : Lane) (part : Fin 2) :
    ((inputs geometry).reject (PiRLCSamplerCandidateWiring.candidate descriptor part)).eval
        assignment = 0 ∨
      ((inputs geometry).reject (PiRLCSamplerCandidateWiring.candidate descriptor part)).eval
        assignment = 1 := by
  have decoder := rowsZero_implies_decoder relation geometry assignment one rows descriptor part
  rw [← reject_eval]
  rw [decoder.reject_eq]
  split <;> simp

private theorem candidate_parts (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    ∃ descriptor : Lane, ∃ part : Fin 2,
      PiRLCSamplerCandidateWiring.candidate descriptor part = candidate := by
  rcases candidate with ⟨source, round⟩
  let outer : Fin 8 × Fin 8 := Fin.decodeProd round
  let inner : Fin 4 × Fin 2 := Fin.decodeProd outer.2
  refine ⟨⟨source, outer.1, inner.1⟩, inner.2, ?_⟩
  unfold PiRLCSamplerCandidateWiring.candidate
  apply congrArg (PiRLCFirst54DirectSchedule.Candidate.mk source)
  apply Fin.ext
  simp [outer, inner, Fin.decodeProd]
  omega

/-- Both fields read by First54 are determined by the exact verifier candidate. -/
theorem rowsZero_implies_verifierValues
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (candidate : PiRLCFirst54DirectSchedule.Candidate) :
    ((inputs geometry).reject candidate).eval assignment =
        (if (verifierCandidate geometry assignment candidate).val = rejectionBucket then 1 else 0) ∧
      ((inputs geometry).symbol candidate).eval assignment =
        Sampler.coefficientWord (verifier.symbol (verifierCandidate geometry assignment candidate)) := by
  obtain ⟨descriptor, part, rfl⟩ := candidate_parts candidate
  have decoder := rowsZero_implies_decoder relation geometry assignment one rows descriptor part
  have chunk := rowsZero_implies_candidate_verifier relation geometry assignment one rows
    semantics descriptor part
  constructor
  · exact (reject_eval geometry assignment descriptor part).symm.trans
      (decoder.reject_eq.trans (congrArg (fun value =>
        if value = rejectionBucket then (1 : F) else 0) chunk))
  · apply Fin.ext
    rw [← symbol_eval, decoder.remainder_eq, Sampler.coefficientWord_val]
    exact congrArg (fun value => value % 5) chunk

theorem rowsZero_implies_selector_verifier
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (semantics : PiRLCSamplerPoseidonPreservation.CanonicalSemantics
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry) assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) (round : Fin First54.candidateCount) :
    let selected := First54.semanticVerifier
      (ActualPiRLCSelector.interface (inputs geometry) assignment source) 0
      (ActualPiRLCSelector.decodedEnv (inputs geometry) assignment source)
    selected.accepts round = verifier.accepts (verifierCandidate geometry assignment ⟨source, round⟩) ∧
      selected.symbol round = Sampler.coefficientWord
        (verifier.symbol (verifierCandidate geometry assignment ⟨source, round⟩)) := by
  have values := rowsZero_implies_verifierValues relation geometry assignment one rows semantics
    ⟨source, round⟩
  dsimp only
  refine ⟨?_, values.2⟩
  rw [Bool.eq_iff_iff, accepts_eq_true_iff_ne_rejectionBucket]
  change decide ((1 - ((inputs geometry).reject ⟨source, round⟩).eval assignment : F) = 1) = true ↔ _
  rw [values.1]
  by_cases rejected : (verifierCandidate geometry assignment ⟨source, round⟩).val = rejectionBucket
  · simp [rejected, goldilocksModulus]
  · simp [rejected]

theorem rowsZero_implies_all_rejects
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (rows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    ∀ round : Fin First54.candidateCount,
      ((inputs geometry).reject ⟨source, round⟩).eval assignment = 0 ∨
        ((inputs geometry).reject ⟨source, round⟩).eval assignment = 1 := by
  intro round
  obtain ⟨descriptor, part, same⟩ := candidate_parts ⟨source, round⟩
  simpa only [same] using
    rowsZero_implies_reject_boolean relation geometry assignment one rows descriptor part

/-- Accepted decoder and selector rows force the actual retained output to
be the first 54 accepted symbols. Boolean flags are derived from the rows. -/
theorem rowsZero_implies_bounded_selector
    (relation : ProductionKey.LogicalRelation relationLogicalWidth relationPublicFits)
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (PiRLCSamplerOrdinaryRetainedGeometry.oneColumn geometry) = 1)
    (decoderRows : (PiRLCSamplerOrdinaryDirectPlan.plan relation geometry).RowsZero assignment)
    (selectorRows : (PiRLCFirst54DirectPlan.plan (inputs geometry)).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount) :
    Sampling.FirstAccepted.boundedSample
        (First54.semanticVerifier (ActualPiRLCSelector.interface (inputs geometry) assignment source)
          0 (ActualPiRLCSelector.decodedEnv (inputs geometry) assignment source))
        First54.outputCount (First54.semanticCandidates First54.candidateCount) =
      some (ActualPiRLCSelector.outputValues (inputs geometry) assignment source) := by
  exact ActualPiRLCSelector.rowsZero_and_rejects_imply_boundedSample (inputs geometry)
    assignment one selectorRows source
    (rowsZero_implies_all_rejects relation geometry assignment one decoderRows source)

/-- The selected Stage 1 rows and actual public input supply the decoder,
selector and one-cell premises for every source. -/
theorem selectedRowsAndPublic_imply_bounded_selectors
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let geometry := DirectApplicationPrefixPlan.prefixGeometry
      (PerApplicationFixedPoint.geometry application)
    ∀ source : Fin PiRLCFirst54DirectSchedule.sourceCount,
      Sampling.FirstAccepted.boundedSample
          (First54.semanticVerifier (ActualPiRLCSelector.interface (inputs geometry) assignment source)
            0 (ActualPiRLCSelector.decodedEnv (inputs geometry) assignment source))
          First54.outputCount (First54.semanticCandidates First54.candidateCount) =
        some (ActualPiRLCSelector.outputValues (inputs geometry) assignment source) := by
  let geometry := PerApplicationFixedPoint.geometry application
  let relation := PerApplicationFixedPoint.relation application fits
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one geometry assignment digest publicBound
  have selected : (DirectApplicationPrefixPlan.plan relation fits.package geometry).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have applicationRows := (DirectApplicationPrefixPlan.rowsZero_iff relation
    fits.package geometry assignment).mp selected
  have prefixRows := (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment).mp applicationRows.1.1.1
  have piRlcRows := prefixRows.2.2.1
  change (PiRLCRetainedPlan.plan _ _).RowsZero assignment at piRlcRows
  have selectorRows := (PiRLCRetainedPlan.rowsZero_iff _ _ assignment).mp piRlcRows
  dsimp only
  intro source
  exact rowsZero_implies_bounded_selector relation
    (DirectApplicationPrefixPlan.prefixGeometry geometry) assignment one prefixRows.2.1
    selectorRows.2 source

end NightstreamFPrime.Export.Stage1.ActualPiRLCCandidates

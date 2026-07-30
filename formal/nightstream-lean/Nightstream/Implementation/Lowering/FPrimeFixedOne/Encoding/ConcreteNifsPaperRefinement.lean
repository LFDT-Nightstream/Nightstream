import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition

/-!
Occurrence-bound paper refinement for the Lean-owned selected NIFS rows.

The physical recipe remains deterministic: its `CallRecipe.activeSoundness`
concludes the exact selected call result with no event disjunction.  This file
adds the separate security/refinement layer required by the paper relation.

The first failure is the concrete analogue of the paper NIFS coordinate-fork
event.  It is not a generic source-binding escape: it says that the exact
decoded occurrence has no authoritative Split-NC source family whose public
surfaces are bound and whose verifier-materialized `K + k` output batch has
complete corrected-ambient openings at the occurrence's derived point.

Once that opening exists, the remaining alternatives are the exact delayed
packed-`y_zcol` projection failure, exact child-opening extraction failure,
or the existing typed Split-NC FE/NC algebraic event.  Every constructor is
indexed by the decoded context and certificate.  `OccurrenceBoundEvent`
additionally binds it to the unique output decoded from the same satisfying
assignment.

No `SourceAuthority`, generic source-binding proposition, accepted result,
semantic conclusion, or event branch is supplied by the caller.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private def selectedContext
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    FixedActive.Context shape TranscriptState publicRingColumns publicFits
      verifierRows :=
  (ConcreteNifsParameters.context
    (keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
    running fresh proof).materialize

/-- A complete corrected-ambient source opening for this exact physical
occurrence.  The two public source surfaces are bound to one `Data` value and
every materialized PiCCS output has a corrected-ambient opening at the
verifier-derived row point. -/
structure CorrectedAmbientOpening
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (data : Data shape) : Prop where
  input : ConcretePhi81.SemanticInput context data
  outputs :
    Protocol.BlockLane.OutputRefinement.AmbientOutputHolds
      publicRingColumns publicFits (ConcretePhi81.commit context.key) data
      context.alignment context.input
      (ConcretePhi81.derive context certificate).piCcs.fePoint.row
      certificate.piCcs.output

/-- Exact coordinate-fork extraction failure for the selected occurrence.
Unlike a generic source-binding alternative, the negated witness includes
both authoritative public surfaces and the complete corrected-ambient output
batch at the derived point. -/
def PiRlcCoordinateForkExtractionFailure
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context) : Prop :=
  ¬ ∃ data : Data shape,
    CorrectedAmbientOpening context certificate data

/-- Exact child-opening extraction failure for the actual fourteen PiDEC
children computed by this occurrence. -/
def PiDecChildExtractionFailure
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (data : Data shape) : Prop :=
  ¬ ConcretePhi81.ChildOpenings context data certificate

/-- Exact failure of the delayed old-point packed-`y_zcol` projection for the
same extracted source data and the occurrence's derived block point. -/
def DelayedPackedProjectionFailure
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (data : Data shape) : Prop :=
  ¬ Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
      context.covers data
      (ConcretePhi81.derive context certificate).piCcs.ncPoint.block
      certificate.piCcs.output

/-- Closed selected-occurrence failure family.  Later constructors retain the
positive evidence from every earlier extraction stage, so an unrelated
failure cannot be attached to the occurrence. -/
inductive BadEvent
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context) : Prop where
  | piRlcCoordinateForkExtraction
      (failure :
        PiRlcCoordinateForkExtractionFailure context certificate)
  | delayedPackedProjection
      (data : Data shape)
      (ambient : CorrectedAmbientOpening context certificate data)
      (failure :
        DelayedPackedProjectionFailure context certificate data)
  | piDecChildExtraction
      (data : Data shape)
      (ambient : CorrectedAmbientOpening context certificate data)
      (packed :
        Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
          context.covers data
          (ConcretePhi81.derive context certificate).piCcs.ncPoint.block
          certificate.piCcs.output)
      (failure : PiDecChildExtractionFailure context certificate data)
  | piCcsAlgebraic
      (data : Data shape)
      (ambient : CorrectedAmbientOpening context certificate data)
      (packed :
        Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
          context.covers data
          (ConcretePhi81.derive context certificate).piCcs.ncPoint.block
          certificate.piCcs.output)
      (children : ConcretePhi81.ChildOpenings context data certificate)
      (failure : ConcretePhi81.PiCcsBadEvent context data certificate)

/-- Positive paper acceptance bound to the exact public output decoded from
the same satisfying occurrence. -/
structure PaperAcceptedAtOutput
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Prop where
  resultExact :
    SelectedRunning.ofResult
        (FixedActive.resultOf context certificate) =
      output
  transition :
    FixedActive.PaperProfile.Transition
      (FixedActive.paperProfileOf context) context.input
      (ConcretePhi81.outputChildren context certificate)

/-- A bad event is occurrence-bound only when it also carries the exact
decoded output equation established by the physical rows. -/
structure OccurrenceBoundEvent
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows) :
    Prop where
  resultExact :
    SelectedRunning.ofResult
        (FixedActive.resultOf context certificate) =
      output
  failure : BadEvent context certificate

/-- An accepted paper transition or named event cannot be attached to a
different public output.  This is the negative control for occurrence
binding: both branches carry the same row-derived result equation. -/
theorem no_outcome_at_wrong_output
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (different :
      SelectedRunning.ofResult
          (FixedActive.resultOf context certificate) ≠
        output) :
    ¬ (PaperAcceptedAtOutput context certificate output ∨
        OccurrenceBoundEvent context certificate output) := by
  intro outcome
  rcases outcome with accepted | event
  · exact different accepted.resultExact
  · exact different event.resultExact

private theorem yRing_bound_of_ambient
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (data : Data shape)
    (ambient : CorrectedAmbientOpening context certificate data) :
    certificate.piCcs.output.yRing =
      Polynomial.Fe.sourceYRingAt data
        (ConcretePhi81.derive context certificate).piCcs.fePoint.row := by
  exact
    Protocol.BlockLane.OutputRefinement.yRing_eq_sourceYRingAt_of_ambientOutputHolds
      publicRingColumns publicFits (ConcretePhi81.commit context.key) data
      context.alignment context.input
      (ConcretePhi81.derive context certificate).piCcs.fePoint.row
      certificate.piCcs.output ambient.input.sources ambient.outputs

private theorem paper_transition_of_refinement
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (data : Data shape)
    (refinement :
      ConcretePhi81.CertificateRefinement context data certificate) :
    FixedActive.PaperProfile.Transition
      (FixedActive.paperProfileOf context) context.input
      (ConcretePhi81.outputChildren context certificate) := by
  have resultTransition :
      FixedActive.ResultTransition context
        (FixedActive.resultOf context certificate) := by
    exact Result.resultOf_refines refinement
  rcases
      (FixedActive.resultTransition_iff_exists_paperDecomposition
        context (FixedActive.resultOf context certificate)).mp
        resultTransition with
    ⟨paperData, witness, decomposed⟩
  exact ⟨paperData, witness, by
    simpa using decomposed.paper⟩

/-- Deterministic physical acceptance plus exact output materialization
refines the unchanged fixed-active paper relation or one closed
occurrence-bound event. -/
theorem accepted_refinesPaper_or_boundEvent
    (prime : EuclidPrime goldilocksP)
    (context :
      FixedActive.Context shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context)
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (accepted : ConcretePhi81.Accepted context certificate)
    (resultExact :
      SelectedRunning.ofResult
          (FixedActive.resultOf context certificate) =
        output) :
    PaperAcceptedAtOutput context certificate output ∨
      OccurrenceBoundEvent context certificate output := by
  let noZeroDivisors : NormRange.BaseFieldNoZeroDivisors :=
    NormRange.baseFieldNoZeroDivisors_of_modulusEuclid prime
  by_cases extracted :
      ∃ data : Data shape,
        CorrectedAmbientOpening context certificate data
  · rcases extracted with ⟨data, ambient⟩
    by_cases packed :
        Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
          context.covers data
          (ConcretePhi81.derive context certificate).piCcs.ncPoint.block
          certificate.piCcs.output
    · by_cases children :
        ConcretePhi81.ChildOpenings context data certificate
      · have outputBound :
          ConcretePhi81.OutputBound context data certificate := by
          exact ⟨yRing_bound_of_ambient context certificate data ambient,
            packed⟩
        rcases
            ConcretePhi81.accepted_implies_refinement_or_outputUnbound_or_badEvent
              noZeroDivisors ambient.input children accepted with
          refinement | outputUnbound | bad
        · exact Or.inl {
            resultExact := resultExact
            transition :=
              paper_transition_of_refinement context certificate data
                refinement
          }
        · exact False.elim (outputUnbound outputBound)
        · exact Or.inr {
            resultExact := resultExact
            failure := .piCcsAlgebraic data ambient packed children bad
          }
      · exact Or.inr {
          resultExact := resultExact
          failure := .piDecChildExtraction data ambient packed children
        }
    · exact Or.inr {
        resultExact := resultExact
        failure := .delayedPackedProjection data ambient packed
      }
  · exact Or.inr {
      resultExact := resultExact
      failure := .piRlcCoordinateForkExtraction extracted
    }

/-- **Headline raw-row paper refinement.**  The complete raw selected-NIFS
rows determine their own output and then refine the frozen paper relation or
one exact occurrence-bound event.  No output decoder, verifier acceptance,
source witness, semantic conclusion, or event is a caller premise. -/
theorem rawRows_refinePaper_or_boundEvent
    (prime : EuclidPrime goldilocksP)
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantOne : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application profile frame)
        assignment) :
    ∃ output :
        SelectedRunning shape publicRingColumns publicFits verifierRows,
      frame.outputs.Decodes (FamilyFor application) assignment
          (.cons output .nil) ∧
        (PaperAcceptedAtOutput
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate output ∨
          OccurrenceBoundEvent
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate output) := by
  rcases
      ConcreteNifsRawSemantics.call_result_and_output_of_rawRows
        prime application profile frame assignment running fresh proof
        constantOne decoded satisfied with
    ⟨output, evaluated, decodedOutput⟩
  have exact :=
    (ConcreteNifsSelectedCallFrame.call_result_exact
      running fresh proof output).mp evaluated
  refine ⟨output, decodedOutput, ?_⟩
  exact accepted_refinesPaper_or_boundEvent
    prime (selectedContext (keys := keys) running fresh proof)
    proof.certificate output exact.1 exact.2

/-- **Headline activation-aware paper refinement.**  Satisfaction of the
actual selected `CallRecipe` rows at active one first projects to the raw
program, then returns the exact decoded output and the separate
paper-refinement/event result. -/
theorem selectedNifs_refinesPaper_or_boundEvent
    (application : Poseidon23ApplicationProfile Selected)
    (certificate :
      ConcreteNifsVerifyCallRecipe.Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      Satisfies
        (ConcreteNifsActivatedProgram.rows
          application certificate.operational frame)
        assignment) :
    ∃ output :
        SelectedRunning shape publicRingColumns publicFits verifierRows,
      frame.outputs.Decodes (FamilyFor application) assignment
          (.cons output .nil) ∧
        (PaperAcceptedAtOutput
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate output ∨
          OccurrenceBoundEvent
            (selectedContext (keys := keys) running fresh proof)
            proof.certificate output) := by
  have activatedRaw :
      RawSatisfies
        (ConcreteNifsActivatedProgram.rawRows
          application certificate.operational frame)
        assignment := by
    exact
      (satisfies_ownRows_iff frame.owner
        (ConcreteNifsActivatedProgram.rawRows
          application certificate.operational frame)
        assignment).mp
        (by
          simpa [ConcreteNifsActivatedProgram.rows] using satisfied)
  have rawSatisfied :=
    ActivatedRawProgram.active_sound frame.active
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
      assignment
      (ConcreteNifsActivatedProgram.residuals_length
        application certificate.operational certificate.footprint frame).symm
      activeOne activatedRaw
  exact rawRows_refinePaper_or_boundEvent
    certificate.prime application certificate.operational frame assignment
    running fresh proof constantOne decoded rawSatisfied

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement

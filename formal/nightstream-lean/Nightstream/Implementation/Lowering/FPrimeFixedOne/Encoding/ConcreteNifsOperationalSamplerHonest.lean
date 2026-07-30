import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerConservation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalTypedHonest
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerHonestRefinement

/-!
Contract: construct one typed honest assignment for the selected operational
ΠCCS plus fixed-active ΠRLC sampler prefix.

The sampler witness is derived from the accepted certificate's exact bounded
batch.  Its 15×54 selector outputs are then proved equal to the authoritative
proof-codec coordinates before the numeric witness is pulled back through the
typed call-frame map.

No candidate list, challenge vector, sampler-success bit, decoded transcript,
Rust row, or generated artifact is accepted as a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerHonest

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

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

private theorem operands_subset_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ column, column ∈ frame.operands.ids → column ∈ frame.visibleIds := by
  intro column member
  have contextMember : column ∈ frame.contextBundles.ids :=
    RefBundles.fromSchema_ids_subset _ _ column member
  simp [CallFrame.visibleIds, contextMember]

/-- The end of the complete physical sampler prefix lies in the exact typed
temporary namespace whenever the raw call frame fits. -/
theorem actionBase_le_orderedLength
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    ConcreteNifsRawProgram.actionBase application profile frame ≤
      (orderedIds frame).length := by
  rw [orderedIds_eq_visible_append_temporaries, List.length_append]
  change
    ConcreteNifsRawProgram.actionBase application profile frame ≤
      temporaryBase frame + frame.temporaries.ids.length
  have rawBound :
      ConcreteNifsRawProgram.allocationWidth application profile frame ≤
        frame.temporaries.ids.length := fits
  rw [ConcreteNifsAllocationCoverage.actionBase_eq_temporarySource
    application profile frame]
  unfold temporarySource
  have prefixBound :
      10 +
          (ConcreteNifsOperationalSampler.cost
            application profile frame).auxiliaryColumns ≤
        ConcreteNifsRawProgram.allocationWidth
          application profile frame := by
    unfold ConcreteNifsRawProgram.allocationWidth
    omega
  omega

/-- Every proof challenge coordinate read by the binding rows is in the
caller-owned prefix preceding the sampler allocation. -/
private theorem challengeLocation_before_samplerBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    (ConcreteNifsOperationalSampler.challengeLocation
        application profile frame coordinate position).numeric <
      ConcreteNifsOperationalSampler.samplerBase
        application profile frame := by
  have beforeVisible :
      (ConcreteNifsOperationalSampler.challengeLocation
        application profile frame coordinate position).numeric <
        temporaryBase frame := by
    simpa [ConcreteNifsOperationalSampler.challengeLocation,
      ConcreteNifsOperationalOccurrence.proofFieldLocation,
      ConcreteNifsCarrierFrame.proofFLocation] using
      ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.samplerViews.challenge coordinate position)
  exact Nat.lt_of_lt_of_le beforeVisible
    (ConcreteNifsOperationalTypedHonest.temporaryBase_le_samplerBase
      application profile frame)

/-- One fully accepted selected certificate has an honest typed assignment
for every operational ΠCCS, fixed-active sampler, and challenge-binding row. -/
theorem rows_honest
    (prime : EuclidPrime goldilocksP)
    (field : PiRlcCanonicalCandidateHonest.FieldInverse)
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) initial
        (.cons running (.cons fresh (.cons proof .nil))))
    (constantWire : initial frame.one = 1)
    (accepted :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Accepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate) :
    ∃ completed : ColumnId → Field,
      AgreesOn frame.visibleIds initial completed ∧
        ChangesOnly frame.temporaries.ids initial completed ∧
        frame.operands.Encodes (FamilyFor application) completed
            (.cons running (.cons fresh (.cons proof .nil))) ∧
          completed frame.one = 1 ∧
          Satisfies
            (ConcreteNifsOperationalSampler.rows application profile frame)
            (numericAssignment (columnMap frame) completed) := by
  let operational :=
    ConcreteNifsOperationalTypedHonest.assignment
      application profile frame initial running fresh proof
  let initialNumeric := numericAssignment (columnMap frame) operational
  let base := ConcreteNifsOperationalSampler.samplerBase
    application profile frame
  let lanes :=
    ConcreteNifsOperationalSampler.samplerLanes application profile frame
  have operationalEncodes :
      frame.operands.Encodes (FamilyFor application) operational
        (.cons running (.cons fresh (.cons proof .nil))) :=
    ConcreteNifsOperationalTypedHonest.assignment_encodes
      application profile frame initial running fresh proof fits encoded
  have operationalDecodes :
      frame.operands.Decodes (FamilyFor application) operational
        (.cons running (.cons fresh (.cons proof .nil))) :=
    frame.operands.decodes_of_encodes
      (FamilyFor application) operational
      (.cons running (.cons fresh (.cons proof .nil))) operationalEncodes
  have proofDecodes :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame operational running fresh proof
      operationalDecodes
  have operationalWire : operational frame.one = 1 :=
    ConcreteNifsOperationalTypedHonest.assignment_constantWire
      application profile frame initial running fresh proof fits constantWire
  have numericWire : initialNumeric 0 = 1 :=
    ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
      application frame operational operationalWire
  have initialCanonical : ∀ column, initialNumeric column < goldilocksP :=
    numericAssignment_canonical (columnMap frame) operational
  have basePositive : 0 < base := by
    simpa [base] using
      ConcreteNifsOperationalSamplerConservation.samplerBase_positive
        application profile frame
  have lanesInPrefix :
      ∀ lane : Fin Poseidon2Core.width,
        SymbolicDuplexPlacement.ValueInPrefix base (lanes lane) := by
    simpa [base, lanes] using
      ConcreteNifsOperationalSamplerConservation.samplerLanes_inPrefix
        application profile frame
  have selectedInitial :=
    ConcreteNifsOperationalTypedHonest.decodedSamplerInitial_eq
      application profile frame initial running fresh proof fits encoded
      constantWire accepted.piCcs
  rcases accepted.sampler with ⟨selectedBound⟩
  have samplerBound :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Bound
        (PiRlcCanonicalMachine.machine profile.constants)
        (SymbolicDuplexSemantics.decodedBuilder initialNumeric
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
        proof.certificate.piRlcChallenges := by
    rw [← profile.selectedSamplerMachine, selectedInitial]
    exact selectedBound
  let enough :=
    PiRlcCanonicalSamplerHonestRefinement.honestEnough_of_bound
      prime field base profile.constants lanes initialNumeric basePositive
      lanesInPrefix initialCanonical numericWire samplerBound
  let samplerNumeric :=
    PiRlcCanonicalSamplerProgram.honestAssignment field base
      profile.constants lanes initialNumeric enough
  have samplerCanonical : ∀ column, samplerNumeric column < goldilocksP :=
    PiRlcCanonicalSamplerProgram.honestAssignment_canonical field base
      profile.constants lanes initialNumeric initialCanonical enough
  have samplerWire : samplerNumeric 0 = 1 :=
    PiRlcCanonicalSamplerProgram.honestAssignment_constantWire field base
      profile.constants lanes initialNumeric enough basePositive numericWire
  have samplerSatisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.samplerRows
          application profile frame)
        samplerNumeric := by
    simpa [ConcreteNifsOperationalSampler.samplerRows, base, lanes,
      samplerNumeric] using
      PiRlcCanonicalSamplerProgram.honestAssignment_satisfies field base
        profile.constants lanes initialNumeric basePositive lanesInPrefix
        initialCanonical numericWire enough
  have challengeSatisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.challengeRows
          application profile frame)
        samplerNumeric := by
    intro row member
    unfold ConcreteNifsOperationalSampler.challengeRows at member
    rcases List.mem_flatten.1 member with
      ⟨group, groupMember, rowMember⟩
    rcases List.mem_ofFn.1 groupMember with ⟨coordinate, rfl⟩
    rcases List.mem_ofFn.1 rowMember with ⟨position, rfl⟩
    let outputColumn :=
      PiRlcCanonicalSelector.outputColumn
        (PiRlcCanonicalSamplerProgram.selectorBase base)
        (ConcreteNifsOperationalSampler.samplerCoordinate coordinate)
        (ConcreteNifsOperationalSampler.samplerPosition position)
    let location :=
      ConcreteNifsOperationalSampler.challengeLocation
        application profile frame coordinate position
    apply
      (KEquality.equalityRow_iff samplerNumeric
        [(outputColumn, 1)] location.carried samplerWire).2
    rw [KMul.lcEval_singleton_col, Nat.mod_eq_of_lt (samplerCanonical _)]
    change samplerNumeric outputColumn =
      lcEval samplerNumeric location.carried
    rw [show location.carried = [(location.numeric, 1)] by rfl,
      KMul.lcEval_singleton_col,
      Nat.mod_eq_of_lt (samplerCanonical _)]
    have outputBound :=
      PiRlcCanonicalSamplerHonestRefinement.honestAssignment_output_eq_bound
        prime field base profile.constants lanes initialNumeric basePositive
        lanesInPrefix initialCanonical numericWire samplerBound
        (ConcreteNifsOperationalSampler.samplerCoordinate coordinate)
        (ConcreteNifsOperationalSampler.samplerPosition position)
    have coordinateEqual :
        ConcreteNifsOperationalSampler.samplerCoordinate coordinate =
          coordinate := by
      apply Fin.ext
      rfl
    have positionEqual :
        PiRlcCanonicalSamplerCheckerRefinement.outputRingPosition
            (ConcreteNifsOperationalSampler.samplerPosition position) =
          position := by
      apply Fin.ext
      rfl
    rw [coordinateEqual, positionEqual] at outputBound
    have decodedValue :
        (profile.samplerViews.challenge coordinate position
            |>.column (proofOperand frame.operands)
              (proof_widthsAgree frame)).value operational =
          proof.certificate.piRlcChallenges coordinate position :=
      (profile.samplerViews.challenge coordinate position
        |>.value_eq_of_bundle_decodes
          (FamilyFor application) (.data .nifsProof)
          (proofOperand frame.operands) (proof_widthsAgree frame)
          operational proof proofDecodes)
    have locationField :
        residue (initialNumeric location.numeric) =
          proof.certificate.piRlcChallenges coordinate position := by
      exact (location.numeric_value_eq operational).trans decodedValue
    have locationValue :
        initialNumeric location.numeric =
          (proof.certificate.piRlcChallenges coordinate position).val := by
      apply residue_injective_of_lt
        (initialCanonical location.numeric)
      · simpa [Numeric.modulus, goldilocksP, goldilocksModulus] using
          (proof.certificate.piRlcChallenges coordinate position).isLt
      · exact locationField.trans
          (residue_field_val
            (proof.certificate.piRlcChallenges coordinate position)).symm
    have locationPreserved :
        samplerNumeric location.numeric = initialNumeric location.numeric := by
      have locationBefore :=
        challengeLocation_before_samplerBase
          application profile frame coordinate position
      exact PiRlcCanonicalSamplerProgram.honestAssignment_before_base field
        base profile.constants lanes initialNumeric enough
        (by simpa [base, location] using locationBefore)
    have outputValue :
        samplerNumeric outputColumn =
          (proof.certificate.piRlcChallenges coordinate position).val := by
      simpa [samplerNumeric, outputColumn] using outputBound
    exact outputValue.trans (locationPreserved.trans locationValue).symm
  have operationalSatisfiedInitial :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        initialNumeric :=
    ConcreteNifsOperationalTypedHonest.assignment_satisfies
      application profile frame initial running fresh proof fits encoded
      constantWire accepted.piCcs
  have operationalSatisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        samplerNumeric := by
    apply KHornerSupport.satisfies_extend _
      initialNumeric samplerNumeric
    · intro row member column mentioned
      have below :=
        ConcreteNifsEndpointConservation.operationalRows_below_afterAllocation
          application profile frame row member column mentioned
      exact
        (PiRlcCanonicalSamplerProgram.honestAssignment_before_base field
          base profile.constants lanes initialNumeric enough
          (by simpa [base, ConcreteNifsOperationalSampler.samplerBase]
            using below)).symm
    · exact operationalSatisfiedInitial
  have prefixSatisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        samplerNumeric := by
    intro row member
    unfold ConcreteNifsOperationalSampler.rows at member
    rcases List.mem_append.1 member with inPrefix | inChallenge
    · rcases List.mem_append.1 inPrefix with inOperational | inSampler
      · exact operationalSatisfied row inOperational
      · exact samplerSatisfied row inSampler
    · exact challengeSatisfied row inChallenge
  let completed :=
    ConcreteNifsNumericCompletion.complete frame operational samplerNumeric
  have completedEncodes :
      frame.operands.Encodes (FamilyFor application) completed
        (.cons running (.cons fresh (.cons proof .nil))) := by
    apply RefBundles.encodes_of_agrees
      (FamilyFor application) operational completed frame.operands
      (.cons running (.cons fresh (.cons proof .nil)))
    · exact agreesOn_of_subset (operands_subset_visible application frame)
        (ConcreteNifsNumericCompletion.complete_agrees_visible
          frame operational samplerNumeric)
    · exact operationalEncodes
  have completedWire : completed frame.one = 1 := by
    exact
      (ConcreteNifsNumericCompletion.complete_agrees_visible
        frame operational samplerNumeric frame.one
        (by simp [CallFrame.visibleIds])).trans operationalWire
  have temporaryToBase : temporaryBase frame ≤ base := by
    simpa [base] using
      ConcreteNifsOperationalTypedHonest.temporaryBase_le_samplerBase
        application profile frame
  have finalNumericEqual :
      ∀ source, source < (orderedIds frame).length →
        numericAssignment (columnMap frame) completed source =
          samplerNumeric source := by
    intro source sourceBound
    exact ConcreteNifsNumericCompletion.numericAssignment_complete_of_lt
      frame operational samplerNumeric samplerCanonical
      (fun visibleSource before =>
        PiRlcCanonicalSamplerProgram.honestAssignment_before_base field
          base profile.constants lanes initialNumeric enough
          (Nat.lt_of_lt_of_le before temporaryToBase))
      source sourceBound
  have completedSatisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        (numericAssignment (columnMap frame) completed) := by
    apply KHornerSupport.satisfies_extend _
      samplerNumeric
      (numericAssignment (columnMap frame) completed)
    · intro row member column mentioned
      have below :=
        ConcreteNifsOperationalSamplerConservation.rows_below_actionBase
          application profile frame row member column mentioned
      exact (finalNumericEqual column
        (Nat.lt_of_lt_of_le below
          (actionBase_le_orderedLength application profile frame fits))).symm
    · exact prefixSatisfied
  have completedAgrees :
      AgreesOn frame.visibleIds initial completed := by
    apply agreesOn_trans
    · simpa [operational] using
        (ConcreteNifsOperationalTypedHonest.assignment_agrees_visible
          application profile frame initial running fresh proof fits)
    · exact ConcreteNifsNumericCompletion.complete_agrees_visible
        frame operational samplerNumeric
  have completedChanges :
      ChangesOnly frame.temporaries.ids initial completed := by
    intro column notTemporary
    change
      ConcreteNifsNumericCompletion.complete frame operational samplerNumeric
          column =
        initial column
    rw [ConcreteNifsNumericCompletion.complete_changesOnly
      frame operational samplerNumeric column notTemporary]
    simpa [operational] using
      (ConcreteNifsOperationalTypedHonest.assignment_changesOnly
        application profile frame initial running fresh proof fits
        column notTemporary)
  exact
    ⟨completed, completedAgrees, completedChanges, completedEncodes,
      completedWire, completedSatisfied⟩

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerHonest

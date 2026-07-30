import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSampler
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: semantic refinement of the selected operational ΠRLC sampler rows.

The physical sampler starts from the ΠCCS output builder, computes every
fixed-active challenge independently, and binds all 15×54 coordinates to the
authoritative decoded NIFS proof.  No challenge, transcript state, or sampler
acceptance is supplied as a premise.

This module does not own ΠDEC, activation, output materialization, or the
complete `nifsVerify` `CallRecipe`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerSelected

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrenceSemantics
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSampler
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

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private theorem duplexState_eq
    (left right : Poseidon2Duplex.State)
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) :
    left = right := by
  cases left
  cases right
  simp only at lanes absorbed
  cases lanes
  cases absorbed
  rfl

/-- The sampler's empty local builder preserves exactly the ΠCCS output
lanes and its protocol-fixed cursor one.  Honest sampler construction uses
this bridge to connect the typed call-frame witness to the selected checker
state without accepting either state as a premise. -/
theorem decodedSamplerInitial_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : Nat → Nat) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder
          (samplerLanes application profile frame)) =
      SymbolicDuplexSemantics.decodedBuilder assignment
        (KSplitNcTranscript.outputBuilder
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)) := by
  apply duplexState_eq
  · rfl
  · change
      1 =
        (KSplitNcTranscript.outputBuilder
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).absorbed
    unfold KSplitNcTranscript.outputBuilder
      KSplitNcTranscript.absorbTagged
    rw [SymbolicDuplexCursor.absorbMany_absorbed]
    have beforeZero :
        (KSplitNcTranscript.ncLaneReplay
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)).builder.absorbed = 0 := by
      rfl
    rw [beforeZero]
    have payloadLength :
        (KSplitNcTranscript.taggedFields .output
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame).outputFields).length =
          2 + profile.outputSources.length := by
      simp [KSplitNcTranscript.taggedFields,
        ConcreteNifsOperationalOccurrence.transcriptInput]
      omega
    rw [payloadLength, profile.outputCursorOne]

/-- Every challenge-binding row equates one independently computed selector
coordinate with the matching decoded proof-codec coordinate. -/
theorem challengeCoordinate_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      Satisfies
        (challengeRows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    residue
        (numericAssignment (columnMap frame) assignment
          (PiRlcCanonicalSelector.outputColumn
            (PiRlcCanonicalSamplerProgram.selectorBase
              (samplerBase application profile frame))
            (samplerCoordinate coordinate) (samplerPosition position))) =
      proof.certificate.piRlcChallenges coordinate position := by
  let numeric := numericAssignment (columnMap frame) assignment
  let outputColumn :=
    PiRlcCanonicalSelector.outputColumn
      (PiRlcCanonicalSamplerProgram.selectorBase
        (samplerBase application profile frame))
      (samplerCoordinate coordinate) (samplerPosition position)
  let location :=
    challengeLocation application profile frame coordinate position
  have rowMember :
      challengeRow application profile frame coordinate position ∈
        challengeRows application profile frame := by
    apply List.mem_flatten.mpr
    refine
      ⟨List.ofFn fun index =>
          challengeRow application profile frame coordinate index,
        List.mem_ofFn.mpr ⟨coordinate, rfl⟩, ?_⟩
    exact List.mem_ofFn.mpr ⟨position, rfl⟩
  have equal :
      Nightstream.Implementation.R1CS.lcEval numeric [(outputColumn, 1)] =
        Nightstream.Implementation.R1CS.lcEval numeric location.carried := by
    exact
      (KEquality.equalityRow_iff numeric _ _
        (numericConstantWire application frame assignment constantWire)).1
        (satisfied _ rowMember)
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have carried :
      residue
          (Nightstream.Implementation.R1CS.lcEval numeric
            location.carried) =
        (profile.samplerViews.challenge coordinate position
          |>.column (proofOperand frame.operands)
            (proof_widthsAgree frame)).value assignment := by
    exact location.carried_value_eq assignment
  have decodedValue :
      (profile.samplerViews.challenge coordinate position
          |>.column (proofOperand frame.operands)
            (proof_widthsAgree frame)).value assignment =
        proof.certificate.piRlcChallenges coordinate position := by
    exact
      (profile.samplerViews.challenge coordinate position
        |>.value_eq_of_bundle_decodes
          (FamilyFor application) (.data .nifsProof)
          (proofOperand frame.operands) (proof_widthsAgree frame)
          assignment proof proofDecoded)
  change residue (numeric outputColumn) =
    proof.certificate.piRlcChallenges coordinate position
  calc
    residue (numeric outputColumn) =
        residue
          (Nightstream.Implementation.R1CS.lcEval numeric
            [(outputColumn, 1)]) := by
      rw [KMul.lcEval_singleton_col]
      exact (residue_mod (numeric outputColumn)).symm
    _ = residue
          (Nightstream.Implementation.R1CS.lcEval numeric
            location.carried) :=
      congrArg residue equal
    _ = _ := carried.trans decodedValue

/-- The complete physical sampler and binding rows compute exactly one
carried challenge of the selected proof. -/
theorem semanticChallenge_eq_proof
    (prime : EuclidPrime goldilocksP)
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (samplerSatisfied :
      Satisfies
        (samplerRows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (challengeSatisfied :
      Satisfies
        (challengeRows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total) :
    PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge
        prime
        (samplerBase application profile frame)
        (PiRlcCanonicalSamplerProgram.u64Base
          (samplerBase application profile frame))
        (PiRlcCanonicalSamplerProgram.candidateBase
          (samplerBase application profile frame))
        PiRlcCanonicalSamplerProgram.coordinateCount
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder
          (samplerLanes application profile frame))
        (numericAssignment_canonical (columnMap frame) assignment)
        (numericConstantWire application frame assignment constantWire)
        (PiRlcCanonicalSamplerProgram.u64Rows_satisfied
          (samplerBase application profile frame) profile.constants
          (samplerLanes application profile frame)
          (numericAssignment (columnMap frame) assignment)
          (by simpa [samplerRows] using samplerSatisfied))
        (samplerCoordinate coordinate) =
      proof.certificate.piRlcChallenges coordinate := by
  let base := samplerBase application profile frame
  let lanes := samplerLanes application profile frame
  let numeric := numericAssignment (columnMap frame) assignment
  have programSatisfied :
      Satisfies
        (PiRlcCanonicalSamplerProgram.rows base profile.constants lanes)
        numeric := by
    simpa [samplerRows, base, lanes, numeric] using samplerSatisfied
  have u64Satisfied :=
    PiRlcCanonicalSamplerProgram.u64Rows_satisfied
      base profile.constants lanes numeric programSatisfied
  have candidateSatisfied :=
    PiRlcCanonicalSamplerProgram.candidateRows_satisfied
      base profile.constants lanes numeric programSatisfied
  have selectorSatisfied :=
    PiRlcCanonicalSamplerProgram.selectorRows_satisfied
      base profile.constants lanes numeric programSatisfied
  have canonical : ∀ column, numeric column < goldilocksP :=
    numericAssignment_canonical (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
    numericConstantWire application frame assignment constantWire
  funext position
  apply Fin.ext
  have physical :=
    PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge_coordinate_eq_outputColumn
      prime base
      (PiRlcCanonicalSamplerProgram.u64Base base)
      (PiRlcCanonicalSamplerProgram.candidateBase base)
      (PiRlcCanonicalSamplerProgram.selectorBase base)
      PiRlcCanonicalSamplerProgram.coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      canonical wire u64Satisfied candidateSatisfied selectorSatisfied
      (samplerCoordinate coordinate) (samplerPosition position)
  have samePosition :
      PiRlcCanonicalSamplerCheckerRefinement.outputRingPosition
          (samplerPosition position) =
        position := by
    apply Fin.ext
    rfl
  rw [samePosition] at physical
  have bound :=
    challengeCoordinate_eq application profile frame assignment
      running fresh proof constantWire decoded challengeSatisfied
      coordinate position
  have boundValue :
      numeric
          (PiRlcCanonicalSelector.outputColumn
            (PiRlcCanonicalSamplerProgram.selectorBase base)
            (samplerCoordinate coordinate) (samplerPosition position)) =
        (proof.certificate.piRlcChallenges coordinate position).val := by
    apply residue_injective_of_lt
      (canonical
        (PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase base)
          (samplerCoordinate coordinate) (samplerPosition position)))
    · simpa [Numeric.modulus, goldilocksP, goldilocksModulus] using
        (proof.certificate.piRlcChallenges coordinate position).isLt
    · exact (by
        simpa [base, numeric] using
          bound.trans
            (residue_field_val
              (proof.certificate.piRlcChallenges coordinate position)).symm)
  rw [physical]
  exact boundValue

/-- One complete physical sampler occurrence makes the selected executable
checker return the corresponding challenge carried by the decoded proof. -/
theorem selectedSampleChallenge_eq_some
    (prime : EuclidPrime goldilocksP)
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        (numericAssignment (columnMap frame) assignment))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total) :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piRlcMachine)
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piRlcInitialState
        coordinate.val =
      some (proof.certificate.piRlcChallenges coordinate) := by
  let base := samplerBase application profile frame
  let lanes := samplerLanes application profile frame
  let numeric := numericAssignment (columnMap frame) assignment
  have piCcsSatisfied :=
    ConcreteNifsOperationalSampler.piCcsRows_satisfied
      application profile frame numeric satisfied
  have samplerSatisfied :=
    ConcreteNifsOperationalSampler.samplerRows_satisfied
      application profile frame numeric satisfied
  have challengeSatisfied :=
    ConcreteNifsOperationalSampler.challengeRows_satisfied
      application profile frame numeric satisfied
  have programSatisfied :
      Satisfies
        (PiRlcCanonicalSamplerProgram.rows base profile.constants lanes)
        numeric := by
    simpa [samplerRows, base, lanes, numeric] using samplerSatisfied
  have u64Satisfied :=
    PiRlcCanonicalSamplerProgram.u64Rows_satisfied
      base profile.constants lanes numeric programSatisfied
  have candidateSatisfied :=
    PiRlcCanonicalSamplerProgram.candidateRows_satisfied
      base profile.constants lanes numeric programSatisfied
  have selectorSatisfied :=
    PiRlcCanonicalSamplerProgram.selectorRows_satisfied
      base profile.constants lanes numeric programSatisfied
  have transcriptSatisfied :=
    PiRlcCanonicalSamplerProgram.transcriptRows_satisfied
      base profile.constants lanes numeric programSatisfied
  have canonical : ∀ column, numeric column < goldilocksP :=
    numericAssignment_canonical (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
    numericConstantWire application frame assignment constantWire
  have validFixed :
      SymbolicDuplexSemantics.Valid base profile.constants numeric
        (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder base lanes) :=
    SymbolicDuplexSemantics.valid_of_satisfied
      base profile.constants
      (PiRlcCanonicalSymbolicMachineHonest.fixedBuilder base lanes)
      numeric canonical wire transcriptSatisfied
  have validBatch :
      SymbolicDuplexSemantics.Valid base profile.constants numeric
        (PiRlcCanonicalSymbolicMachine.stateAt base
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
          PiRlcCanonicalSamplerProgram.coordinateCount) := by
    simpa [PiRlcCanonicalSymbolicMachineHonest.fixedBuilder,
      PiRlcCanonicalSamplerProgram.coordinateCount] using validFixed
  have initialEqual :
      SymbolicDuplexSemantics.decodedBuilder numeric
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes) =
        (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piRlcInitialState := by
    exact
      (decodedSamplerInitial_eq application profile frame numeric).trans
        (ConcreteNifsOperationalSelected.selectedPiRlcInitialState_eq
          application profile frame assignment running fresh proof
          constantWire decoded piCcsSatisfied)
  have sampled :=
    PiRlcCanonicalSamplerCheckerRefinement.sampleChallenge?_eq_some_semanticChallenge
      prime base
      (PiRlcCanonicalSamplerProgram.u64Base base)
      (PiRlcCanonicalSamplerProgram.candidateBase base)
      (PiRlcCanonicalSamplerProgram.selectorBase base)
      PiRlcCanonicalSamplerProgram.coordinateCount profile.constants
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      canonical wire u64Satisfied candidateSatisfied selectorSatisfied
      validBatch (samplerCoordinate coordinate)
  have challengeEqual :=
    semanticChallenge_eq_proof prime application profile frame assignment
      running fresh proof constantWire decoded samplerSatisfied
      challengeSatisfied coordinate
  calc
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.piRlcMachine)
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate).piRlcInitialState
          coordinate.val =
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.sampleChallenge?
          (PiRlcCanonicalMachine.machine profile.constants)
          (SymbolicDuplexSemantics.decodedBuilder numeric
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes))
          (samplerCoordinate coordinate).val := by
      rw [profile.selectedSamplerMachine, initialEqual]
      rfl
    _ = some
          (PiRlcCanonicalSamplerCheckerRefinement.semanticChallenge
            prime base
            (PiRlcCanonicalSamplerProgram.u64Base base)
            (PiRlcCanonicalSamplerProgram.candidateBase base)
            PiRlcCanonicalSamplerProgram.coordinateCount
            (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
            canonical wire u64Satisfied
            (samplerCoordinate coordinate)) := sampled
    _ = some (proof.certificate.piRlcChallenges coordinate) :=
      congrArg some challengeEqual

/-- Row satisfaction of the exact ΠCCS-plus-sampler prefix establishes the
selected checker's complete ΠRLC sampler acceptance predicate. -/
theorem selectedSamplerAccepted_of_rows
    (prime : EuclidPrime goldilocksP)
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.CertificateAccepted
      (ConcreteNifsParameters.context
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        running fresh proof).materialize
      proof.certificate := by
  apply
    (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.certificateCheck_eq_true_iff_accepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate).1
  unfold
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.certificateCheck
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.check
  apply List.all_eq_true.mpr
  intro coordinate _member
  apply
    (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Sampler.Checker.challengeMatches_eq_true_iff
      _ _).2
  exact
    selectedSampleChallenge_eq_some prime application profile frame assignment
      running fresh proof constantWire decoded satisfied coordinate

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerSelected

import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputDecode
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerSelected
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthoritySemantics

/-!
Contract: deterministic semantics of the complete ungated selected-NIFS row
program.

Owns: exact satisfaction transport for numeric slices; decomposition of the
six ordered raw slices; and composition of those slices into the unchanged
fixed-active `ConcretePhi81.Accepted` relation and its unique selected result.

Does not own: activation, output-codec existence, honest temporary completion,
the final `CallRecipe`, paper bad events, Rust, or generated artifacts.

No accepted predicate, output equality, or source-authority proposition is a
premise below. The only decoded output premise is a representation fact; a
later module must derive it from the physical output coordinates.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawSemantics

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
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

/-- Translating a numeric row list through the selected call-frame column map
preserves and reflects whole-list satisfaction. -/
theorem translate_satisfies_iff
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : List Nightstream.Implementation.R1CS.Row)
    (assignment : ColumnId → Field) :
    RawSatisfies
        (ConcreteNifsRawProgram.translate application frame source)
        assignment ↔
      Nightstream.Implementation.R1CS.Satisfies source
        (NumericRowBridge.numericAssignment (columnMap frame) assignment) := by
  induction source with
  | nil =>
      simp [ConcreteNifsRawProgram.translate,
        Nightstream.Implementation.R1CS.Satisfies]
  | cons row tail inductionHypothesis =>
      change
        ((NumericRowBridge.row (columnMap frame) row).Holds assignment ∧
            RawSatisfies
              (ConcreteNifsRawProgram.translate application frame tail)
              assignment) ↔
          Nightstream.Implementation.R1CS.Satisfies (row :: tail)
            (NumericRowBridge.numericAssignment
              (columnMap frame) assignment)
      rw [NumericRowBridge.row_holds_iff, inductionHypothesis]
      constructor
      · rintro ⟨head, rest⟩ candidate member
        rcases List.mem_cons.1 member with equal | inTail
        · simpa [equal] using head
        · exact rest candidate inTail
      · intro satisfied
        constructor
        · exact satisfied row (List.mem_cons_self)
        · intro candidate inTail
          exact satisfied candidate (List.mem_cons_of_mem row inTail)

/-- Satisfaction of the complete operational prefix implies satisfaction of
its leading ΠCCS occurrence without inspecting any profiler or owner label. -/
theorem operationalOccurrence_satisfied
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
    (satisfied :
      Nightstream.Implementation.R1CS.Satisfies
        (ConcreteNifsOperationalSampler.rows application profile frame)
        (NumericRowBridge.numericAssignment (columnMap frame) assignment)) :
    Nightstream.Implementation.R1CS.Satisfies
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      (NumericRowBridge.numericAssignment (columnMap frame) assignment) := by
  intro row member
  exact satisfied row
    (List.mem_append_left _
      (List.mem_append_left _ member))

/-- **Headline raw selected-NIFS refinement.** Satisfaction of all six
Lean-owned raw slices, together with the exact call-frame decoders, implies
the unchanged fixed-active checker relation and the unique selected output.

The source-structure member of `TailAccepted` is constructed from the
canonical context itself. It is not inferred from a structural row range and
is not supplied by the caller. -/
theorem accepted_and_output_of_rawRows
    (prime :
      Nightstream.Implementation.R1CS.EuclidPrime
        Nightstream.Implementation.R1CS.goldilocksP)
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
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application profile frame)
        assignment) :
    ConcretePhi81.Accepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate ∧
      SelectedRunning.ofResult
          (FixedActive.resultOf
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate) =
        output := by
  unfold ConcreteNifsRawProgram.rawRows at satisfied
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp satisfied with
    ⟨prefixFive, outputRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixFive with
    ⟨prefixFour, piDecRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixFour with
    ⟨prefixThree, actionSatisfied⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixThree with
    ⟨prefixTwo, pointRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixTwo with
    ⟨prefixOne, operationalRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixOne with
    ⟨_canonicalRaw, runningRaw⟩
  have runningSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsRunningAuthorityRows.rows application profile frame)
      assignment).mp runningRaw
  have operationalSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsOperationalSampler.rows application profile frame)
      assignment).mp operationalRaw
  have pointSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsPiRlcPointRows.rows application profile frame)
      assignment).mp pointRaw
  have piDecSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsPiDecRows.rows application profile frame)
      assignment).mp piDecRaw
  have outputSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsOutputRows.rows application profile frame)
      assignment).mp outputRaw
  have occurrenceSatisfied :=
    operationalOccurrence_satisfied application profile frame assignment
      operationalSatisfied
  have runningEquations :=
    ConcreteNifsRunningAuthoritySemantics.equations_of_rows
      application profile frame assignment running fresh proof constantWire
      decodedInputs runningSatisfied
  have runningAccepted :
      ConcretePhi81.RunningAuthority.Accepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize :=
    FixedActive.Canonical.RunningAuthority.accepted_of_equations
      runningEquations
  have piCcsAccepted :=
    ConcreteNifsOperationalSelected.selectedPiCcsAccepted_of_rows
      application profile frame assignment running fresh proof constantWire
      decodedInputs occurrenceSatisfied
  have samplerAccepted :=
    ConcreteNifsOperationalSamplerSelected.selectedSamplerAccepted_of_rows
      prime application profile frame assignment running fresh proof
      constantWire decodedInputs operationalSatisfied
  have piDecAccepted :=
    ConcreteNifsPiDecSemantics.recomposition_of_rows
      application profile frame assignment running fresh proof output
      (ConcreteNifsRawProgram.actionBase application profile frame)
      constantWire decodedInputs decodedOutput occurrenceSatisfied
      actionSatisfied pointSatisfied piDecSatisfied
  have outputExact :=
    ConcreteNifsOutputSemantics.output_eq_selectedResult_of_rows
      application profile frame assignment running fresh proof output
      (ConcreteNifsRawProgram.actionBase application profile frame)
      constantWire decodedInputs decodedOutput occurrenceSatisfied
      actionSatisfied pointSatisfied outputSatisfied
  refine ⟨{
    running := runningAccepted
    piCcs := piCcsAccepted
    sampler := samplerAccepted
    tail := {
      sourceStructures :=
        FixedActive.Canonical.Context.sourceStructures
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof)
      piDecRecomposition := piDecAccepted
    }
  }, outputExact⟩

/-- The raw rows reach the exact deterministic `nifsVerify` call result once
the physical output bundle is decoded. -/
theorem call_result_of_rawRows
    (prime :
      Nightstream.Implementation.R1CS.EuclidPrime
        Nightstream.Implementation.R1CS.goldilocksP)
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
    (output :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil))
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application profile frame)
        assignment) :
    callEval Selected Call.nifsVerify
        (.cons running (.cons fresh (.cons proof .nil))) =
      some (.cons output .nil) := by
  exact
    (ConcreteNifsSelectedCallFrame.call_result_exact
      running fresh proof output).2
      (accepted_and_output_of_rawRows prime application profile frame
        assignment running fresh proof output constantWire decodedInputs
        decodedOutput satisfied)

/-- The complete raw program itself determines the selected output codec.
Unlike `accepted_and_output_of_rawRows`, this theorem accepts no semantic
output and no output-decoder proposition from its caller. -/
theorem output_decodes_selectedResult_of_rawRows
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
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application profile frame)
        assignment) :
    frame.outputs.Decodes (FamilyFor application) assignment
      (.cons
        (SelectedRunning.ofResult
          (FixedActive.resultOf
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              running fresh proof).materialize
            proof.certificate))
        .nil) := by
  unfold ConcreteNifsRawProgram.rawRows at satisfied
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp satisfied with
    ⟨prefixFive, outputRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixFive with
    ⟨prefixFour, _piDecRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixFour with
    ⟨prefixThree, actionSatisfied⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixThree with
    ⟨prefixTwo, pointRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixTwo with
    ⟨prefixOne, operationalRaw⟩
  rcases
      (rawSatisfies_append_iff _ _ assignment).mp prefixOne with
    ⟨_canonicalRaw, _runningRaw⟩
  have operationalSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsOperationalSampler.rows application profile frame)
      assignment).mp operationalRaw
  have pointSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsPiRlcPointRows.rows application profile frame)
      assignment).mp pointRaw
  have outputSatisfied :=
    (translate_satisfies_iff application frame
      (ConcreteNifsOutputRows.rows application profile frame)
      assignment).mp outputRaw
  have occurrenceSatisfied :=
    operationalOccurrence_satisfied application profile frame assignment
      operationalSatisfied
  exact
    ConcreteNifsOutputDecode.output_decodes_selectedResult_of_rows
      application profile frame assignment running fresh proof
      (ConcreteNifsRawProgram.actionBase application profile frame)
      constantWire decodedInputs occurrenceSatisfied actionSatisfied
      pointSatisfied outputSatisfied

/-- **Premise-free raw call refinement.** Raw row satisfaction fixes both the
frozen selected call result and the complete physical output encoding. -/
theorem call_result_and_output_of_rawRows
    (prime :
      Nightstream.Implementation.R1CS.EuclidPrime
        Nightstream.Implementation.R1CS.goldilocksP)
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
    (constantWire : assignment frame.one = 1)
    (decodedInputs :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows application profile frame)
        assignment) :
    ∃ output :
        SelectedRunning shape publicRingColumns publicFits verifierRows,
      callEval Selected Call.nifsVerify
          (.cons running (.cons fresh (.cons proof .nil))) =
        some (.cons output .nil) ∧
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil) := by
  let output :=
    SelectedRunning.ofResult
      (FixedActive.resultOf
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate)
  have decodedOutput :
      frame.outputs.Decodes (FamilyFor application) assignment
        (.cons output .nil) := by
    simpa [output] using
      output_decodes_selectedResult_of_rawRows
        application profile frame assignment running fresh proof
        constantWire decodedInputs satisfied
  refine ⟨output, ?_, decodedOutput⟩
  exact
    call_result_of_rawRows prime application profile frame assignment
      running fresh proof output constantWire decodedInputs decodedOutput
      satisfied

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawSemantics

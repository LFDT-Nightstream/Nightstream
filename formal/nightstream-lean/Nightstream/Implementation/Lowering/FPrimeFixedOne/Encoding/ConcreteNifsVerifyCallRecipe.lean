import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawSemantics

/-!
Contract: the complete activation-aware `CallRecipe` for the selected
fixed-one `nifsVerify` call.

All rows, temporary coordinates, costs, and completion functions come from
the Lean-owned selected verifier.  Active soundness reaches the deterministic
frozen call result.  Honest active completeness starts from authoritative
encoded operands and output; inactive satisfiability leaves every visible
coordinate arbitrary.

The certification package contains only primitive arithmetic support and
static footprint alignment.  It contains no accepted proposition, verifier
result, output equation, source-authority premise, Rust observation, or
generated-row fact.

Emits constraints: the exact activated footprint derived by
`ConcreteNifsActivatedProgram.footprint`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe

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

/-- Complete non-semantic support needed to construct the selected recipe. -/
structure Certification
    (application : Poseidon23ApplicationProfile Selected) where
  operational : ConcreteNifsOperationalProfile.Profile application
  prime : EuclidPrime goldilocksP
  field : PiRlcCanonicalCandidateHonest.FieldInverse
  footprint :
    ConcreteNifsActivatedProgram.FootprintAlignment
      application operational

private theorem residual_mem_temporaries
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : ColumnId)
    (member :
      column ∈
        ConcreteNifsActivatedProgram.residuals application profile frame) :
    column ∈ frame.temporaries.ids :=
  List.mem_of_mem_drop member

private theorem completed_changesOnly
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial middle : ColumnId → Field)
    (middleChanges :
      ChangesOnly frame.temporaries.ids initial middle) :
    ChangesOnly frame.temporaries.ids initial
      (ActivatedRawProgram.complete middle
        (ConcreteNifsRawProgram.rawRows application profile frame)
        (ConcreteNifsActivatedProgram.residuals
          application profile frame)) := by
  intro column notTemporary
  rw [ActivatedRawProgram.complete_changesOnly middle
      (ConcreteNifsRawProgram.rawRows application profile frame)
      (ConcreteNifsActivatedProgram.residuals application profile frame)
      column
      (by
        intro residualMember
        exact notTemporary
          (residual_mem_temporaries application profile frame
            column residualMember)),
    middleChanges column notTemporary]

/-- Active rows recover the exact deterministic selected call and its output
decoder, without an output value supplied as a premise. -/
theorem active_soundness
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
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
    ∃ outputs :
        Schema.Values (typeSystem Selected)
          ((signature Selected).callOutputs Call.nifsVerify),
      callEval Selected Call.nifsVerify
          (.cons running (.cons fresh (.cons proof .nil))) =
        some outputs ∧
      frame.outputs.Decodes (FamilyFor application) assignment outputs := by
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
  rcases
      ConcreteNifsRawSemantics.call_result_and_output_of_rawRows
        certificate.prime application certificate.operational frame assignment
        running fresh proof constantOne decoded rawSatisfied with
    ⟨output, evaluated, decodedOutput⟩
  exact ⟨.cons output .nil, evaluated, decodedOutput⟩

/-- A successful selected call has one activated witness that changes only
the declared temporary receipt and preserves every visible coordinate. -/
theorem active_honest_completeness
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
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
    (constantOne : assignment frame.one = 1)
    (_activeOne : assignment frame.active = 1)
    (encodedInputs :
      frame.operands.Encodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (encodedOutput :
      frame.outputs.Encodes (FamilyFor application) assignment
        (.cons output .nil))
    (evaluated :
      callEval Selected Call.nifsVerify
          (.cons running (.cons fresh (.cons proof .nil))) =
        some (.cons output .nil)) :
    ∃ completed : ColumnId → Field,
      AgreesOn frame.visibleIds assignment completed ∧
        ChangesOnly frame.temporaries.ids assignment completed ∧
        Satisfies
          (ConcreteNifsActivatedProgram.rows
            application certificate.operational frame)
          completed := by
  have fits :=
    ConcreteNifsActivatedProgram.frameFits
      application certificate.operational certificate.footprint frame
  rcases
      ConcreteNifsRawHonest.rows_honest
        certificate.prime certificate.field application
        certificate.operational frame assignment running fresh proof output
        fits constantOne encodedInputs encodedOutput evaluated with
    ⟨middle, middleAgrees, middleChanges, rawSatisfied⟩
  let completed :=
    ActivatedRawProgram.complete middle
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
  have residualAgrees :
      AgreesOn frame.visibleIds middle completed := by
    exact ActivatedRawProgram.complete_agreesOn middle
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
      frame.visibleIds
      (ConcreteNifsActivatedProgram.residuals_disjoint_visible
        application certificate.operational frame)
  have activatedRaw :
      RawSatisfies
        (ConcreteNifsActivatedProgram.rawRows
          application certificate.operational frame)
        completed := by
    exact ActivatedRawProgram.active_complete frame.active
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
      middle
      (ConcreteNifsActivatedProgram.residuals_length
        application certificate.operational certificate.footprint frame).symm
      (ConcreteNifsActivatedProgram.residuals_nodup
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals_fresh
        application certificate.operational certificate.footprint frame)
      rawSatisfied
  refine ⟨completed, agreesOn_trans middleAgrees residualAgrees,
    completed_changesOnly application certificate.operational frame
      assignment middle middleChanges, ?_⟩
  exact
    (satisfies_ownRows_iff frame.owner
      (ConcreteNifsActivatedProgram.rawRows
        application certificate.operational frame)
      completed).mpr
      (by
        simpa [ConcreteNifsActivatedProgram.rows] using activatedRaw)

/-- An inactive occurrence accepts arbitrary visible operand and output
coordinates by filling only its residual suffix. -/
theorem inactive_satisfiable
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (_constantOne : assignment frame.one = 1)
    (activeZero : assignment frame.active = 0) :
    ∃ completed : ColumnId → Field,
      AgreesOn frame.visibleIds assignment completed ∧
        ChangesOnly frame.temporaries.ids assignment completed ∧
        Satisfies
          (ConcreteNifsActivatedProgram.rows
            application certificate.operational frame)
          completed := by
  let completed :=
    ActivatedRawProgram.complete assignment
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
  have completedAgrees :
      AgreesOn frame.visibleIds assignment completed := by
    exact ActivatedRawProgram.complete_agreesOn assignment
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
      frame.visibleIds
      (ConcreteNifsActivatedProgram.residuals_disjoint_visible
        application certificate.operational frame)
  have completedChanges :
      ChangesOnly frame.temporaries.ids assignment completed := by
    intro column notTemporary
    exact ActivatedRawProgram.complete_changesOnly assignment
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
      column
      (by
        intro residualMember
        exact notTemporary
          (residual_mem_temporaries application certificate.operational
            frame column residualMember))
  have activatedRaw :
      RawSatisfies
        (ConcreteNifsActivatedProgram.rawRows
          application certificate.operational frame)
        completed := by
    exact ActivatedRawProgram.inactive_complete frame.active
      (ConcreteNifsRawProgram.rawRows
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals
        application certificate.operational frame)
      assignment
      (ConcreteNifsActivatedProgram.residuals_length
        application certificate.operational certificate.footprint frame).symm
      (ConcreteNifsActivatedProgram.residuals_nodup
        application certificate.operational frame)
      (ConcreteNifsActivatedProgram.residuals_fresh
        application certificate.operational certificate.footprint frame)
      activeZero
      (ConcreteNifsActivatedProgram.active_not_residual
        application certificate.operational frame)
  refine ⟨completed, completedAgrees, completedChanges, ?_⟩
  exact
    (satisfies_ownRows_iff frame.owner
      (ConcreteNifsActivatedProgram.rawRows
        application certificate.operational frame)
      completed).mpr
      (by
        simpa [ConcreteNifsActivatedProgram.rows] using activatedRaw)

/-- Exact selected `nifsVerify` row function before it is packaged with the
remaining recipe certificates. -/
def recipeRows
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
    {context : Schema (typeSystem Selected)}
    {references :
      Refs (typeSystem Selected) context
        ((signature Selected).callInputs Call.nifsVerify)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify references) :
    List OwnedRow := by
  cases references with
  | cons runningRef tail =>
      cases tail with
      | cons freshRef tail =>
          cases tail with
          | cons proofRef tail =>
              cases tail
              exact ConcreteNifsActivatedProgram.rows
                application certificate.operational frame

/-- Complete selected `nifsVerify` physical recipe. -/
def recipe
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application) :
    CallRecipe (signature Selected) (FamilyFor application)
      Call.nifsVerify := by
  refine
    { rows := recipeRows application certificate
      rowCount := ?_
      rowsOwned := ?_
      rowIdsNodup := ?_
      rowsSupported := ?_
      activeSoundness := ?_
      activeHonestCompleteness := ?_
      inactiveSatisfiable := ?_ }
  · intro context references frame
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                exact ConcreteNifsActivatedProgram.rows_length
                  application certificate.operational certificate.footprint
                  frame
  · intro context references frame row member
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                exact ConcreteNifsActivatedProgram.rows_owned
                  application certificate.operational frame row member
  · intro context references frame
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                exact ConcreteNifsActivatedProgram.rowIds_nodup
                  application certificate.operational frame
  · intro context references frame row member column columnMember
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                exact ConcreteNifsActivatedProgram.rows_supported
                  application certificate.operational certificate.footprint
                  frame row member column columnMember
  · intro context references frame assignment inputs
      constantOne activeOne decoded satisfied
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                cases inputs with
                | cons running inputs =>
                    cases inputs with
                    | cons fresh inputs =>
                        cases inputs with
                        | cons proof inputs =>
                            cases inputs
                            exact active_soundness application certificate
                              frame assignment running fresh proof constantOne
                              activeOne decoded satisfied
  · intro context references frame assignment inputs outputs
      constantOne activeOne encodedInputs encodedOutput evaluated
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                cases inputs with
                | cons running inputs =>
                    cases inputs with
                    | cons fresh inputs =>
                        cases inputs with
                        | cons proof inputs =>
                            cases inputs
                            cases outputs with
                            | cons output outputs =>
                                cases outputs
                                exact active_honest_completeness
                                  application certificate frame assignment
                                  running fresh proof output constantOne
                                  activeOne encodedInputs encodedOutput
                                  evaluated
  · intro context references frame assignment constantOne activeZero
    cases references with
    | cons runningRef tail =>
        cases tail with
        | cons freshRef tail =>
            cases tail with
            | cons proofRef tail =>
                cases tail
                exact inactive_satisfiable application certificate frame
                  assignment constantOne activeZero

/-- Packaging the selected verifier does not change its named row function. -/
@[simp] theorem recipe_rows
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
    {context : Schema (typeSystem Selected)}
    {references :
      Refs (typeSystem Selected) context
        ((signature Selected).callInputs Call.nifsVerify)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify references) :
    (recipe application certificate).rows frame =
      recipeRows application certificate frame :=
  rfl

/-- The selected recipe emits exactly the activated row program. -/
@[simp] theorem rows_exact
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (recipe application certificate).rows frame =
      ConcreteNifsActivatedProgram.rows
        application certificate.operational frame :=
  rfl

/-- The selected receipt is nonoptional and contains exactly the call-frame
allocations and the Lean-derived activated row list. -/
theorem receipt_exact
    (application : Poseidon23ApplicationProfile Selected)
    (certificate : Certification application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (recipe application certificate).receipt frame =
      { outputBundles := frame.outputs.portColumns
        temporaryBundles := frame.temporaries.bundleColumns
        rows := ConcreteNifsActivatedProgram.rows
          application certificate.operational frame } :=
  CallRecipe.receipt_exact (recipe application certificate) frame

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsVerifyCallRecipe
